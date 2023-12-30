import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

CLASSNAMES = []
global top6_inps
global TARGET
global D_TRANSFORM

from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from functions import selected_confidient_samples_ours, kl_div_loss, ternary_plot, plot_img
from functions import save_pil_plot, tensor_to_pil_image
import copy

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def double_augmentations(tensor, n_views=5):
    """
    Args:
        tensor (3,224,224): Input Image tensor
        n_views (int): Number of views to be generated
    Returns:
        tensors [(orig_view + n_views), 3,224,224]: Multiple views of an image
    """
    # img = transforms.ToPILImage()(tensor)
    # img = tensor_to_pil_image(tensor)
    images = D_TRANSFORM(tensor, again_transform=False, n_views=n_views)
    tensors = torch.stack(images, dim=0)
    return tensors

def print_plot_preds(output, inputs, plot=True, print_msg='Logits', print_loss=True):
    logit = output - output.logsumexp(dim=-1, keepdim=True)
    avg_logit = logit.logsumexp(dim=0) - np.log(logit.shape[0])
    print(print_msg)
    for i in range(3): #plotting first 3 images
        print(torch.topk(torch.exp(logit[i]), k=3, largest=True).values.tolist(), torch.topk(torch.exp(logit[i]), k=3, largest=True).indices.tolist())    
        plot_img(inputs[i], save_path= f'plots/examples/{print_msg}_{i+1}_Aug.png', target= CLASSNAMES[TARGET.item()], predicted= CLASSNAMES[torch.topk(torch.exp(logit[i]), k=3, largest=True).indices[0].item()])
    print(torch.topk(torch.exp(avg_logit), k=3, largest=True).values.tolist(), torch.topk(torch.exp(avg_logit), k=3, largest=True).indices.tolist())
    for i in range(3):
        if print_loss:
            print("H1 Loss:", avg_entropy(output[i]).item())
            print("H2 Loss:", data_uncertainity(output[i]).item())

#FIXME: Actual TPT STARTS here ---------------------------------------------------------------------------||^

def select_confident_samples(logits, top): #FIXME: 10% (top=0.1) confident views of the total augmented views to be selected
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1) # H(P1), H(P2), ..., H(Pn) #FIXME: Entropy of each of the 64 views
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)] # Filter the best 6 views
    return logits[idx], idx #shapes: (6,1000), and 6. Return best 6 logits

def avg_entropy(outputs, plot=True): # Total Uncertainty = H[E(Pi)]
    # majority_classes = torch.argmax(outputs, dim=1) 
    # class_counts = torch.bincount(majority_classes)
    # majority_class = torch.argmax(class_counts)
    # valid_rows = (majority_classes == majority_class)
    # outputs = outputs[valid_rows]
    
    
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]. Filtered logits. Representing in probability distribution/space
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]. Averaging filtered logits
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    
    #---
    # print("Logits:")
    # for i in range(len(logits[0:3])):
    #     print(torch.topk(torch.exp(logits[i]), k=3, largest=True).values.tolist(), torch.topk(torch.exp(logits[i]), k=3, largest=True).indices.tolist())    
    #     if plot:
    #         plot_img(top6_inps[i], save_path= f'plots/examples/Before_Update_Top_{i+1}_Aug.png', target= CLASSNAMES[TARGET.item()], predicted= CLASSNAMES[torch.topk(torch.exp(logits[i]), k=3, largest=True).indices[0].item()])
    
    # print("Avg_logits:")
    # print(torch.topk(torch.exp(avg_logits), k=3, largest=True).values.tolist(), torch.topk(torch.exp(avg_logits), k=3, largest=True).indices.tolist())
    #---    
    
    # if (torch.topk(torch.exp(avg_logits), k=3, largest=True).indices[0].item()) == (TARGET.item()):
    #     print("Correct Pred")
    # else:
    #     print("Wrong Pred")
    
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1) # Computing Self-Entropy of averaged logits

def data_uncertainity(outputs): # Data Uncertainty = E[H(Pi)]
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logits = outputs.log_softmax(dim=1) [N, 1000]. Filtered logits
    entropy_per_set = -(logits * torch.exp(logits)).sum(dim=-1) # entropy for each set of logits
    avg_entropy = entropy_per_set.mean(dim=0) # mean entropy across all sets 
    return avg_entropy
    # return entropy_per_set.min()

def model_uncertainity(outputs, plot=True): # Model Uncertainty = Total Uncertainty - Data Uncertainty = H[E(Pi)] - E[H(Pi)]
    return avg_entropy(outputs, plot) - data_uncertainity(outputs)

def test_time_tuning(model, inputs, optimizer, scaler, args):
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    
    selected_idx = None
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            else:
                output = model(inputs) #FIXME: output.shape = torch.Size([64, 1000])
                # ternary_plot(output, save_path= 'plots/simplex/Before_Update.png')
            if selected_idx is not None: 
                output = output[selected_idx] #FIXME: Now, output.shape = torch.Size([6, 1000])
            else:
                output, selected_idx = select_confident_samples(output, top=args.selection_p)
                # output, _ , selected_idx, norm = selected_confidient_samples_ours(output, top= 6, reduction= 'sum')
            
            #--
            if args.double_aug == True:
                global top6_inps
                top6_inps = inputs[selected_idx] # (For plotting top-k augs) # Select top6 inputs
                top6xN_inps = []
                for i in range(top6_inps.shape[0]): # top6_inps.shape[0] = 6
                    new_images = double_augmentations(top6_inps[i], n_views=5) # new_images.shape = torch.Size([64, 3, 224, 224])
                    top6xN_inps.append(new_images)
                top6xN_inps = torch.cat(top6xN_inps, dim=0)  # Double augmented images # 6 x 64 images  
                
                # print_plot_preds(output, top6_inps, plot=True, print_msg='Logits Before Second Filtering:')
                
                output = model(top6xN_inps) #FIXME: outputs.shape = torch.Size([384, 1000])
                output, selected_idx = select_confident_samples(output, top=6)
                inputs = top6xN_inps[selected_idx]
                
                # print_plot_preds(output, inputs, plot=True, print_msg='Logits After Second Filtering:')
            #--
            
            if args.double_aug == False:
                output = output.float()
                # print("(2)\nLogits Before Update:")
                loss = avg_entropy(output) # Loss = To Minimize (Self-Entropy of averaged logits)
                # print("H12 Loss Before Update:", loss.item())
                # kl_matrix, loss = kl_div_loss(output= output, reduction = 'sum', symmetric= True)
        
        if args.aug == False: #when not using aug, update the loss/model
            # Backward pass
            optimizer.zero_grad() # Zero the gradients
            scaler.scale(loss).backward() # compute gradient and do SGD step
            scaler.step(optimizer) # Unscales the gradients of optimizer's assigned params in-place
            scaler.update() # Update weights
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
        
        # ---
        # inputs = inputs[selected_idx] #FIXME: Now, inputs.shape = torch.Size([6, 3, 224, 224])
        # final_input = inputs[0].unsqueeze(0) #Selecting original input
        
        if args.aug == True:
            if args.majority_vote == True:
                #Block for filtering 6-k augs preds
                majority_classes = torch.argmax(output, dim=1) 
                class_counts = torch.bincount(majority_classes)
                majority_class = torch.argmax(class_counts)
                valid_rows = (majority_classes == majority_class)
                output = output[valid_rows]
            
            final_output = output.mean(0).unsqueeze(0) #For Top6 Avg
            # final_output = output[0].unsqueeze(0) #For Top1
        
        """---"""
        
        # output2 = model(inputs)
        # # # ternary_plot(output2, save_path= 'plots/simplex/After_Update.png')
        
        # outputs2 = output2[selected_idx]
        # # print("(3)\nLogits After Update:")
        # # loss2 = avg_entropy(outputs2)
        # # print("H1 Loss After Update:", loss2.item())
        
        # logits2 = outputs2 - outputs2.logsumexp(dim=-1, keepdim=True)
        # # avg_logits2 = logits2.logsumexp(dim=0) - np.log(logits2.shape[0])
        # print("Logits After Update:")
        # for i in range(len(logits2[0:3])):
        #     print(torch.topk(torch.exp(logits2[i]), k=3, largest=True).values.tolist(), torch.topk(torch.exp(logits2[i]), k=3, largest=True).indices.tolist())    
        #     plot_img(top6_inps[i], save_path= f'plots/examples/After_Update_Top_{i+1}_Aug.png', target= CLASSNAMES[TARGET.item()], predicted= CLASSNAMES[torch.topk(torch.exp(logits2[i]), k=3, largest=True).indices[0].item()])
             
        # print("avg_Logits2 After Update:")
        # print(torch.topk(torch.exp(avg_logits2), k=3, largest=True).values.tolist(), torch.topk(torch.exp(avg_logits2), k=3, largest=True).indices.tolist())
        
        # print("H12 Loss After Update:", loss2.item())
        # ---
        
    if args.aug == False:
        return

    # final_input = inputs[0].unsqueeze(0) #FIXME (For Top1 Input): Now after selecting aug input with least entropy, final_input.shape = torch.Size([1, 3, 224, 224])
    # final_input = final_input #(For Top6 Input)
    # return final_input
    else:
        return final_output
    
def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    assert args.gpu is not None
    main_worker(args.gpu, args)

@torch.enable_grad()
def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))
    
    # from transformers import AutoProcessor, Blip2ForConditionalGeneration
    # def init_blip():
    #     blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    #     blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16) 
    #     blip_model.to(args.device)
        

    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes
        print('len(classnames)', len(classnames))
    if args.cocoop:
        pass
    else:
        if args.lora_encoder == 'prompt': 
            from clip.custom_clip_old import get_coop
            model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
        else:
            from clip.custom_clip import get_coop
            model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init, layer_range=args.layer_range, init_method=args.init_method, lora_encoder=args.lora_encoder)
        print("Model loaded")
        if args.load is not None: 
            print("Use pre-trained soft prompt (CoOp) as initialization")
            pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
            print('COOP pretrained_ctx:', pretrained_ctx)
            assert pretrained_ctx.size()[0] == args.n_ctx
        model_state = None

    lora_enc = 'text_encoder'
    if args.lora_encoder == 'text':
        lora_enc = 'text_encoder'
    elif args.lora_encoder == 'image':
        lora_enc = 'image_encoder'
        
    for name, param in model.named_parameters():
        if not args.cocoop:
            # if ("text_encoder" in name and ("lora_A" in name or "lora_B" in name)):
            if args.lora_encoder == 'prompt':
                if ("prompt_learner" in name):
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)    
            elif (lora_enc in name and ("lora_A" in name or "lora_B" in name) \
                and any(f"layers.{i}." in name for i in range(args.layer_range[0], args.layer_range[1] + 1))):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        else:
            if "text_encoder" not in name:
                param.requires_grad_(False)
    print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    if args.cocoop:
        pass
    else:
        if args.lora_encoder == 'prompt':
            trainable_param = model.prompt_learner.parameters() #TODO: Defining Optimizer
            optimizer = torch.optim.AdamW(trainable_param, args.lr)
        else:
        # For new CLIP
            parameters_to_optimize = []
            if args.lora_encoder == 'text':
                layers = model.text_encoder.text_model.encoder.layers
            elif args.lora_encoder == 'image':
                layers = model.image_encoder.vision_model.encoder.layers
                
            for i, layer in enumerate(layers):
                if args.layer_range[0] <= i <= args.layer_range[1]:
                    lora_A_params_q = layer.self_attn.q_proj.lora_A.parameters()
                    lora_B_params_q = layer.self_attn.q_proj.lora_B.parameters()
                    
                    lora_A_params_v = layer.self_attn.v_proj.lora_A.parameters()
                    lora_B_params_v = layer.self_attn.v_proj.lora_B.parameters()

                    parameters_to_optimize.extend([
                        {'params': lora_A_params_q},
                        {'params': lora_B_params_q},
                        {'params': lora_A_params_v},
                        {'params': lora_B_params_v},
                    ])

            print('len(parameters_to_optimize)', len(parameters_to_optimize))
            optimizer = torch.optim.AdamW(parameters_to_optimize, lr=args.lr)

        optim_state = deepcopy(optimizer.state_dict())

# model.text_encoder.text_model.encoder.layers[0].self_attn.v_proj.lora_A.default.weight

# Above and below both are same

# model.text_encoder.text_model.encoder.layers[0].self_attn.v_proj.lora_A.parameters()
# for param in x:
#     print(param.data) 

# But optimizer.param_groups[0]['params'][0] is being updated which is not same as above

    scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    print('=> Using native Torch AMP. Training in mixed precision.')
    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    datasets = args.test_sets.split("/")
    results = {}
    for set_id in datasets:
        print(set_id)
        if args.tpt:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC, antialias=True),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                            augmix=len(set_id)>1) #TODO: Augmentation here
            batchsize = 1
        else:
            data_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.batch_size

        global D_TRANSFORM
        D_TRANSFORM = data_transform
        
        print("evaluating: {}".format(set_id))
        if len(set_id) > 1: 
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all
        if args.cocoop:
            pass
        else:
            model.reset_classnames(classnames, args.arch)

        global CLASSNAMES
        CLASSNAMES = classnames
        # val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode) 
        val_dataset = build_dataset(set_id= set_id, transform= data_transform, args= args)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
        
        results[set_id] = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args)
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))




    print("======== Result Summary ========")
    print("params: nstep	lr	bs")
    print("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        print("{}".format(id), end="	")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="	")
    print("\n")
    
    
    

@torch.enable_grad()
def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    # model.train()
    if not args.cocoop: # no need to reset cocoop because it's fixed
        with torch.no_grad():
            if args.lora_encoder == 'prompt':
                model.reset() #for promptlearner class
            else:
                model.LoRA_reset()
    end = time.time()
    
    initial_weights = {name: param.clone() for name, param in model.named_parameters()}
    # For some reason,, only layer 11's weights are being updated
    
    # init_args = args
    for i, (images, target) in enumerate(val_loader): #FIXME: at one loop, processing one image, i.e., its +63 (augmented) variation in total 
        # args = init_args
        assert args.gpu is not None
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0] #TODO: The first actual image
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        target = target.cuda(args.gpu, non_blocking=True) #FIXME: Actual label of the actual input image
        if args.tpt:
            images = torch.cat(images, dim=0)


        #---
        global TARGET
        TARGET = target
        # with torch.no_grad():
        #     with torch.cuda.amp.autocast():
        #         if args.cocoop:
        #             output = model((image_feature, pgen_ctx))
        #         else:
        #             output = model(image)

        # print("\n")
        # print("H1 Loss Before Update (Inference):", avg_entropy(output, plot=False).item())       
                 
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # acc_init = acc1.tolist()[0]
        
        # print("(1)")
        # print("Target:", target)
        # print("Acc_init:", acc_init)
        # logits = output - output.logsumexp(dim=-1, keepdim=True)
        # print("Logits Before update (Inferencing):")
        # print(torch.topk(torch.exp(logits), k=3, largest=True).values.tolist(), torch.topk(torch.exp(logits), k=3, largest=True).indices.tolist())
        # plot_img(image, save_path= 'plots/examples/Before_Update(Input).png', target= CLASSNAMES[target.item()], predicted= CLASSNAMES[torch.topk(torch.exp(logits), k=3, largest=True).indices[0][0].item()])
        # print("INIT-------")
        
        # print("H12 Loss Before Update (Inference):", avg_entropy(output, plot=False).item())
        #---

        if not args.cocoop: # no need to reset cocoop because it's fixed
            if args.tta_steps > 0:
                with torch.no_grad():
                    if args.lora_encoder == 'prompt':
                        model.reset() #for promptlearner class
                    else:
                        model.LoRA_reset()
            optimizer.load_state_dict(optim_state)
            if args.aug == False:
                test_time_tuning(model, images, optimizer, scaler, args) #FIXME: The proposed test-time prompt tuning
            #--
            # final_input = test_time_tuning(model, images, optimizer, scaler, args)
            else:
                final_output = test_time_tuning(model, images, optimizer, scaler, args)
            #--
            
            if args.test_aug ==True:
                test_args = copy.deepcopy(args)
                test_args.majority_vote = False #FIXME: Majority vote has to be switched manually (not fixed in argeparse call) 
                test_args.double_aug = False
                test_args.aug = True
                final_output = test_time_tuning(model, images, optimizer, scaler, test_args)
        else:
            pass

        #FIXME: Infernce
        if args.tpt:
            if args.cocoop:
                image_feature = image_feature[0].unsqueeze(0)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if args.cocoop:
                    output = model((image_feature, pgen_ctx))
                else:
                    if args.aug == False:
                        output = model(image)
                    #--
                    else:
                    # output = model(final_input)
                        output = final_output
                    #--

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        # # ---
        # print("H12 Loss After Update (Inference):", avg_entropy(output, plot=False).item())       
        
        # print("(4)")
        # print("Target:", target)
        # print("Acc1:", acc1)
        # logits = output - output.logsumexp(dim=-1, keepdim=True)
        # print("Logits After update (Inferencing):")
        # print(torch.topk(torch.exp(logits), k=3, largest=True).values.tolist(), torch.topk(torch.exp(logits), k=3, largest=True).indices.tolist())
        # plot_img(image, save_path= 'plots/examples/After_Update(Input).png', target= CLASSNAMES[target.item()], predicted= CLASSNAMES[torch.topk(torch.exp(logits), k=3, largest=True).indices[0][0].item()])
        
        # print("H12 Loss After Update (Inference):", avg_entropy(output, plot=False).item())       
        # print(f"Target: {CLASSNAMES[target.item()]}, Pred: {CLASSNAMES[torch.topk(torch.exp(logits), k=3, largest=True).indices[0][0].item()]}")
        
        
        # print("------------------------------------>>>\n\n")
        
        # if (acc_init != acc1.tolist()[0]):
        #     print("Accuracy CHANGED !!!")
        #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")
        # elif (acc_init == acc1.tolist()[0]) and (int(acc_init)==0):
        #     print("Accuracy NOT CHANGED !!!")
        #     # break
        #     print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n\n")
        # ---
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i+1) % args.print_freq == 0:
            progress.display(i)
    progress.display_summary()
    return [top1.avg, top5.avg]


if __name__ == '__main__':
    default_data_root = '/home/raza.imam/Documents/TPT/datasets/'
    default_test_sets = 'flower102/DTD/Pets/Cars' #'A/V/R/K' #flower102/DTD/Pets/UCF101/Caltech101/Aircraft/eurosat/Cars/Food101/SUN397
    default_arch = 'ViT-B/16' #ViT-B/16 #RN50
    default_bs = 64
    default_ctx_init = 'a_photo_of_a' 
    default_lr = 5e-3
    default_tta_steps = 1
    default_print_frq = 10
    default_images_per_class = None
    default_gpu = 0
    default_selection_p = 0.1 #0.1=6. 1.0=64
    default_layer_range = [9, 11]
    default_init_method = 'xavier'
    default_lora_encoder = 'prompt'
    default_aug = True #simple augmentation
    default_majority_vote = False #majority vote (with args.aug=True)
    default_double_aug = False #double-step augmentation
    default_test_aug = False #use augmented samples for inferencing trained model

    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', nargs="?", default=default_data_root, help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default=default_test_sets, help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default=default_arch)
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=default_bs, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=default_lr, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=default_print_frq, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=default_gpu, type=int, help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=True, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=default_selection_p, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=default_tta_steps, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens') 
    parser.add_argument('--ctx_init', default=default_ctx_init, type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0) #No modify need
    
    parser.add_argument('--images_per_class', default=default_images_per_class, type=int, help='Number fo images per class to load (should be <=10)')
    parser.add_argument('--layer_range', nargs=2, type=int, default=default_layer_range, help='inclusive range of layers to include for lora_A and lora_B. Default is [0, 11].')
    parser.add_argument('--init_method', default=default_init_method, choices=['xavier', 'gaussian', 'kaiming', None], help='Initialization method for LoRA weights (None=in_built xavier)')
    parser.add_argument('--lora_encoder', default=default_lora_encoder, choices=['text', 'image', 'prompt'], help='Which encoder to apply LoRA on (text or image), not both for now')
    parser.add_argument('--aug', default=default_aug, choices=[True, False], help='Whether to use simple augmentation or not')
    parser.add_argument('--majority_vote', default=default_majority_vote, choices=[True, False], help='Whether to use majority vote or not during augmentation')
    parser.add_argument('--double_aug', default=default_double_aug, choices=[True, False], help='Whether to use double-step in the augmentation or not')
    parser.add_argument('--test_aug', default=default_test_aug, choices=[True, False], help='Whether to use augmented sample predictions for inferencing trained model or not')
    
    args = parser.parse_args()
    
    main()
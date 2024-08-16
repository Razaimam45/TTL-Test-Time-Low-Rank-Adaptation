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

from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
import copy
import pandas as pd


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def quartile_selection(batch_entropy, quartile=0):
    """returns indices of the desired quartile of the batch_entropy"""
    sorted_indices = torch.argsort(batch_entropy, descending=False)
    num_chunks = 8
    chunk_size = len(sorted_indices) // num_chunks
    reshaped_indices = sorted_indices[:num_chunks * chunk_size].view(num_chunks, chunk_size)
    idx = reshaped_indices[quartile]
    return idx

def select_confident_samples(logits, top): #FIXME: 10% (top=0.1) confident views of the total augmented views to be selected
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1) # H(P1), H(P2), ..., H(Pn) #FIXME: Entropy of each of the 64 views
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)] # Filter the best 6 views
    # idx = quartile_selection(batch_entropy, quartile=7)
    return logits[idx], idx

def avg_entropy(outputs, plot=True): # Total Uncertainty = H[E(Pi)] 
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]. Filtered logits. Representing in probability distribution/space
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]. Averaging filtered logits
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1) # Computing Self-Entropy of averaged logits

def data_uncertainity(outputs): # Data Uncertainty = E[H(Pi)]
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logits = outputs.log_softmax(dim=1) [N, 1000]. Filtered logits
    entropy_per_set = -(logits * torch.exp(logits)).sum(dim=-1) # entropy for each set of logits
    avg_entropy = entropy_per_set.mean(dim=0) # mean entropy across all sets 
    return avg_entropy

# Test-time Adaptation function for our proposed TTL
def test_time_tuning(model, inputs, optimizer, scaler, args):
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    
    if args.deyo_selection and args.lora_encoder != 'prompt':
        import deyo
        for j in range(args.tta_steps):
            with torch.cuda.amp.autocast():
                DeYO = deyo.DeYO(model, args, optimizer, scaler, steps=args.tta_steps, deyo_margin=args.deyo_margin, margin_e0=args.deyo_margin_e0)
                outputs, backward, final_backward = DeYO(inputs)
                # loss = DeYO(inputs)
                
        return
    
    else: #i.e., if args.lora_encoder == 'prompt' (i.e., below block will run TPT)
        selected_idx = None
        for j in range(args.tta_steps):
            with torch.cuda.amp.autocast():
                        
                # Sample Selection Block
                if args.cocoop:
                    output = model((image_feature, pgen_ctx))
                else:
                    output = model(inputs) #FIXME: output.shape = torch.Size([64, 1000])
                
                if selected_idx is not None: 
                    output = output[selected_idx] #FIXME: Now, output.shape = torch.Size([6, 1000])
                else:
                    output, selected_idx = select_confident_samples(output, top=args.selection_p)            
                                
                output = output.float()
                loss = avg_entropy(output) # Loss = To Minimize (Self-Entropy of averaged logits)                
            
            optimizer.zero_grad() # Zero the gradients
            scaler.scale(loss).backward() # compute gradient and do SGD step
            scaler.step(optimizer) # Unscales the gradients of optimizer's assigned params in-place
            scaler.update() # Update weights
                    
        return
    
def main():
    args = parser.parse_args()
    set_random_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    global visual_means, visual_vars
    
    assert args.gpu is not None
    main_worker(args.gpu, args)

@torch.enable_grad()
def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))       

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
            from clip.custom_clip import get_coop, LoRA_AB
            model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init, layer_range=args.layer_range, init_method=args.init_method, lora_encoder=args.lora_encoder)
        
        print("Model loaded")
        model_state = None

    lora_enc = 'text_encoder'
    if args.lora_encoder == 'text':
        lora_enc = 'text_encoder'
    elif args.lora_encoder == 'image':
        lora_enc = 'image_encoder'
        
    for name, param in model.named_parameters():
        if not args.cocoop:
            if args.lora_encoder == 'prompt': # (Just prompt learner) 
            # if args.lora_encoder == 'prompt' or args.lora_encoder == 'image': # (Prompt learner + Image encoder)
                if ("prompt_learner" in name):
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)    
            elif (lora_enc in name and ("lora_A" in name or "lora_B" in name) \
                and any(f"layers.{i}." in name for i in range(args.layer_range[0], args.layer_range[1] + 1))): # (Just Image encoder)
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        else:
            if "text_encoder" not in name:
                param.requires_grad_(False)
    
    # for name, param in model.named_parameters(): # Enabling prompt learner
    #     if ("prompt_learner" in name):
    #         param.requires_grad_(True)
    
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
        # For new CLIP (i.e., LoRA embedded CLIP)
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
                    
                    # lora_A_params_k = layer.self_attn.k_proj.lora_A.parameters()
                    # lora_B_params_k = layer.self_attn.k_proj.lora_B.parameters()

                    parameters_to_optimize.extend([
                        {'params': lora_A_params_q},
                        {'params': lora_B_params_q},
                        {'params': lora_A_params_v},
                        {'params': lora_B_params_v},
                        # {'params': lora_A_params_k},
                        # {'params': lora_B_params_k},
                    ])

            # trainable_param = model.prompt_learner.parameters() # Enabling prompt learner
            # parameters_to_optimize.extend([{'params': trainable_param}]) # Adding prompt learner
            print('len(parameters_to_optimize)', len(parameters_to_optimize))
            optimizer = torch.optim.AdamW(parameters_to_optimize, lr=args.lr)
        
        optim_state = deepcopy(optimizer.state_dict())

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
        model.reset_classnames(classnames, args.arch)

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
    if not args.cocoop: # no need to reset cocoop because it's fixed
        with torch.no_grad():
            if args.lora_encoder == 'prompt':
                model.reset() #for promptlearner class
            else:
                model.LoRA_reset()
    end = time.time()    

    for i, (images, target) in enumerate(val_loader): #FIXME: at one loop, processing one image, i.e., its +63 (augmented) variation in total 
        # args = init_args
        assert args.gpu is not None
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0] #TODO: The first actual image
        else:
            if len(images.size()) > 4:
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        target = target.cuda(args.gpu, non_blocking=True) #FIXME: Actual label of the actual input image
        if args.tpt:
            images = torch.cat(images, dim=0)

        if args.tta_steps > 0:
            with torch.no_grad():
                if args.lora_encoder == 'prompt': # i.e. TPT
                    model.reset() #for promptlearner class
                else:
                    model.LoRA_reset() #for image encoder update, i.e., TTL (Ours)
        optimizer.load_state_dict(optim_state)
        
        # Applying TTL here
        test_time_tuning(model, images, optimizer, scaler, args) #FIXME: The proposed test-time prompt tuning

        # Infernce
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(image) # Inferencing after model adaptation

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i+1) % args.print_freq == 0:
            progress.display(i)
            
    progress.display_summary()
    return [top1.avg, top5.avg]


if __name__ == '__main__':
    default_data_root = '/home/raza.imam/Documents/TPT/datasets'
    default_test_sets = 'A' #'A/V/R/K' #flower102/DTD/Pets/UCF101/Caltech101/Aircraft/eurosat/Cars/Food101/SUN397
    default_arch = 'ViT-B/16' #ViT-B/16 #RN50
    default_bs = 64
    default_ctx_init = 'a_photo_of_a' 
    default_lr = 5e-3
    default_tta_steps = 1
    default_print_frq = 10
    default_gpu = 1
    default_selection_p = 0.1 #0.1=6. 1.0=64
    default_layer_range = 9, 11
    default_init_method = 'xavier'
    default_lora_encoder = 'image'
    default_deyo_selection = True

    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', nargs="?", default=default_data_root, help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default=default_test_sets, help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default=default_arch)
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=default_bs, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=default_lr, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print_freq', default=default_print_frq, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=default_gpu, type=int, help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=True, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=default_selection_p, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=default_tta_steps, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens') 
    parser.add_argument('--ctx_init', default=default_ctx_init, type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0) #No modify need
    parser.add_argument('--images_per_class', default=None, type=int, help='Number fo images per class to load (should be <=10)')
    parser.add_argument('--layer_range', type=list_of_ints, default=default_layer_range, help='inclusive range of layers to include for lora_A and lora_B.')
    parser.add_argument('--init_method', default=default_init_method, choices=['xavier', 'gaussian', 'kaiming', 'pretrained', None], help='Initialization method for LoRA weights (None=in_built xavier)')
    parser.add_argument('--lora_encoder', default=default_lora_encoder, choices=['text', 'image', 'prompt'], help='Which encoder to apply LoRA on (text or image), not both for now')
    
    # Deyo args
    parser.add_argument('--deyo_selection', default=default_deyo_selection, help='Whether to use weighted deyo class')
    
    parser.add_argument('--aug_type', default='patch', type=str, help='patch, pixel, occ')
    parser.add_argument('--occlusion_size', default=112, type=int)
    parser.add_argument('--patch_len', default=6, type=int, help='The number of patches per row/column')
    parser.add_argument('--row_start', default=56, type=int)
    parser.add_argument('--column_start', default=56, type=int)
    parser.add_argument('--deyo_margin', default=0.5, type=float,
                        help='Entropy threshold for sample selection $\tau_\mathrm{Ent}$ in Eqn. (8)') # IMPORTANT
    parser.add_argument('--deyo_margin_e0', default=0.4, type=float, help='Entropy margin for sample weighting $\mathrm{Ent}_0$ in Eqn. (10)')
    parser.add_argument('--plpd_threshold', default=0.2, type=float,
                        help='PLPD threshold for sample selection $\tau_\mathrm{PLPD}$ in Eqn. (8)') # IMPORTANT
    parser.add_argument('--fishers', default=0, type=int)
    parser.add_argument('--filter_ent', default=0, type=int)
    parser.add_argument('--filter_plpd', default=0, type=int)
    parser.add_argument('--reweight_ent', default=1, type=int)
    parser.add_argument('--reweight_plpd', default=0, type=int)
    
    args = parser.parse_args()

    main()
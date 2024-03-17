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
global data_list
data_list = []
global clip_features
global clip_labels

from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from functions import selected_confidient_samples_ours, kl_div_loss, ternary_plot, plot_img, save_attn, plot_features
from functions import save_pil_plot, tensor_to_pil_image
import copy
import pandas as pd


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def countParams(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Params: ", total_params)
    print("Trainable Params: ", trainable_params)
    print("Trainable %: ", (trainable_params/total_params)*100)

def add_gaussian_noise(image, mean=0, std=0.25):
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, 0, 1)  # Assuming the image is normalized to [0, 1]
    return noisy_image

# def add_gaussian_noise(image_batch, mean=0, std=1, l2_radius=0.1):
#     noise = torch.randn_like(image_batch)
#     noise_norm = torch.norm(noise.view(noise.size(0), -1), dim=1, keepdim=True)
#     noise = noise * (l2_radius / noise_norm.view(-1, 1, 1, 1))
#     noisy_image_batch = image_batch + noise * std + mean
#     noisy_image_batch = torch.clamp(noisy_image_batch, 0, 1)  # Assuming the images are normalized to [0, 1]
#     return noisy_image_batch


def count_predictions(output):
    probability_tensor = F.softmax(output, dim=1)    
    predicted_classes = torch.argmax(probability_tensor, dim=1)
    
    class_counts = torch.bincount(predicted_classes, minlength=102)
    max_count, max_index = torch.max(class_counts, dim=0)
    class_counts[max_index] = 0
    second_max_count, second_max_index = torch.max(class_counts, dim=0)
    A = max_index.item()
    C_A = max_count.item()
    B = second_max_index.item()
    C_B = second_max_count.item()
    # print(f"Ground Truth:{TARGET.item()}, \
    #       Input Predcited Top1:({torch.topk(probability_tensor[0], k=2).indices[0].item()}),\
    #       Input Predcited Top2:({torch.topk(probability_tensor[0], k=2).indices[1].item()}),  \
    #       A:{A}, C_A:{C_A}, B:{B}, C_B:{C_B}")        
    global data_dict
    data_dict = {
        'Ground Truth': TARGET.item(),
        'Predicted Top1': torch.topk(probability_tensor[0], k=2).indices[0].item(),
        'Predicted Top1 Prob': round(torch.topk(probability_tensor[0], k=2).values[0].item(), 2),
        'Predicted Top2': torch.topk(probability_tensor[0], k=2).indices[1].item(),
        'Predicted Top2 Prob': round(torch.topk(probability_tensor[0], k=2).values[1].item(), 2),
        'A': A,
        'C_A': C_A,
        'B': B,
        'C_B': C_B
    }
    # if TARGET.item() == data_dict['Predicted Top2']:
    #     print(f"Ground Truth: {TARGET.item()}, Input Predcited Top1: ({torch.topk(probability_tensor[0], k=2).indices[0].item()} with Prob: {torch.topk(probability_tensor[0], k=2).values[0].item():.2f}), Input Predcited Top2: ({torch.topk(probability_tensor[0], k=2).indices[1].item()} with Prob: {torch.topk(probability_tensor[0], k=2).values[1].item():.2f})")

def pred_vs_ent(output, entropy):
    
    pred = output.softmax(0)
    data_dict = {
        'Ground Truth': TARGET.item(),
        'Predicted Top1': round(torch.topk(pred, k=2).indices[0].item(), 2),
        'Pred Prob': round(torch.topk(pred, k=2).values[0].item(), 2),
        "Entropy": entropy.item()
    }
    data_list.append(data_dict)
    del data_dict
    df_all_iterations = pd.DataFrame(data_list)
    df_all_iterations.to_csv('output_pred_vs_ent_ia.csv', index=False, header=True)

def double_augmentations(tensor, n_views=5):
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
    return logits[idx], idx #shapes: (6,1000), and 6. Return best 6 logits

def select_confident_samples_ours(inputs, outputs, model):
    gauss_inputs = add_gaussian_noise(inputs, mean=0, std=0.25) 
    gauss_outputs = model(gauss_inputs)
    idx = []
    for i, output in enumerate(outputs):
        if torch.topk(outputs[i].softmax(0), k=1).indices.item() == torch.topk(gauss_outputs[i].softmax(0), k=1).indices.item():
            # print(i, "Different Predictions")
            idx.append(i)
            # pass
        # else:
            # idx.append(i)
    idx = torch.tensor(idx)
    if torch.numel(idx) == 0:
        return outputs, None
    return outputs[idx], idx

def ce_loss_on_flipped(inputs, outputs, model):
    gauss_inputs = add_gaussian_noise(inputs, mean=1.0, std=0.1, l2_radius=0.25) 
    gauss_outputs = model(gauss_inputs)
    idx = []
    for i, output in enumerate(outputs):
        if torch.topk(outputs[i].softmax(0), k=1).indices.item() != torch.topk(gauss_outputs[i].softmax(0), k=1).indices.item():
            idx.append(i)
    idx = torch.tensor(idx)
    if torch.numel(idx) != 0:
        flipped_outs = gauss_outputs[idx]
        orig_outs = outputs[idx]
        loss = F.cross_entropy(flipped_outs, torch.argmax(orig_outs, dim=1))
        # loss = F.l1_loss(flipped_outs, orig_outs)
    else:
        loss = 0
    return loss

def non_flipped_labels(orig_outputs, noisy_ouputs):
    targets = []
    idx = []
    for i, output in enumerate(orig_outputs):
        if output.argmax() == noisy_ouputs[i].argmax():
            targets.append(output.argmax())
            idx.append(i)
    targets = torch.tensor(targets).to(orig_outputs.device)
    idx = torch.tensor(idx)
    return targets, idx

from torch.distributions.normal import Normal
def macer_loss(model, orig_inputs, orig_outputs, std=0.25, mean=0.0, 
               sigma=0.25, lbd=12.0, beta=16.0, gamma=8.0): #try mean=1.0
    
    device = orig_inputs.device
    m = Normal(torch.tensor([0.0]).to(device),
             torch.tensor([1.0]).to(device))

    noisy_inputs = add_gaussian_noise(orig_inputs, mean=mean, std=std)
    outputs = model(noisy_inputs) 
    
    targets, idx = non_flipped_labels(orig_outputs, outputs)
    if torch.numel(idx) == 0:
        loss = 0
        return loss
    outputs = outputs[idx]
    
    # Classification loss
    outputs_softmax = F.softmax(outputs, dim=-1)
    outputs_logsoftmax = torch.log(outputs_softmax + 1e-10)  # avoid nan #log-softmax
    classification_loss = F.nll_loss(
        outputs_logsoftmax, targets, reduction='sum')
    
    # Robustness loss
    beta_outputs = outputs * beta  # only apply beta to the robustness loss
    beta_outputs_softmax = F.softmax(beta_outputs, dim=-1)
    top2 = torch.topk(beta_outputs_softmax, 2)
    top2_score = top2[0]
    top2_idx = top2[1]
    indices_correct = (top2_idx[:, 0] == targets)  # G_theta

    out0, out1 = top2_score[indices_correct,
                            0], top2_score[indices_correct, 1]
    robustness_loss = m.icdf(out1) - m.icdf(out0)
    indices = ~torch.isnan(robustness_loss) & ~torch.isinf(
        robustness_loss) & (torch.abs(robustness_loss) <= gamma)  # hinge
    out0, out1 = out0[indices], out1[indices]
    robustness_loss = m.icdf(out1) - m.icdf(out0) + gamma
    robustness_loss = robustness_loss.sum() * sigma / 2
    
    # Final objective function
    loss = classification_loss + lbd * robustness_loss
    
    return loss
    

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

def model_uncertainity(outputs, plot=True): # Model Uncertainty = Total Uncertainty - Data Uncertainty = H[E(Pi)] - E[H(Pi)]
    return avg_entropy(outputs, plot) - data_uncertainity(outputs)

from torch.nn import functional as F
def distr_align_loss(out_feat, targ_feat, layers_from=0, layers_to=12, moments=5):
    '''
    A feature distibution alignment L1 loss between mean and variance of the features
    '''
    distr_loss = 0
    out_means, out_vars = out_feat
    targ_means, targ_vars = targ_feat
    transf_layers = layers_to
    for l in range(layers_from, transf_layers-1):
        out_mean, out_var = out_means[l], out_vars[l]
        targ_mean, targ_var = targ_means[l], targ_vars[l]
        distr_loss += 0.5 * F.l1_loss(out_mean, targ_mean) + 0.5 * F.l1_loss(out_var, targ_var)
    return distr_loss


def test_time_tuning(model, inputs, optimizer, scaler, args):
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    
    if args.deyo_selection:
        import deyo
        for j in range(args.tta_steps):
            with torch.cuda.amp.autocast():
                DeYO = deyo.DeYO(model, args, optimizer, scaler, steps=args.tta_steps, deyo_margin=args.deyo_margin, margin_e0=args.deyo_margin_e0)
                outputs, backward, final_backward = DeYO(inputs)
                # loss = DeYO(inputs)
                
        return
    
    # eata = False
    # if eata:
    #     from EATA import eata
    #     for j in range(args.tta_steps):
    #         with torch.cuda.amp.autocast():
    #             EATA = eata.EATA(model=model, optimizer=optimizer, steps=1)
    #             outputs = EATA(inputs) 
    #     return

    selected_idx = None
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
                       
            # Sample Selection Block
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            else:
                output = model(inputs) #FIXME: output.shape = torch.Size([64, 1000])
            # count_predictions(output)
            
            # print(TARGET.item())
            # print(torch.topk(output.softmax(1)[0], k=5).indices)
            # print(torch.topk(model(add_gaussian_noise(inputs, mean=0, std=0.5)).softmax(1)[0], k=5).indices)
            # print("\n")
            
            # output, selected_idx = select_confident_samples_ours(inputs, output, model)
            # selected_idx = None
            # ce_loss = ce_loss_on_flipped(inputs, output, model)
            # m_loss = macer_loss(model, inputs, output)
            
            if selected_idx is not None: 
                output = output[selected_idx] #FIXME: Now, output.shape = torch.Size([6, 1000])
            else:
                output, selected_idx = select_confident_samples(output, top=args.selection_p)            

            if args.double_aug == True:
                global top6_inps
                top6_inps = inputs[selected_idx] # (For plotting top-k augs) # Select top6 inputs
                top6xN_inps = []
                for i in range(top6_inps.shape[0]): # top6_inps.shape[0] = 6
                    new_images = double_augmentations(top6_inps[i], n_views=5) # new_images.shape = torch.Size([64, 3, 224, 224])
                    top6xN_inps.append(new_images)
                top6xN_inps = torch.cat(top6xN_inps, dim=0)  # Double augmented images # 6 x 64 images  
                                
                output = model(top6xN_inps) #FIXME: outputs.shape = torch.Size([384, 1000])
                output, selected_idx = select_confident_samples(output, top=6)
                inputs = top6xN_inps[selected_idx]
                            
            if args.double_aug == False:
                output = output.float()
                loss = avg_entropy(output) # Loss = To Minimize (Self-Entropy of averaged logits)
                
                # loss += ce_loss
                # loss += 0.01*m_loss
                # pred_vs_ent(output[0],loss)
                
                DISTR_ALIGN = False
                
                if DISTR_ALIGN:
                    # Only selected indexes
                    target_feat_distr = (visual_means[:,:197], visual_vars[:,:197])
                    # Taking mean of token embeddings for all 12 layers across 6 test augs. Before concat, 12X(B, N, D) = 12 X (6, 199, 768). Mean is across 6 augs, so 12 X (199, 768) = 12, 199, 768 = 12, N, D
                    out_visual_mean = torch.cat([torch.mean(res.visual_feat[:, selected_idx, :], dim=1, keepdims=True).permute(1,0,2) for res in model.image_encoder.transformer.resblocks])
                    out_visual_var = torch.cat([torch.mean(((res.visual_feat[:, selected_idx, :] - out_visual_mean[i, :, :].unsqueeze(0).permute(1,0,2))**2), dim=1, keepdims=True).permute(1,0,2) for i, res in enumerate(model.image_encoder.transformer.resblocks)])
                    out_feat_distr = (out_visual_mean, out_visual_var)
                    
                    DISTR_LOSS_W = 100.0
                    ALIGN_LAYER_FROM = 0
                    ALIGN_LAYER_TO = 3
                    DISTR_LOSS_W = DISTR_LOSS_W / (ALIGN_LAYER_TO - ALIGN_LAYER_FROM)
                    loss += DISTR_LOSS_W * distr_align_loss(out_feat_distr, target_feat_distr, 
                                                layers_from=ALIGN_LAYER_FROM, layers_to=ALIGN_LAYER_TO)
        
        if args.aug == False: #when not using aug, update the loss/model
            optimizer.zero_grad() # Zero the gradients
            scaler.scale(loss).backward() # compute gradient and do SGD step
            scaler.step(optimizer) # Unscales the gradients of optimizer's assigned params in-place
            scaler.update() # Update weights
            
        if args.aug == True:
            if args.majority_vote == True:
                #Block for filtering 6-k augs preds
                majority_classes = torch.argmax(output, dim=1) 
                class_counts = torch.bincount(majority_classes)
                majority_class = torch.argmax(class_counts)
                valid_rows = (majority_classes == majority_class)
                output = output[valid_rows]
            
            final_output = output.mean(0).unsqueeze(0) #For Top6 Avg
                
    if args.aug == False:
        return
    else:
        return final_output
    
def main():
    args = parser.parse_args()
    set_random_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    global visual_means, visual_vars
    visual_vars = torch.load('/home/raza.imam/Documents/PromptAlign/stats/ImgNet_vis_vars.pt',map_location=device).to(device)
    visual_means = torch.load('/home/raza.imam/Documents/PromptAlign/stats/ImgNet_vis_means.pt',map_location=device).to(device)
    
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
        
            lora_pretrained = False
            if lora_pretrained:
                pretrained_state_dict = torch.load('/home/raza.imam/Documents/TPT/weights/output/imagenet/MaPLe/ImageEncoder/e1_0to0_16shot_lr1e5/MultiModalPromptLearner/model.pth.tar-1')
                original_state_dict = pretrained_state_dict['state_dict']
                modified_state_dict = {}
                for key, value in original_state_dict.items():
                    # Replace '.base_layer.' with '.'
                    new_key = key.replace('.base_layer.', '.')
                    modified_state_dict[new_key] = value
                
                if 'prompt_learner.token_prefix' in modified_state_dict:
                    current_size = model.prompt_learner.token_prefix.size()
                    if current_size == modified_state_dict['prompt_learner.token_prefix'].size():
                        model.prompt_learner.token_prefix.data = modified_state_dict['prompt_learner.token_prefix'].data
                    else:
                        del modified_state_dict['prompt_learner.token_prefix']
                model.load_state_dict(modified_state_dict, strict=False)
                
                if args.lora_encoder == 'image':
                    model.LoRA_AB = LoRA_AB(model.image_encoder, layer_range=args.layer_range, init_method=args.init_method, lora_encoder=args.lora_encoder)
                if args.lora_encoder == 'text':
                    model.LoRA_AB = LoRA_AB(model.text_encoder, layer_range=args.layer_range, init_method=args.init_method, lora_encoder=args.lora_encoder)
                    
        print("Model loaded")
        if args.load is not None: 
            print("Use pre-trained soft prompt (CoOp) as initialization")
            pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
            print('COOP pretrained_ctx:', pretrained_ctx)
            assert pretrained_ctx.size()[0] == args.n_ctx
            model.prompt_learner.ctx = torch.nn.Parameter(pretrained_ctx.float()) # loading CooP weights
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
            
        # import deyo
        # model = deyo.configure_model(model)
        # params, param_names = deyo.collect_params(model)
        # optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        
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
        if args.cocoop:
            pass
        else:
            # if args.lora_encoder == "prompt":
            model.reset_classnames(classnames, args.arch)
            # else: 
            #     model.reset_classnames(classnames, model)

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
    
    raza_start_time = time.time()
    
    # targeted_class_dict = {
    #     0: "grasshopper",
    #     1: "hummingbird",
    #     2: "American robin",
    #     3: "goldfinch",
    #     4: "flatworm",
    #     5: "green iguana",
    #     6: "fox squirrel",
    #     7: "ladybug",
    #     8: "fly",
    #     9: "American bullfrog",
    #     10: "mushroom",
    #     11: "scorpion",
    #     12: "lynx",
    #     13: "junco",
    #     14: "sea anemone",
    #     15: "mantis",
    # }
    targeted_class_dict = {i: value for i, value in enumerate(CLASSNAMES)}
    clip_features = []
    clip_labels = []
    
    # init_args = args
    
    # running_mean = torch.zeros((12, 197, 768)).to(f"cuda:{args.gpu}")
    # running_var = torch.zeros((12, 197, 768)).to(f"cuda:{args.gpu}")
    # count = 0
    
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

        global data_dict
        data_dict = {}
        global TARGET
        TARGET = target
        if not args.cocoop: # no need to reset cocoop because it's fixed
            if args.tta_steps > 0:
                with torch.no_grad():
                    if args.lora_encoder == 'prompt':
                        model.reset() #for promptlearner class
                    else:
                        model.LoRA_reset()
            optimizer.load_state_dict(optim_state)
            
            # -------------------------------- ATTN --------------------------------
            # Before Update saving attention
            # grad_cams_path = "/home/raza.imam/Documents/TPT/plots/grad_cams"
            # save_attn(model.image_encoder, image, f"{grad_cams_path}/clip_bs/clip_bs_attn_{i}.png", save_img_path=None)
            # -------------------------------- ATTN --------------------------------
            
            if args.aug == False:
                # clip_bs = deepcopy(model)
                test_time_tuning(model, images, optimizer, scaler, args) #FIXME: The proposed test-time prompt tuning
                # pass
            else:
                final_output = test_time_tuning(model, images, optimizer, scaler, args)
            
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
                    pass
                else:
                    if args.aug == False:
                        output = model(image)
                    #--
                    else:
                        output = final_output
                    #--

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i+1) % args.print_freq == 0:
            progress.display(i)
            
        # Before/After
        # -------------------------------- T-SNE --------------------------------
        if CLASSNAMES[TARGET.item()] in targeted_class_dict.values():
            feat = model.image_encoder.vision_model.encoder.layers[-1].visual_feat[0] # for image
            # feat = model.image_encoder.transformer.resblocks[-1].visual_feat[0] # for prompt
            clip_labels.append(TARGET.item())
            clip_features.append(feat)
        
        if i==(len(val_loader)-1):
            all_clip_features = torch.cat(clip_features, dim=0).to('cpu').numpy()
            all_clip_labels = torch.tensor(clip_labels).to('cpu').numpy()
            plot_features(all_clip_features, all_clip_labels, len(targeted_class_dict.keys()), targeted_class_dict, f"/home/raza.imam/Documents/TPT/plots/t_sne/ttl_features_A_allclasses_rank4")
        # -------------------------------- T-SNE --------------------------------
   

        # -------------------------------- ATTN -------------------------------- 
        # if int(acc1) != 0:
        #     # save_attn(clip_bs.image_encoder, image, f"/home/raza.imam/Documents/TPT/plots/grad_cams/clip_bs/clip_bs_attn_{i}.png") # non-updated model
        #     grad_cams_path = "/home/raza.imam/Documents/TPT/plots/grad_cams"
        #     save_attn(model.image_encoder, image, f"{grad_cams_path}/ttl/ttl_attn_{i}.png", save_img_path=f"{grad_cams_path}/imgs/img_{i}.png") # updated model
        # -------------------------------- ATTN --------------------------------
        
    #     if args.lora_encoder == 'image' and int(acc1) != 0:
    #         out_visual_mean = torch.cat([torch.mean(layer.visual_feat, dim=1, keepdims=True).permute(1,0,2) for layer in  model.image_encoder.vision_model.encoder.layers])
    #         out_visual_var = torch.cat([torch.mean(((layer.visual_feat - out_visual_mean[i, :, :].unsqueeze(0).permute(1,0,2))**2), dim=1, keepdims=True).permute(1,0,2) 
    #                                     for i, layer in enumerate(model.image_encoder.vision_model.encoder.layers)])
    #     elif args.lora_encoder == 'prompt':
    #         out_visual_mean = torch.cat([torch.mean(res.visual_feat, dim=1, keepdims=True).permute(1,0,2) for res in model.image_encoder.transformer.resblocks])
    #         out_visual_var = torch.cat([torch.mean(((res.visual_feat - out_visual_mean[i, :, :].unsqueeze(0).permute(1,0,2))**2), dim=1, keepdims=True).permute(1,0,2) 
    #                                     for i, res in enumerate(model.image_encoder.transformer.resblocks)])
    #     count += 1
    #     running_mean = (running_mean * (count - 1) + out_visual_mean) / count
    #     running_var = (running_var * (count - 1) + out_visual_var) / count
    # torch.save(running_mean, "ours_mean.pt")
    # torch.save(running_var, "ours_var.pt")    
    
        # probability_tensor = F.softmax(output, dim=1)
        # data_dict['Updated Top1']=(torch.topk(probability_tensor[0], k=2).indices[0].item())
        # data_dict['Updated Top2']=(torch.topk(probability_tensor[0], k=2).indices[1].item())
        # data_list.append(data_dict)
        # del data_dict
        # df_all_iterations = pd.DataFrame(data_list)
        # df_all_iterations.to_csv('output_file5.csv', index=False, header=True)

    progress.display_summary()
    raza_end_time = time.time()
    raza_total_time = raza_end_time - raza_start_time
    print(f"Total execution time: {raza_total_time:.4f} seconds")
    return [top1.avg, top5.avg]


import math
if __name__ == '__main__':
    default_data_root = '/home/raza.imam/Documents/TPT/datasets'
    default_test_sets = 'A' #'A/V/R/K' #flower102/DTD/Pets/UCF101/Caltech101/Aircraft/eurosat/Cars/Food101/SUN397
    default_arch = 'ViT-B/16' #ViT-B/16 #RN50
    default_bs = 64
    default_ctx_init = 'a_photo_of_a' 
    default_lr = 5e-3
    default_tta_steps = 1
    default_print_frq = 10
    default_images_per_class = None
    default_gpu = 1
    default_selection_p = 0.1 #0.1=6. 1.0=64
    default_layer_range = [9, 11]
    default_init_method = 'xavier'
    default_lora_encoder = 'image'
    default_deyo_selection = False
    
    # Trivial args
    default_aug = False #simple augmentation
    default_majority_vote = False #majority vote
    default_test_aug = False #use augmented samples for inferencing trained model
    default_double_aug = False #double-step augmentation

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
    parser.add_argument('--init_method', default=default_init_method, choices=['xavier', 'gaussian', 'kaiming', 'pretrained', None], help='Initialization method for LoRA weights (None=in_built xavier)')
    parser.add_argument('--lora_encoder', default=default_lora_encoder, choices=['text', 'image', 'prompt'], help='Which encoder to apply LoRA on (text or image), not both for now')
    parser.add_argument('--aug', default=default_aug, choices=[True, False], help='Whether to use simple augmentation or not')
    parser.add_argument('--majority_vote', default=default_majority_vote, choices=[True, False], help='Whether to use majority vote or not during augmentation')
    parser.add_argument('--test_aug', default=default_test_aug, choices=[True, False], help='Whether to use augmented sample predictions for inferencing trained model or not')
    parser.add_argument('--double_aug', default=default_double_aug, choices=[True, False], help='Whether to use double-step in the augmentation or not')
    
    # Deyo args
    parser.add_argument('--deyo_selection', default=default_deyo_selection, choices=[True, False], help='Whether to use weighted deyo class')
    
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
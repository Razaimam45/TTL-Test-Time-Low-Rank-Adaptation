import torch
import torchvision.transforms as transforms
import numpy as np
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
import time
from data.fewshot_datasets import fewshot_datasets
from data.imagnet_prompts import imagenet_classes
from copy import deepcopy
from data.datautils import AugMixAugmenter, build_dataset
import os
from PIL import Image
from attack import Attack, AttackART
import os 
from torchvision.utils import save_image
from tqdm import tqdm
import clip 
import json
import pyod
from pyod.models.copod import COPOD
from torch.optim.lr_scheduler import StepLR


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from collections import Counter

from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
import torchvision.datasets as datasets


ID_to_DIRNAME={
    'I': 'ImageNet',
    'A': 'imagenet-a',
    'K': 'ImageNet-Sketch',
    'R': 'imagenet-r',
    'V': 'imagenetv2-matched-frequency-format-val',
    'flower102': 'Flower102',
    'dtd': 'DTD',
    'pets': 'OxfordPets',
    'cars': 'StanfordCars',
    'ucf101': 'UCF101',
    'caltech101': 'Caltech101',
    'food101': 'Food101',
    'sun397': 'SUN397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat'
}

def get_intermediate_patches(model, image, num_maksed_patches, num_views, initial_block, final_block, num_blocks = 12):
    patches = {} #store all patches from all blocks.
    def get_activation(name):
        def hook(model, input, output):
            patches[name] = output.detach()
        return hook
    for i in range(num_blocks):
        model.image_encoder.transformer.resblocks[i].register_forward_hook(get_activation(f'patches_block_{i}')) #register the hooks, one after each attention block
    with torch.no_grad():
        original_im_enc = model.image_encoder(image) #forward pass image to activate hooks 
    random_blocks = torch.randint(low = initial_block, high = final_block, size = (num_views,)) # [num_views]
    random_indices = torch.randint(low = 1, high = patches['patches_block_0'].shape[0], size = (num_blocks, num_maksed_patches)) #choose indices to mask out, stay away from cls ---- [12 (blocks), 20 (num_masked_patches)]

    selected_slices = [patches[f'patches_block_{i}'][random_indices[i]] for i in range(num_blocks)] #get the selected tokens 
    for i in range(num_blocks):
        selected_slices[i][:, :, :] = 0.  # Set the third dimension to 0 for selected patches in each block
        patches[f'patches_block_{i}'][random_indices[i]] = selected_slices[i]

    total_patches = []
    for block in random_blocks:
        total_patches.append(patches[f'patches_block_{block}'])

    total_patches = torch.cat(total_patches, dim=1) #will give you (197, 63,768)
    return total_patches, random_blocks, original_im_enc

def get_embedding_intermediate_features(model, random_blocks, total_patches, original_im_enc): 
    embedding_inter_features = []
    with torch.no_grad():
        for block_index, image_index in zip(random_blocks,range(total_patches.shape[1])):
            index = block_index.item()
            if index == 11:
                output = total_patches[:,image_index,:].unsqueeze(dim=1)
            else:
                output = model.image_encoder.transformer.resblocks[index+1:](total_patches[:,image_index,:].unsqueeze(dim=1))
            embedding_inter_features.append(output)
    embedding = torch.cat(embedding_inter_features, dim=1).permute(1,0,2)
    embedding = model.image_encoder.ln_post(embedding[:,0,:])
    embedding = embedding @ model.image_encoder.proj

    embedding = torch.cat((original_im_enc, embedding), dim= 0)
    # normalized features
    image_features = embedding / embedding.norm(dim=1, keepdim=True)
    text_features = model.get_text_features()

    logit_scale = model.logit_scale.exp() #fromCLIPTestTimeTuning 
    logits = logit_scale * image_features @ text_features.t()

    return logits

#for confident selection based on entropy
def select_confident_samples(logits, top, temperature=1.0):
    logits = logits/temperature
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1) #from the logits, get the entropy within each view in the batch
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)] #get the lowest top_k samples (lowest entropy)
    return logits[idx], idx #return the entropy and the views indices.

#get the avg of "self-entropy" over the views
# compute the average entropy of a set of predicted class probabilities or logits across augmented views. 
def avg_entropy(outputs): 
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    # print(avg_logits.min())
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

#find loss, do backprop, and take step to update the prompt.
def test_time_tuning(model, inputs, optimizer, scaler, args, self_augmentation): #inputs = images = one image if self_aug is true
    #dont enter here.
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    
    selected_idx = None
    for j in range(args.tta_steps): #one step only
        with torch.cuda.amp.autocast():
            ############# These lines below to obtain the output #############
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            elif (not args.cocoop) and (self_augmentation): # if self augmentation ---- Negelct this part
                inputs, random_blocks, original_im_enc = get_intermediate_patches(
                    model= model, image = inputs, num_maksed_patches= args.num_maksed_patches, 
                    num_views= args.batch_size  - 1, initial_block = args.initial_block, 
                    final_block = args.final_block) #inputs here are the samples intermediate features (63 + original)
                output = get_embedding_intermediate_features(
                    model, random_blocks= random_blocks, total_patches= inputs, 
                    original_im_enc = original_im_enc)
                # output = model(inputs) #Return logits. (after multiplication of text and image embeddings.)
            elif args.frob_original_copod_1: #COPOD on cls tokens (features not logits.) and get the truncated images
                with torch.no_grad():
                    output = model(inputs)
                range_loop_dict = {32:1, 16:2, 8:3, 6:4} #for copod
                selection_p_dict = {32:0.2, 16:0.4, 8:0.8} #for copod
                range_loop = range_loop_dict[args.number_of_inlaiers] #to get specific images from copod
                image_feaures = model.image_features #get the cls tokens.
                truncated_image_features, indices = get_copod(image_feaures, contamination= 0.5, range_loop=range_loop) #get specific number of images (not 64) (32,16,8)
                output = model.logit_scale.exp() * truncated_image_features @ model.get_text_features().t() #get the logits of the selected images.
            else:
                output = model(inputs) #Return logits. (after multiplication of text and image embeddings.)
            
            #TODO: output = logits = logit_scale * image_features @ text_features.t()
            
            ############# These lines below to obtain filtered images (filtration) #############
            if selected_idx is not None:
                output = output[selected_idx]

            elif (args.frob_on_original):
                output = output
            
            elif args.frob_original_copod_1: #COPOD on cls tokens (features not logits.)
                if args.no_filtration: #if no filtration, then take the logits of the reduced images.
                    output = output # no filter, just get the reduced views 
                elif args.our_filtration and not args.tpt_filtration: #our filtration applied on KL matrix obtained from the logits above (output)
                    output_ours, _ , selected_idx, norm = selected_confidient_samples_ours(output, top= 6, reduction= 'sum') #get the loss: frob of KL matrix
                elif args.our_filtration and args.tpt_filtration:
                    output_tpt, selected_idx = select_confident_samples(output, selection_p_dict[args.number_of_inlaiers])
                    output_ours, _ , selected_idx, norm = selected_confidient_samples_ours(output, top= 6, reduction= 'sum') #get the loss: frob of KL matrix
                else: #this line uses the original filtration (TPT)
                    output, selected_idx = select_confident_samples(output, selection_p_dict[args.number_of_inlaiers], temperature=args.temperature)
            
            elif args.similarity != None:
    
                if (args.similarity=='cosine'): #TPT with cosine similarity instead of self-entropy
                    on_inputs = False
                    if on_inputs:
                        selected_idx = select_confident_samples_cosine_on_inputs(inputs)
                        output = output[selected_idx]
                    else:
                        output, selected_idx, batch_entropy = select_confident_samples_cosine(output, selection_cosine=0.5, selection_selfentro=0.5)
                    
                elif (args.similarity=='mahalanobis'): #TPT with mahalanobis_similarity instead of self-entropy
                    output, selected_idx = select_confident_samples_mahalanobis(output)
                    
                elif (args.similarity=='ssim'): #TPT with structural_similarity instead of self-entropy
                    selected_idx = select_confident_samples_ssim(inputs)
                    output = output[selected_idx]
                    
                elif (args.similarity=='euclidean'): #TPT with euclidean_similarity instead of self-entropy
                    output, selected_idx = select_confident_samples_euclidean(output)
                    
                elif (args.similarity=='pairwise'): #TPT with pairwise_distance instead of self-entropy
                    output, selected_idx = select_confident_samples_pairwise_distance(output)
                    
                elif (args.similarity=='entropy_var'): #TPT with entropy+variance instead of self-entropy
                    output, selected_idx = select_confident_samples_entropy_var(output) 
                    
                elif (args.similarity=='cos_var'): #TPT with entropy+variance instead of self-entropy
                    output, selected_idx = select_confident_samples_cosine_var(output) 
                    
                elif (args.similarity=='kl_div'):
                    output, _ , selected_idx, norm = selected_confidient_samples_ours(output, top= 20, reduction= 'sum', no_filter=args.no_filtration)
                    
            else: #this line for original TPT
                #we are here
                output, selected_idx = select_confident_samples(output, top=0.06) #Filter the views (TPT filtration -- original filtration)
            
            ############# the lines below to obtain the loss #############
            if ((args.frob_on_z) or (args.frob_on_original)) and (not args.frob_on_z_KL_TPT_losses): #implemented but negelct this
                kl_matrix, loss = kl_div_loss(output= output, reduction = 'sum') # loss = frob norm of kl matrix
                            
            elif args.frob_on_z_KL_TPT_losses or ((args.frob_original_copod_1) and (args.ce_kl)): #This line, to combine the original loss of TPT and KL loss. 
                loss_1 = avg_entropy(output)
                _, loss_2 = kl_div_loss(output= output, reduction = 'sum')
                loss = torch.stack((loss_1, loss_2)).mean()

            elif (args.frob_original_copod_1 and (args.our_filtration or args.tpt_filtration)): #COPOD with our filtration.
                if args.tpt_filtration and args.our_filtration: #to include the tpt filtration in addition to the kl loss.
                    loss_2 = avg_entropy(output_tpt)
                    loss_1 = norm
                    # loss_1 = avg_entropy(output_ours)
                    loss = torch.stack((loss_1, loss_2)).mean()
                if args.our_filtration and not args.tpt_filtration: #if our filtration only, then the loss is the norm of kl matrix from above.
                    loss = norm
            elif (args.frob_original_copod_1) and (args.kl or args.no_filtration): 
                _, loss = kl_div_loss(output= output, reduction = 'sum')
            elif (args.frob_original_copod_1) and (args.euclidean):
                _, loss = euclidean_dist(output= output)
            elif (args.frob_original_copod_1) and (args.ce_euclidean):
                _, loss_1 = euclidean_dist(output= output)
                loss_2 = avg_entropy(output)
                loss = torch.stack((loss_1, loss_2)).mean()
            elif (args.no_filtration == True): #if there's No filtration
                loss = norm
            else: #if there's filtration
                if args.loss_type == 'norm' and args.similarity=='kl_div':
                    loss = norm
                else:
                    loss = avg_entropy(output) #avg entroy among views
        # print(output.shape, loss)
        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
    if args.cocoop:
        return pgen_ctx
    return

#get class names for the dataset
def get_class_names(test_sets):
    if test_sets in fewshot_datasets: #few_shot datasets are defined in another folder click on it.
        classnames = eval("{}_classes".format(test_sets.lower()))
    else: 
        classnames = imagenet_classes
    return classnames

#define the optimizer for the prompt.
def define_optimizer(cocoop, model, lr):
    if cocoop:
        #we wont enter here.
        optimizer = None
        optim_state = None
    #set the optimizer and give the prompt parameters only.
    else:
        #we will enter here
        trainable_param = model.prompt_learner.parameters() #which is ctx only.
        optimizer = torch.optim.AdamW(trainable_param, lr) #set the optimizer to optimize ctx embedding only
        optim_state = deepcopy(optimizer.state_dict())
    return optimizer, optim_state

#Prepare dataset
def prepare_test_sets(tpt, resolution, set_id, batch_size, self_augmentation):
    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    #enter here to get the data transformation
    if tpt:
        base_transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=BICUBIC),
            transforms.CenterCrop(resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        
        if not self_augmentation:
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=batch_size-1, 
                                            augmix=len(set_id)>1)
        elif self_augmentation:
            data_transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize])
        batchsize = 1 ################
    else:
        data_transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize,
        ])
        batchsize = batch_size
    # reset the model
    # Reset classnames of custom CLIP model
    if len(set_id) > 1: 
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
    else:
        assert set_id in ['A', 'R', 'K', 'V', 'I']
        classnames_all = imagenet_classes
        classnames = []
        if set_id in ['A', 'R', 'V']:
            label_mask = eval("imagenet_{}_mask".format(set_id.lower())) #here the classes got flipped
            if set_id == 'R':
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(classnames_all[i])
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all
    return data_transform, batchsize, classnames

#prepare dataset for attacks
def prepare_test_set_loader_attack(resolution, data_root, set_id, batch_size= 64, workers= 8):
    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    #enter here to get the data transformation
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        normalize])
    testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
    testset = datasets.ImageFolder(testdir, transform=transform)
    val_loader = torch.utils.data.DataLoader(
                    testset,
                    batch_size=batch_size, shuffle=False,
                    num_workers= workers, pin_memory=True)
    return val_loader

def generate_attack(image, label, model, args, epsilon, attack_name):
    mean= torch.tensor([0.48145466, 0.4578275, 0.40821073], device=args.gpu)
    std= torch.tensor([0.26862954, 0.26130258, 0.27577711], device=args.gpu)

    unnormalized_image = (image * std[:, None, None]) + mean[:, None, None]
    model.eval()
    label = label.cuda(args.gpu)  
    #fix all params except prompt learner. Meaning, we are not fixing ctx_vector (embedding of ctx)
    for name, param in model.named_parameters():
        #will enter here
        if not args.cocoop:
            #fix all params except prompt learner. Meaning, we are not fixing ctx_vector (embedding of ctx)
            if "prompt_learner" in name:
                param.requires_grad_(False)

    attack = Attack(epsilon= epsilon, attack_type=attack_name, model= model, 
                    bounds=(0,1), device= args.gpu, preprocess={'mean':mean, 'std': std, 'axis': -3})
    
    adv_image, successes = attack.generate_attack(unnormalized_image, label)

    # attack = AttackART(epsilon= epsilon, attack_type= attack_name, 
    #                    bounds=(min_image, max_image), device= args.gpu)
    
    # adv_image = attack.generate_adv(samples = image, labels = label, model = model)

    #fix all params except prompt learner. Meaning, we are not fixing ctx_vector (embedding of ctx)
    for name, param in model.named_parameters():
        #will enter here
        if not args.cocoop:
            #fix all params except prompt learner. Meaning, we are not fixing ctx_vector (embedding of ctx)
            if "prompt_learner" in name:
                param.requires_grad_(True)

    return adv_image, successes

def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, set_id, val_dataset, our_attack, base_dir, generate_attack_images, self_augmentation):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE) #name:time, decimalpoints, summary.none = 0
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE) #name:Acc@1, decimalpoints, summary.Average = 1
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE) #name:Acc@5, decimalpoints, summary.Average = 1

    progress = ProgressMeter(
        len(val_loader), #num_batches
        [batch_time, top1, top5], #meters
        prefix='Test: ' #prefix
        )

    # reset model and switch to evaluate mode
    model.eval()
    if not args.cocoop: # no need to reset cocoop because it's fixed
        with torch.no_grad():
            model.reset()
    end = time.time()
    classes_genrated = []
    sum_acc_1 = 0. 
    sum_acc_5 = 0.
    count = 0
    accs = {
        'top1':[], 
        'top5':[]
        } 
    
    for i, (images, target) in enumerate(val_loader): #batchsize = 64, one original and 63 views of it
        assert args.gpu is not None

        if self_augmentation: 
            if not args.cluster:
                images = images.cuda(args.gpu, non_blocking=True) #images here is just one pic if self_aug is True
                image = images.clone() #one image as well
            else: 
                images = images.cuda(non_blocking=True) #images here is just one pic if self_aug is True
                image = images.clone() #one image as well
        
        else:
            if isinstance(images, list):
                for k in range(len(images)):
                    if not args.cluster:
                        images[k] = images[k].cuda(args.gpu, non_blocking=True)
                    else: 
                        images[k] = images[k].cuda(non_blocking=True)
                image = images[0]
            else:
                if len(images.size()) > 4:
                    # when using ImageNet Sampler as the dataset
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                if not args.cluster:
                    images = images.cuda(args.gpu, non_blocking=True)
                else: 
                    images = images.cuda(non_blocking=True)
                image = images
        # target = get_actual_label_one_sample(val_dataset.class_to_idx, target).cuda(args.gpu, non_blocking=True)
        if not args.cluster:
            target = target.cuda(args.gpu, non_blocking=True)
        else: 
            target = target.cuda(non_blocking=True)
            
        if args.tpt:
            if not self_augmentation:
                images = torch.cat(images, dim=0) #make them 64,3,224,224

        # reset the tunable prompt to its initial state.
        if not args.cocoop: # no need to reset cocoop because it's fixed
            if args.tta_steps > 0:
                with torch.no_grad():
                    model.reset() #get embedding of initial promp, get copy of it
            optimizer.load_state_dict(optim_state)
            test_time_tuning(model, images, optimizer, scaler, args, self_augmentation) #take step for the prompt. if self_augmentations = Trrue, then the image is just one pic   FIRST FORWARD
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_feature, pgen_ctx = model.gen_ctx(images, args.tpt)
            optimizer = None
            pgen_ctx = test_time_tuning(model, (image_feature, pgen_ctx), optimizer, scaler, args) #first forward

        # The actual inference goes here, after taking a step, perform the inference.
        if args.tpt:
            if args.cocoop:
                image_feature = image_feature[0].unsqueeze(0)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if args.cocoop:
                    output = model((image_feature, pgen_ctx))
                else:
                    #we are here
                    output = model(image) #get logits,  wont have grad SECOND FORWARD
        
        if our_attack and generate_attack_images:

            adv_image, succes_adv = generate_attack(image= image, 
                                                    label= target, 
                                                    model= model, 
                                                    args= args, epsilon=  args.epsilon, 
                                                    attack_name= args.attack_name)
            
            # if succes_adv == True:
            image_name = f"{i}.png"
            actual_class = next((key for key, value in val_dataset.class_to_idx.items() if value == target.item()), None)
            classes_genrated.append(actual_class)
            save_image(adv_image, os.path.join(base_dir, actual_class, image_name))
            
            if ((i+1) % 200) == 0:
                print(f'Num of generated successful attacks so far = ({len(classes_genrated)} out of {i+1} original samples)')
                # print(f"Generated attacks original classes",'\n', f'{Counter(classes_genrated)}')
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
                
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        count+=1
        sum_acc_1 += acc1[0]
        sum_acc_5 += acc5[0]
        avg_acc_1_final = sum_acc_1 / count
        avg_acc_5_final = sum_acc_5 / count

        if (i+1) % args.print_freq == 0:
            progress.display(i)
            accs['top1'].append(avg_acc_1_final)
            accs['top5'].append(avg_acc_5_final)
            # with open(f'./exps_output/{args.initial_block}-{args.final_block}Blocks_{args.num_maksed_patches}MP-{args.attack_folder_name}.json', 'w') as json_file:
                # json.dump(accs, json_file)
    progress.display_summary()

    return [top1.avg, top5.avg]

def create_attack_folders(base_directory, dataset_id, n_classes, steps_our, creat_fold, attack_name):
    base_directory = os.path.join(base_directory, ID_to_DIRNAME[dataset_id]+f'-attack-{steps_our}steps-{attack_name}')

    if creat_fold:
        for i in range(n_classes):
            folder_name = str(i)
            folder_path = os.path.join(base_directory, folder_name)
            
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        print("Folders created successfully.")

    return base_directory

def zeroshot_classifier(classnames, templates, model, device, ensemble = True):
    if ensemble:
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template.format(classname) for template in templates] #format with class
                texts = clip.tokenize(texts).to(device) #tokenize
                class_embeddings = model.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        return zeroshot_weights
    else: 
        with torch.no_grad():
            texts = [f'a photo of a {class_}' for class_ in classnames ] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            return class_embeddings.t()
        

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy() for k in topk]

def get_actual_label_one_sample(dictionary, value):
    return torch.tensor([int([key for key, val in dictionary.items() if val == value][0])])

def get_actual_label_batch(dictionary, targets):
    keys = torch.ones_like(targets)
    for target in range(targets.shape[0]):
        for key, val in dictionary.items():
            if targets[target] == val:
                keys[target] = int(key)
    return keys

def kl_div_loss(output, reduction = 'batchhmean', symmetric = False): 
    epsilon = 1e-10
    kl_div_matrix = torch.zeros(output.shape[0], output.shape[0]).to(output.device)
    output = torch.softmax(output, dim=-1)
    output  = torch.clamp(output, min= epsilon)
    for i in range(output.shape[0]):
        for j in range(output.shape[0]):
            kl_div = torch.nn.functional.kl_div(output[i].log(), output[j], reduction=reduction)
            kl_div_matrix[i, j] = kl_div
    if symmetric: 
        kl_div_matrix = (kl_div_matrix + kl_div_matrix.t())/2.0
    norm = kl_div_matrix.norm(p = 'fro')
    return kl_div_matrix, norm

def selected_confidient_samples_ours(logits, top, reduction = 'batchhmean', no_filter=False):
    #logits are 32x1000, get kl_div_matrix from them. Get symmetric KL matrix
    kl_div_matrix, _ = kl_div_loss(logits, reduction= reduction, symmetric= True) 
    if no_filter:
        norm = kl_div_matrix.norm(p = 'fro')
        return logits, kl_div_matrix, None, norm
    selected_idx = torch.sum(kl_div_matrix, dim = -1).argsort()[:top]
    final_kl_matrix = kl_div_matrix[selected_idx][:, selected_idx]
    norm = final_kl_matrix.norm(p = 'fro')
    return logits[selected_idx], final_kl_matrix, selected_idx, norm

#TODO: BEST so far
def select_confident_samples_cosine(logits, selection_cosine=0.8, selection_selfentro=0.3):
    cosine_distan = [torch.nn.CosineSimilarity(dim=0)(logits[0], logits[i]) for i in range(1, logits.shape[0])] #len=63
    cosine_distan = torch.stack(cosine_distan)
    idx_cosine = torch.argsort(cosine_distan, descending=True)[:int(cosine_distan.size()[0] * selection_cosine)] #64*0.8=50 
    # idx
    for i in range(idx_cosine.shape[0]):
        # print(idx_cosine[i])
        idx_cosine[i] +=1
    logits_cos = logits[idx_cosine] #shape=(50,1000)
    logits = torch.cat((logits[0, :].unsqueeze(0), logits_cos), dim=0) #shape=(51,1000)
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1) #Rest same as select_confident_samples function
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * selection_selfentro)]
    return logits[idx], [idx_cosine, idx], cosine_distan

def select_confident_samples_cosine_on_inputs(image_list, top=6):
    # tensor_list = [F.to_tensor(image) for image in image_list]
    inputs = (image_list)
    cos_sim = torch.tensor([torch.nn.CosineSimilarity(dim=0)(inputs[0].flatten(), inputs[i].flatten()) for i in range(1, inputs.shape[0])])
    selected_indices = torch.argsort(cos_sim, descending=True)[:top]
    return selected_indices

def select_confident_samples_mahalanobis(logits, top=6, mean=None, cov=None):
    if mean is None or cov is None:
        mean, cov = logits.mean(dim=0), logits.cov()
    mahalanobis_distances = torch.norm(logits - mean, dim=1) / torch.sqrt(torch.diag(cov))
    selected_indices = torch.argsort(mahalanobis_distances, descending=False)[:top]
    return logits[selected_indices], selected_indices

from pytorch_msssim import ssim
def select_confident_samples_ssim(inputs, top=6):
    s_sim = torch.tensor([ssim(inputs[i].unsqueeze(0), inputs[0].unsqueeze(0)).item() for i in range(inputs.size(0))])
    selected_indices = torch.argsort(s_sim, descending=True)[:top]
    return selected_indices

def select_confident_samples_euclidean(logits, top=6):
    reference_image = logits[0].unsqueeze(0)
    euclidean_distance = torch.norm(logits - reference_image, dim=1)
    selected_indices = torch.argsort(euclidean_distance, descending=False)[:top]
    return logits[selected_indices], selected_indices

import torch
def select_confident_samples_pairwise_distance(logits, top=6):
    reference_logit = logits[0].unsqueeze(0)
    pairwise_distances = torch.nn.functional.pairwise_distance(reference_logit, logits)
    selected_indices = torch.argsort(pairwise_distances, descending=False)[:top]
    return logits[selected_indices], selected_indices

def select_confident_samples_entropy_var(outputs, top=6):
    metric = torch.tensor([entropy_var(outputs[i]) for i in range(outputs.size(0))])
    selected_indices = torch.argsort(metric, descending=True)[:top]
    return outputs[selected_indices], selected_indices
#Below combines the entropy and variance of the logits to identify samples that are both uncertain and diverse
def entropy_var(outputs):
    probs = torch.nn.functional.softmax(outputs, dim=0)
    entropy = -(probs * probs.log()).sum(dim=0)
    variance = torch.var(outputs, dim=0)
    return (0.5*entropy) + (0.5*variance)
    # if entropy > variance:
    #     return entropy
    # else:
    #     return variance
#Below combines the entropy, variance, and Jensen-Shannon divergence to identify samples that are both 
# uncertain, diverse, and different from the uniform distribution.
def entropy_var_JS(outputs):
    probs = torch.nn.functional.softmax(outputs, dim=0)
    entropy = -(probs * probs.log()).sum(dim=0)
    variance = torch.var(outputs, dim=0)
    metric = entropy + 0.5 * variance
    # Jensen-Shannon divergence between the logits and the uniform distribution.
    uniform_dist = torch.ones_like(probs) / probs.shape[0]
    jsd = torch.nn.functional.kl_div(torch.log(probs), uniform_dist, reduction='mean')
    # Add the Jensen-Shannon divergence to the metric.
    metric += jsd
    return metric

def select_confident_samples_cosine_var(logits, selection_cosine=0.8, selection_selfentro=0.3, selection_var=0.5):
    cosine_distan = [torch.nn.CosineSimilarity(dim=0)(logits[0], logits[i]) for i in range(1, logits.shape[0])]
    cosine_distan = torch.stack(cosine_distan)
    idx_cosine = torch.argsort(cosine_distan, descending=True)[:int(cosine_distan.size()[0] * selection_cosine)]
    # idx
    for i in range(idx_cosine.shape[0]):
        idx_cosine[i] +=1
    logits_cos = logits[idx_cosine]
    logits = torch.cat((logits[0, :].unsqueeze(0), logits_cos), dim=0)
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    variance = torch.var(logits, dim=-1)
    metric = (1.5 * batch_entropy) + (selection_var * variance)
    idx = torch.argsort(metric, descending=False)[:int(metric.size()[0] * selection_selfentro)]
    return logits[idx], [idx_cosine, idx]

def select_confident_samples_KL_div(output, reduction = 'sum', symmetric = True, top=6): 
    epsilon = 1e-10
    kl_div_matrix = torch.zeros(output.shape[0], output.shape[0]).to(output.device)
    output = torch.softmax(output, dim=-1)
    output  = torch.clamp(output, min= epsilon)
    for i in range(output.shape[0]):
        for j in range(output.shape[0]):
            kl_div = torch.nn.functional.kl_div(output[i].log(), output[j], reduction=reduction)
            kl_div_matrix[i, j] = kl_div
    if symmetric: 
        kl_div_matrix = (kl_div_matrix + kl_div_matrix.t())/2.0
        
    selected_idx = torch.sum(kl_div_matrix, dim = -1).argsort()[:top]
    final_kl_matrix = kl_div_matrix[selected_idx][:, selected_idx]
    norm = final_kl_matrix.norm(p = 'fro')
    return output[selected_idx], final_kl_matrix, selected_idx, norm

# logits = torch.randn((64, 1000))
# output, final_kl_matrix, selected_idx, norm = selected_confidient_samples_ours(logits, top=6, reduction='sum')
# print(output.shape)

def euclidean_dist(output):
    matrix = torch.cdist(output, output)
    norm = matrix.norm(p = 'fro')
    return matrix, norm

def get_copod(output, contamination = 0.5, range_loop = 1):
    for i in range(range_loop):
        copod_class = COPOD(contamination= contamination, n_jobs=1)
        copod_class.fit(np.float32(output.detach().cpu()))
        indices = torch.tensor(np.where(copod_class.labels_ == 0)[0])
        inliers_output = output[indices]
        if (range_loop > 1) and  (i < range_loop - 1):
            output = inliers_output
        if (range_loop > 3) and (i == (range_loop - 2)): 
            contamination = 0.2
    return inliers_output, indices

def print_messages(args, attack_step = None, level = None):
    
    if level == 'Method':
        print("========= Method =========")
        if args.frob_on_z_KL_TPT_losses: 
            print('Implementation with TPT, including extra loss (Frob on Z)')
        if args.frob_original_copod_1: 
            if args.kl:
                print('Implementation COPOD1 with KL loss only on z')
            elif args.ce_kl:
                print('Implementation COPOD1 with combination of losses KL and Avg CE only (TPT loss)')
            elif args.euclidean: 
                print('Implementation COPOD1 with Euclidean distance only on z')
            elif args.ce_euclidean:
                print('Implementation COPOD1 with combination of losses Euclidean and Avg CE only (TPT loss)')
            elif args.tpt_filtration and args.our_filtration:
                print("Implementation COPOD1 with combination of tpt filteration and our filtration")
            else:
                print('Implementation COPOD1 with Avg CE only (TPT loss)')
                
        if args.frob_original_copod_2: 
            print('Implementation with TPT, COPOD2 and KL-div loss')
        if args.frob_on_z and (not args.our_attack)  and (not args.evaluate_on_attack):
            print('Implementation with TPT and KL-div matrix on Z not CLS')
        if args.frob_on_original and (not args.our_attack)  and (not args.evaluate_on_attack):
            print('Implementation with TPT and KL-div matrix on all images not z')
        if args.self_augmentation and (not args.our_attack)  and (not args.evaluate_on_attack): 
            print(f'Self augmentation only with clean samples: \tBlocks: {args.initial_block}-{args.final_block} \tNumber of masked patches: {args.num_maksed_patches}')

        if args.our_attack and (not args.self_augmentation)  and (not args.evaluate_on_attack): 
            print(f'Implementation of our attack on TPT using: \tAttack: {args.attack_name} \tTotal steps{args.steps_our} (1 normal and rest is generation of attacks)') 

        if args.evaluate_on_attack and (not args.self_augmentation)  and (not args.our_attack): 
            print(f'Evaluate TPT on attacks \tFolder name: {args.attack_folder_name}') 

        if args.self_augmentation and args.evaluate_on_attack and (not args.our_attack):
            print('Evaluate our slef-augmentation method with adversarial attacks.')
            print(f'Evaluate TPT on attacks \tFolder name: {args.attack_folder_name}') 
            print(f'\tBlocks: {args.initial_block}-{args.final_block} \tNumber of masked patches: {args.num_maksed_patches}')

        if (not args.self_augmentation) and (not args.evaluate_on_attack) and (not args.our_attack):
            print('Run TPT only with clean samples')
            
        if args.similarity != None:
            print(f"Using {args.similarity} similarity instead of self-entropy")
    
    if level == "Dataset":
        if args.our_attack and (not args.self_augmentation)  and (not args.evaluate_on_attack) and (attack_step!=0):
            print(f' ====== Apply TPT with attacks from previous step: Round {attack_step+1} of attack generation ===== ')
            print('Load images from attacks generated in previous step.')

        if args.self_augmentation and args.evaluate_on_attack and (not args.our_attack):
            print(f'====== Evaluate TPT with self-augmentation method on attack samples =======')
            print('Load attack samples')

        if (not args.self_augmentation) and (not args.our_attack)  and (not args.evaluate_on_attack):
            print('====== Load data for original TPT -- Original clean dataset ======')
            
            
            
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def ternary_plot(outputs, save_path='ternary_plot.png'):
    outputs = outputs.float().to('cpu')
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    probs = torch.exp(logits) # 64, 1000
    # print(probs.shape)
    
    prob_distribution, prob_indices = torch.topk(probs, k=3, dim=-1) # 64, 3
    # prob_distribution /= torch.sum(prob_distribution, dim=-1, keepdim=True)
    # print(prob_distribution[0].sum())
    # print(prob_distribution.shape)
    
    # print(prob_distribution[0][0:3])
    # print(prob_distribution[-1][0:3])
        
    vertices = torch.tensor([[0, 0], [1, 0], [0.5, torch.sqrt(torch.tensor(3.0)) / 2]], dtype=torch.float32)

    plt.figure(figsize=(2, 2))

    # Draw the triangle with color and thicker line
    plt.fill(vertices[:, 0], vertices[:, 1], color='lightblue', alpha=0.6)

    # Plot the probability points for each distribution
    for i in range(prob_distribution.shape[0]):
        point = torch.matmul(prob_distribution[i], vertices)
        
        transparency = 1.0 - i / (prob_distribution.shape[0] - 1) * 0.9
        
        plt.scatter(point[0].item(), point[1].item(), c='red', alpha=transparency, edgecolors='black', linewidth=0.5)

        # # Annotate the point with class probabilities
        # for j, txt in enumerate(prob_distribution[i]):
        #     plt.annotate(f'{txt.item():.2f}', (vertices[j, 0].item(), vertices[j, 1].item()), textcoords="offset points", xytext=(0, 10), ha='center', color='black', fontsize=8)

    plt.title((save_path.rsplit(".", 1)[0]).rsplit("/", 1)[1], fontsize=10)
    # plt.xlabel('Class 1', fontsize=12)
    # plt.ylabel('Class 2', fontsize=12)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()
        
import matplotlib.pyplot as plt
def plot_img(image, save_path='saved_plot.png', target=None, predicted=None):
    image_array = image.to('cpu').squeeze().permute(1, 2, 0).detach().numpy()
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    plt.figure(figsize=(3, 3))
    plt.imshow(image_array)
    title = f'Target: {target}, Pred: {predicted}'
    plt.axis('off')
    plt.title(title, fontsize=10)
    plt.savefig(save_path)
    plt.close()
    
from PIL import Image
def tensor_to_pil_image(tensor):
    """
    Convert a PyTorch tensor to a PIL Image.
    """
    # tensor = tensor.clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    return image

def save_pil_plot(image, file_path='saved_pil.png'):
    # Save a PIL Image object to a file.
    try:
        image.save(file_path)
        print(f"Image saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
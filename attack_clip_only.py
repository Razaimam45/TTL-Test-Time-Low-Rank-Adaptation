import torch
import clip
from PIL import Image
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from attack import Attack
from PIL import Image
from clip.model import CLIP
import torchvision.transforms as transforms
import foolbox
import clip_attack
import torchvision.datasets as datasets
from tqdm import tqdm
from functions import zeroshot_classifier, accuracy, get_actual_label_batch, get_actual_label_one_sample
from imagenetv2_pytorch import ImageNetV2Dataset
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
from clip.custom_clip import MyClip

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.imagnet_prompts import imagenet_templates, imagenet_classes

def main(
        attack_name = 'PGD', steps = 100, gpu = 3, batch_size = 128, num_workers = 12, 
        image_size = 224,epsilon = 0.03, ensemble = False, generate_attacks = True, 
        evaluate_on_attacks= True, evaluate_on_clean = True, attack_images_dir_path = None):
    

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"

    model_clip, _, preprocess_clip = clip.load(name = "ViT-B/16", device='cpu')

    for param in model_clip.parameters():
        param.requires_grad = False

    model_clip = model_clip.to(device=device)
    model = MyClip(clip_model= model_clip, class_names= imagenet_classes, templetaes= imagenet_templates, ensemble= ensemble, device= device)
    model.eval() 

    for param in model.parameters():
        param.requires_grad = False
    
    mean= torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
    std= torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
    preprocess_attack={'mean':mean, 'std': std, 'axis': -3}


    attack_ = Attack(epsilon=epsilon, attack_type= attack_name, model= model, bounds = (0,1), 
                     device= device, preprocess=preprocess_attack, clip_flag= False, steps= steps)
    
    def _convert_image_to_rgb(image):
        return image.convert("RGB")
    
    if evaluate_on_clean:
        print("===== Get clean test data ===== ")
        dataset_dir = '/home/faris_almalik/Desktop/TPT/TPT_attack/data/imagenetv2-matched-frequency-format-val'
        data_transform_clean_samples = transforms.Compose([
                        transforms.Resize(image_size, interpolation=BICUBIC),
                        transforms.CenterCrop(image_size),
                        _convert_image_to_rgb,
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                        ])

        testset = datasets.ImageFolder(dataset_dir, transform=data_transform_clean_samples)
        print(f"Number of samples = {len(testset)}")

        testloader = torch.utils.data.DataLoader(
                                testset,
                                batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

        counter = 1
        counter_attack = 0.

        print('===== Evaluate the clean performance ===== ')
        with torch.no_grad():
            top1, top5, n = 0., 0., 0.
            for images, label in tqdm(testloader):
                images = images.to(device)
                labels = get_actual_label_batch(testset.class_to_idx, label).to(device= device)
                logits = model.inference(images)
                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
                top1 += acc1[0]
                top5 += acc5[0]
                n += images.size(0)
        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100 
        print(f"Top1 = {top1:.3f} \t Top5 = {top5:.3f}")

        del images, labels, logits

    if generate_attacks: ###### This  Flag to generate attacks.
        print('===== Generating Attacks ===== ')
        dataset_dir = '/home/faris_almalik/Desktop/TPT/TPT_attack/data/imagenetv2-matched-frequency-format-val'

        if attack_name == 'PGD':
            base_dir = f'/home/faris_almalik/Desktop/TPT/TPT_attack/data/imagenetv2-matched-frequency-format-val-attack-{attack_name}-{steps}steps-CLIP-only'
        else:
            base_dir = f'/home/faris_almalik/Desktop/TPT/TPT_attack/data/imagenetv2-matched-frequency-format-val-attack-{attack_name}-CLIP-only'

        for i in range(len(imagenet_classes)):
            folder_name = str(i)
            folder_path = os.path.join(base_dir, folder_name)
            
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        print("Attack folders are created successfully")
        print(f'Get images without normalization')
        data_transform_for_attack_gen = transforms.Compose([
                        transforms.Resize(image_size, interpolation=BICUBIC),
                        transforms.CenterCrop(image_size),
                        _convert_image_to_rgb,
                        transforms.ToTensor(),
                        ])
        
        testset = datasets.ImageFolder(dataset_dir, transform=data_transform_for_attack_gen)

        testloader = torch.utils.data.DataLoader(
                                testset,
                                batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        
        print(f"Number of samples = {len(testset)}")
        counter = 0
        counter_attack = 0
        for images, label in tqdm(testloader):
            images = images.to(device)
            labels = get_actual_label_batch(testset.class_to_idx, label).to(device)
            adv_image, success = attack_.generate_attack(images, labels)
            
            # if adv_image[success].shape[0] > 0:
                # for image, lab in zip(adv_image[success], labels[success]):
            for image, lab in zip(adv_image, labels):
                image_name = f"{counter}.jpeg"
                counter_attack += success.sum().item()
                save_image(image, os.path.join(base_dir, str(lab.item()), image_name))
                counter+=1
    
    if evaluate_on_attacks: #this flag to eval the model on adversarial samples
        print('===== Evaluate CLIP model on Generated Attack ===== ')
        if (attack_name == 'FGSM') and (attack_images_dir_path is None):
            print(f"attack_name: {attack_name}, Epsilon: {epsilon}")
            dataset_dir = base_dir
        elif (attack_name == 'PGD') and (attack_images_dir_path is None):
            print(f"attack_name: {attack_name}, Epsilon: {epsilon}, Steps: {steps}")
            dataset_dir = base_dir
        else:
            print(f' Attack images folder : {attack_images_dir_path}')
            dataset_dir = attack_images_dir_path

        print('Get attack images')
        data_transform = transforms.Compose([
                        transforms.Resize(image_size, interpolation=BICUBIC),
                        transforms.CenterCrop(image_size),
                        _convert_image_to_rgb,
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                        ])

        testset = datasets.ImageFolder(dataset_dir, transform=data_transform)

        testloader = torch.utils.data.DataLoader(
                                testset,
                                batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        print(f"Number of samples = {len(testset)}")
        
        with torch.no_grad():
            top1, top5, n = 0., 0., 0.
            for i, (images, target) in enumerate(tqdm(testloader)):
                images = images.to(device)
                labels = get_actual_label_batch(testset.class_to_idx, target).to(device)
                # predict
                logits = model.inference(images)
                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
                top1 += acc1[0]
                top5 += acc5[0]
                n += images.size(0)
        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100 
        print(f"Top1 = {top1:.3f} \t Top5 = {top5:.3f}")


if __name__ == "__main__":

    main(
        attack_name = 'FGSM',
        steps = 100,
        gpu = 3,
        batch_size = 128,
        num_workers = 12,
        image_size = 224,
        epsilon = 0.03,
        ensemble = True,
        generate_attacks = False, 
        evaluate_on_attacks = True,
        evaluate_on_clean= True,
        # attack_images_dir_path= '/home/faris_almalik/Desktop/TPT/TPT_attack/data/imagenetv2-matched-frequency-format-val-attack-2steps-PGD'
        attack_images_dir_path= '/home/faris_almalik/Desktop/TPT/TPT_attack/data/imagenetv2-matched-frequency-format-val-attack-PGD-100steps-CLIP-only'
        # attack_images_dir_path= '/home/faris_almalik/Desktop/TPT/TPT_attack/data/imagenetv2-matched-frequency-format-val-attack-FGSM-CLIP-only'
    )
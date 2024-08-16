import cv2
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
        
import matplotlib.pyplot as plt
def plot_img(image, save_path='saved_plot.png', target=None, predicted=None):
    if type(image) == torch.Tensor:
        image_array = image.to('cpu').squeeze().permute(1, 2, 0).detach().numpy()
    else:
        image_array = image
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    plt.figure(figsize=(3, 3), tight_layout=True)
    plt.imshow(image_array)
    # title = f'Target: {target}, Pred: {predicted}'
    plt.axis('off')
    # plt.title(title, fontsize=10)
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
        
        
# Attention code for HuggingFace CLIP model
def attention_rollout(model, input_):
    model.config.output_attentions = True
    with torch.no_grad():
        output = model(input_)
        attention_maps = torch.concat(output.attentions, dim=0)
        num_attentions, num_heads, num_tokens, num_tokens = attention_maps.size()
        attention_maps_fused = torch.mean(attention_maps, dim=1)
        attention_maps_fused_aug = attention_maps_fused + torch.eye(num_tokens).to(attention_maps_fused.device)
        attention_maps_fused_aug_normalized = attention_maps_fused_aug / attention_maps_fused_aug.sum(dim=-1, keepdim=True)

        rollout, *ms = attention_maps_fused_aug_normalized
        for m in ms:
            rollout = torch.matmul(m, rollout)
        patch_size = int((num_tokens - 1)**0.5)
        mask = rollout[0, 1:].reshape(patch_size, patch_size)
        mask_normalized = (mask - mask.min()) / (mask.max() - mask.min())
        return mask_normalized

# Put the mask on the image
def apply_mask(mask, img):
    h_img, w_img, c = img.shape
    mask = cv2.resize(mask, (w_img, h_img))
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255.0 * mask), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) / 255.0
    cam = heatmap + np.float32(img)
    # cam = cam / np.max(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    return np.uint8(255.0 * cam)

def post_process(img): # De-normalize the image tensor and swap the axes to fit opencv
    return img.squeeze(0).permute(1, 2, 0) * 0.5 + 0.5

def save_attn(image_encoder, image, out_path, save_img_path=None):
    mask_ref = attention_rollout(image_encoder.vision_model, image)
    cam_ref = apply_mask(mask_ref.to('cpu').numpy(), post_process(image).to('cpu').numpy())
    # viz_image = Image.fromarray(np.uint8(cam_ref))
    # viz_image.save(out_path)
    plot_img(np.uint8(cam_ref), out_path)
    if save_img_path is not None:
        plot_img(image, save_img_path)
    return cam_ref

# T_SNE plot
from sklearn.manifold import TSNE
def plot_features(features, labels, num_classes, targeted_class_dict, save_path="plots/t_sne"):
    
    tsne = TSNE(n_components=2, random_state=0, perplexity=5.0)
    features = tsne.fit_transform(features)
    
    colors = ['C{}'.format(i) for i in range(num_classes)]
    plt.figure(figsize=(3, 3), dpi=350)
    for idx, label_idx in enumerate(list(targeted_class_dict.keys())):
        plt.scatter(
            features[labels.flatten()==label_idx, 0],
            features[labels.flatten()==label_idx, 1],
            c=colors[idx],
            s=15,
        )

    plt.grid(True, which='major', color='gray', linestyle='-', alpha=0.4)
    plt.grid(True, which='minor', color='gray', linestyle='--', alpha=0.1)
    plt.minorticks_on()
    plt.xticks([])
    plt.yticks([])
    class_names = list(targeted_class_dict.values())
    # plt.legend(ncol=2, labelspacing=0.1, prop={'size': 10, 'weight': 'bold'}, bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.title(f"CLIP Vision Feature Space", weight='bold', size=16)
    plt.tight_layout()
    plt.show()
    # dirname = os.path.join(save_dir)
    # if not os.path.exists(dirname):
    #     os.makedirs(dirname)
    plt.savefig(f'{save_path}.pdf', dpi=350, facecolor='auto', edgecolor='auto',
            orientation='portrait', format="pdf",
            transparent=False, bbox_inches='tight', pad_inches=0.05)
    plt.close()
#blip model 
from PIL import Image
import requests
from transformers import AutoProcessor, BlipModel

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


clean_folder_dir = './data/imagenetv2-matched-frequency-format-val'
attack_folder_dir = './data/imagenetv2-matched-frequency-format-val-attack-2steps-FGSM'

#get the model 
model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

#freeze all params. 
for params in model.parameters():
    params.requires_grad = False

#put in eval mode 
model.eval()

#get the dataset 
clean_transform = processor.image
clean_dataset = datasets.ImageFolder(clean_folder_dir, transform=processor)

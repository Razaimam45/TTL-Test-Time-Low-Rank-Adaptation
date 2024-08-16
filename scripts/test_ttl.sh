#!/bin/bash

# Default parameters
DATA_ROOT='/home/raza.imam/Documents/TPT/datasets' # '/home/raza.imam/Documents/TPT/datasets'
TEST_SETS='A' # Options: A/V/R/K
MODE='test'
ARCH='ViT-B/16' # Options: ViT-B/16, ViT-B/32, RN50
BS=64
CTX_INIT='a_photo_of_a' #prompt initialization
LR=5e-3
TTA_STEPS=1
PRINT_FRQ=10
GPU=1
SELECTION_P=0.1 # Options: 0.1=6, 1.0=64
LAYER_RANGE=9,11 # Apply LORA on layers 9-11
INIT_METHOD='xavier'
LORA_ENCODER='image' # Options: image, text, prompt (for TPT)
DEYO_SELECTION=True # Options: True, False (True for weighted entropy)

# Command to run ttl.py with the above parameters
python3 ttl.py --data $DATA_ROOT \
               --test_sets $TEST_SETS \
               --dataset_mode $MODE \
               --arch $ARCH \
               --b $BS \
               --ctx_init $CTX_INIT \
               --lr $LR \
               --tta_steps $TTA_STEPS \
               --print_freq $PRINT_FRQ \
               --gpu $GPU \
               --selection_p $SELECTION_P \
               --layer_range $LAYER_RANGE \
               --init_method $INIT_METHOD \
               --lora_encoder $LORA_ENCODER \
               --deyo_selection $DEYO_SELECTION

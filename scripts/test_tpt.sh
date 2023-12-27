#!/bin/bash
# data_root='../data'
# data_root='/l/users/faris.almalik/data'
data_root='../data'
testsets=V
arch=ViT-B/16
ctx_init=a_photo_of_a
bs=64
epsilon=0.03
attack_name=PGD
steps_our=2
initial_block=0
final_block=11
num_maksed_patches=20
attack_folder_name='attack-2steps-FGSM'
gpu=1
# arch=RN50

######epsilon, steps_our_attack
# --epsilon ${epsilon}\
# --attack_name ${attack_name} \
# --our_attack \
# --steps_our ${steps_our}\
##########for self-augmentation
# --self_augmentation \
# --num_maksed_patches ${num_maksed_patches} \
# --initial_block ${initial_block} \
# --final_block ${final_block} \
###### Evaluate on Attacks
# --evaluate_on_attack \
# --attack_folder_name ${attack_folder_name}\


# python ../tpt_classification_my_edit.py \
#         ${data_root} \
#         --test_sets ${testsets} \
#         -a ${arch} \
#         -b ${bs} \
#         --gpu ${gpu} \
#         --tpt \
#         -j 8 \
#         --seed 125 \
#         --ctx_init ${ctx_init} \
#         --frob_original_copod_1 \
#         --number_of_inlaiers 32 \
#         --our_filtration 

# default_all_data = False
images_per_class=2
data_root='/home/raza.imam/Documents/TPT/datasets/'
testsets='V/A'
arch='RN50'
workers=2
bs=64
ctx_init='a_photo_of_a'
print_freq=10

python tpt_classification_my_edit.py \
    ${data_root} \
    --images_per_class ${images_per_class}\
    --test_sets ${testsets} \
    --dataset_mode test \
    -a ${arch} \
    -j ${workers} \
    -b ${bs} \
    -p ${print_freq} \
    --gpu ${gpu} \
    --tpt \
    --ctx_init ${ctx_init} \
    --seed 0 \
#     --frob_original_copod_1 \

#     --our_attack \
#     --steps_our 2 \
#     --self_augmentation \
#     --num_maksed_patches 10 \
#     --initial_block 0 \
#     --final_block 12 \
#     --evaluate_on_attack \
#     --attack_folder_name ${attack_folder_name} \
#     --cluster \
#     --frob_on_z \
#     --frob_on_original \
#     --frob_original_copod_2 \
#     --number_of_inlaiers 32 \
#     --kl \
#     --no_filtration \
#     --our_filtration \
#     --tpt_filtration \
#     --ce_kl \
#     --euclidean \
#     --ce_euclidean \
#     --frob_on_z_KL_TPT_losses

        
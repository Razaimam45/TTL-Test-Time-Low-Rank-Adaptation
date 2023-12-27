import argparse
from copy import deepcopy
from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from functions import test_time_adapt_eval, get_class_names, define_optimizer
from functions import prepare_test_sets, create_attack_folders, print_messages
import json    

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop
from clip.cocoop import get_cocoop
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import load_model_weight, set_random_seed
from data.cls_to_names import *

#for model names only
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def main(args):

    if True:    
        device_test = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device Available: {device_test}")

        # This codebase has only been tested under the single GPU setting
        if not args.cluster:
            assert args.gpu is not None

        #gpu and seed
        if not args.cluster:
            print("========= Use GPU: {} for training =========".format(args.gpu))
        else: 
            print("========= Use GPU on Cluster =========")
        set_random_seed(args.seed)

        # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
        #get classnames
        classnames = get_class_names(args.test_sets) #normal classes not fixed ones

        # This is False by default.
        if args.cocoop:
            model = get_cocoop(args.arch, args.test_sets, 'cpu', args.n_ctx)
            assert args.load is not None
            load_model_weight(args.load, model, 'cpu', args) # to load to cuda: device="cuda:{}".format(args.gpu)
            model_state = deepcopy(model.state_dict())
        #we will enter here
        else:
            #get class names, model (ClipTestTimeTuning model)
            if not args.cluster:
                model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
            else: 
                model = get_coop(args.arch, args.test_sets, "cuda" if torch.cuda.is_available() else "cpu", args.n_ctx, args.ctx_init)
            #default is none, if not none get the pretrained cocoop
            if args.load is not None:
                print("========= Use pre-trained soft prompt (CoOp) as initialization =========")
                pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
                assert pretrained_ctx.size()[0] == args.n_ctx
                with torch.no_grad():
                    model.prompt_learner[0].ctx.copy_(pretrained_ctx)
                    model.prompt_learner[0].ctx_init_state = pretrained_ctx
            model_state = None

        #fix all params except prompt learner. Meaning, we are not fixing ctx_vector (embedding of ctx)
        for name, param in model.named_parameters():
            #will enter here
            if not args.cocoop:
                #fix all params except prompt learner. Meaning, we are not fixing ctx_vector (embedding of ctx)
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
            #we dont enter here
            else:
                if "text_encoder" not in name:
                    param.requires_grad_(False)
        
        print("========= Model created: visual backbone {} =========".format(args.arch))
        
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
        else:
            if not args.cluster:
                assert args.gpu is not None
                torch.cuda.set_device(args.gpu)
                model = model.cuda(args.gpu)
            else: 
                model = model.cuda()

        # define optimizer, anf set the params to the ctx only.
        optimizer, optim_state = define_optimizer(args.cocoop, model, args.lr)
    
        # setup automatic mixed-precision (Amp) loss scaling
        scaler = torch.cuda.amp.GradScaler(init_scale=1000)
        print('========= Using native Torch AMP. Training in mixed precision. =========')
        cudnn.benchmark = True

        # iterating through eval datasets
        datasets = args.test_sets.split("/") #can provide multiple datasets
        results = {}
    
    for set_id in datasets: 
        our_attack = args.our_attack #param to chaange
        steps_our = args.steps_our #param to change (the number of rounds of attacks generation + 1 (e.g. if we need 4 rounds of generation, put it 2 --- 1 round clean and one generation.))
        number_steps_our_attack = 1 if not our_attack else steps_our 
        self_augmentation = args.self_augmentation #used to perform the intermediate patches

        #print message for specific method used
        print_messages(args= args, attack_step= None, level= 'Method')

        for attack_step in range(number_steps_our_attack):
            #prepare test_set transformation, views, and classnames
            #here the classes got flipped to be the ones that can be used. (barmbling has index of 2 not 100)
            data_transform, batchsize, classnames = prepare_test_sets(args.tpt, args.resolution, set_id, args.batch_size, self_augmentation = args.self_augmentation)
        
            print("========= evaluating: {} ========= ".format(set_id))
            #dont enter
            if args.cocoop:
                model.prompt_generator.reset_classnames(classnames, args.arch)
                model = model.cpu()
                model_state = model.state_dict()
                model = model.cuda(args.gpu)
            else:
                #get the class_names, num_classes, tokenized prompts, prefix/suffix embeddings.
                model.reset_classnames(classnames, args.arch)

            print_messages(args= args, attack_step= attack_step, level= 'Dataset') #printing
            val_dataset = build_dataset(set_id= set_id, transform= data_transform, args= args, attack_step= attack_step)
            
            print(" ========= number of test samples: {} ========= ".format(len(val_dataset)))
            val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=batchsize, shuffle=True,
                        num_workers=args.workers, pin_memory=True)
            
            if (our_attack):
                print(' ========= Create Attack Images Folder =========')
                base_dir = create_attack_folders(base_directory='../data', dataset_id= set_id, n_classes=len(classnames), steps_our = steps_our, creat_fold = True, attack_name = args.attack_name)
            else: 
                base_dir = None
            # we need base_di to save attack images.
            
            generate_attack_images = False if (attack_step+1) == number_steps_our_attack else True #if this is last round, no need to generate attack samples

            results[set_id] = test_time_adapt_eval(val_loader, model, model_state, optimizer, 
                                                optim_state, scaler, args, set_id= set_id, generate_attack_images= generate_attack_images,
                                                val_dataset= val_dataset, our_attack =  our_attack, base_dir= base_dir, 
                                                self_augmentation= self_augmentation)
            
            del val_dataset, val_loader
            
            try:
                print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
            except:
                print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

            if (our_attack) and (attack_step != 0):   
                #get dataset from the folder, and do transform.
                print(f'========= Rround = {attack_step}/{number_steps_our_attack - 1} of attack generation will start now =============')

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
    if our_attack:
        print(f'Our attack with: attack name --> {args.attack_name}, normal steps --> 1, generattion round --> {number_steps_our_attack -1}, Tuning with attacks steps --> {number_steps_our_attack}')

if __name__ == '__main__':
    # Defining default values for debugging
    default_all_data = False
    default_images_per_class = None # None or int
    
    default_data_root = '/home/raza.imam/Documents/TPT/datasets/'
    default_test_sets = 'V/A'
    default_arch = 'ViT-B/16' #ViT-B/16 or RN50
    default_workers = 2
    default_bs = 32
    default_lr = 5e-3
    default_tta_steps = 1
    default_ctx_init = 'a_photo_of_a'
    default_print_freq = 10
    
    default_our_attack = False #No need
    default_self_aug = False #No need
    default_num_maksed_patches = 10 #No need
    default_initial_block = 0 #No need
    default_final_block = 12 #No need
    default_evaluate_on_attack = False #No need
    default_attack_folder_name = None #No need
    default_cluster = False #No need
    default_frob_on_z = False
    default_frob_on_original = False
    default_frob_original_copod_2 = False #No need
    default_frob_original_copod_1 = False
    default_number_of_inlaiers = 32
    default_kl = False
    default_no_filtration = False
    default_our_filtration = False
    default_tpt_filtration = False    
    default_ce_kl = False
    default_euclidean = False
    default_ce_euclidean = False
    default_frob_on_z_KL_TPT_losses = False
    
    default_temperature = 1.0
    default_similarity = 'cosine' #None(i.e. CE)/cosine/mahalanobis/ssim/euclidean/pairwise/entropy_var/cos_var /kl_div
    default_loss_type = "entropy" #norm/entropy
    default_gpu = 1

    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    
    parser.add_argument('--all_data', action='store_true', default=default_all_data, help='Test on all 10k images')
    parser.add_argument('--images_per_class', default=default_images_per_class, type=int, help='Number fo images per class to load (should be <=10)')

    
    parser.add_argument('data', metavar='DIR', nargs="?", default=default_data_root, help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default=default_test_sets, help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default=default_arch)
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=default_workers, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=default_bs, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=default_lr, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=default_print_freq, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=default_gpu, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=True, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile') #FIXME: 10% confident views of the total augmented views to be selected
    parser.add_argument('--tta_steps', default=default_tta_steps, type=int, help='test-time-adapt steps') #No modify need
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens') 
    parser.add_argument('--ctx_init', default=default_ctx_init, type=str, help='init tunable prompts') #TODO: Initial input prompt to tune, if not give, then a random initialization of (X * num_tokens) = X X X X
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    #--- Above args same of tpt_classification.py
    
    parser.add_argument('--epsilon', default=0.1, type=float,  help="eps for attack")
    parser.add_argument('--attack_name', default='PGD', type=str, help='attack name')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--our_attack', action='store_true', default=default_our_attack, help='Our attack Flag')
    parser.add_argument('--steps_our', default=2, type=int, help='number of rounds to train on adv attacks. default = 2, meaning one round of training on attacks.')
    parser.add_argument('--self_augmentation', action='store_true', default=default_self_aug, help='Self-augmentation Flag')
    parser.add_argument('--num_maksed_patches', type=int, default=default_num_maksed_patches,  help='Number of masked patches in self-augmentation')
    parser.add_argument('--initial_block', type=int, default=default_initial_block,  help='Initial block to choose for self-augmentation')
    parser.add_argument('--final_block', type=int, default=default_final_block, help='Final block to choose for self-augmentation')
    parser.add_argument('--evaluate_on_attack', action='store_true', default=default_evaluate_on_attack, help='Flag for loading adv dataset')
    parser.add_argument('--attack_folder_name', default=default_attack_folder_name, type=str, help='Name of attack Folder')
    parser.add_argument('--cluster', action='store_true', default=default_cluster, help='Cluster Flag')
    parser.add_argument('--frob_on_z', action='store_true', default=default_frob_on_z, help='Flag for Frob on z')
    parser.add_argument('--frob_on_original', action='store_true', default=default_frob_on_original, help='Flag for Frob on original (all images without filtration)')
    parser.add_argument('--frob_original_copod_2', action='store_true', default=default_frob_original_copod_2, help='Flag for Frob on original with COPOD2 Filtration')
    parser.add_argument('--frob_original_copod_1', action='store_true', default=default_frob_original_copod_1, help='Flag for Frob on original with COPOD1 Filtration')
    parser.add_argument('--number_of_inlaiers', default= default_number_of_inlaiers, type=int, help='confidence selection percentile')
    parser.add_argument('--kl', action='store_true', default=default_kl, help='Flag for kl only for COPOD1')
    parser.add_argument('--no_filtration', action='store_true', default=default_no_filtration, help='No Filtration and get 32*1000 for COPOD1')
    parser.add_argument('--our_filtration', action='store_true', default=default_our_filtration, help='KL filtration based on least sum of KL')
    parser.add_argument('--tpt_filtration', action='store_true', default=default_tpt_filtration, help='TPT Filtration and get 32*1000 for COPOD1')
    parser.add_argument('--ce_kl', action='store_true', default=default_ce_kl, help='Flag for kl and avg CE for COPOD1')
    parser.add_argument('--euclidean', action='store_true', default=default_euclidean, help='Flag for euclidean rather than kl')
    parser.add_argument('--ce_euclidean', action='store_true', default=default_ce_euclidean, help='Flag for euclidean and CE rather')
    parser.add_argument('--frob_on_z_KL_TPT_losses', action='store_true', default=default_frob_on_z_KL_TPT_losses, help='Flag for Frob on z, including KL and TPT avg CE')
    
    parser.add_argument('--temperature', default=default_temperature, type=float,  help="temperature for logits") 
    parser.add_argument('--similarity', default=default_similarity, type=str, help='Type of similarity instead of self-entropy')
    parser.add_argument('--loss_type', default=default_loss_type, type=str, help='Type of loss instead of self-entropy')
    
    
    args = parser.parse_args()
    
    print(args)
    
    main(args)
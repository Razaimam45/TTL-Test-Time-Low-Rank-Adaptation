
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *
from clip.model import CLIP
from pkg_resources import packaging
from tqdm import tqdm

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='~/.cache/clip'

class ClipImageEncoder(nn.Module):
    """
    Takes the original Clip model, Load it and: 
        1- get the embedding dimension (512), and origianl clip model
        2- store image encoder in self.encoder (take image and give its embeddings) --> (batch_size,512)
        3- Delete the text encoder. 
        4- Create linear linear to project from emb_dim to n_classes 
    Forward pass: 
        1- takes an image, pass it through self.encoder. (batch_size,512)
        2- pass the output through a linear layer to get the logits over n_classes (batch_size, n_classes)
    
    Return: 
        Logits over n_classes (batch_size, n_classes)
    """
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        #load original clip model, return clip model, and embed_dim, image proccessor.
        #neglect image processor which transform PIL image and do resize, normalize and make it tensor so that image encoder can take 
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        #save the Image encoder from the original clip model
        self.encoder = clip.visual
        # Remove the text encoder implemented in the original clip
        del clip.transformer
        torch.cuda.empty_cache()
        #Linear layer projects from 512 (emb) to number of classes.
        self.cls_head = nn.Linear(embed_dim, n_class)
    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        #forward pass for image encoder. Takes an image and pass it to ViT (encoder)
        x = self.encoder(image.type(self.dtype))
        #pass te result to class head (project from 512cto n_classes). output logits.
        output = self.cls_head(x)
        #return Logits
        return output

# Raza
class VisionEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.get_image_features = clip_model.get_image_features
        self.vision_model = clip_model.vision_model
        self.dtype = clip_model.dtype
        
    def forward(self, image):
        x = self.get_image_features(image.type(self.dtype))
        return x

class PromptEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.get_text_features = clip_model.get_text_features
        self.text_model = clip_model.text_model
        self.dtype = clip_model.dtype
        
    def forward(self, prompts):
        x = self.get_text_features(prompts)
        return x
    
    
class TextEncoder(nn.Module):
    """
    
    
    """
    def __init__(self, clip_model):
        super().__init__()
        #get the transformer (text encoder) from original clip)
        self.transformer = clip_model.transformer
        #Get the text positional embedding, default the model takes max seq of 77 (77x512)
        self.positional_embedding = clip_model.positional_embedding
        # Final Layer norm from clip original model 512
        self.ln_final = clip_model.ln_final
        # Project it via a linear layer (512 -> 512)
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        #add pos embedding to prompts
        x = prompts + self.positional_embedding.type(self.dtype)
        # change order from (batch, num_tokens, embed) to (num_token, batch_size, embed)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass to text encoder (transformer in original clip)
        x = self.transformer(x)
        # change order from (num_token, batch_size, embed) to (batch, num_tokens, embed)  
        x = x.permute(1, 0, 2)  # LND -> NLD
        #get the final output (b_size, num_tokens, 512)
        x = self.ln_final(x).type(self.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        #tokenized prompts are the tokenized text we alreadey enter --- a photo of a dog (eg)
        #x has a shape of (b_size, num_tokens, 512)
        #x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] --- (batch_size, embed_dim)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        #x now has a shape of (batch_size, 1, 512)
        return x

# def __init__(self, model, layer_range):
#         # model is the text encoder (in future can be vision encoder)
#         # super().__init__()
#         self.model = model
#         self.layer_range = layer_range
#         # saving the initial weights during model initialization

#         self.init_weights = []
#         for layer in self.model.text_model.encoder.layers:
#             init_weight_q_A = layer.self_attn.q_proj.lora_A.default.weight.detach().clone()
#             init_weight_q_B = layer.self_attn.q_proj.lora_B.default.weight.detach().clone()
#             init_weight_v_A = layer.self_attn.v_proj.lora_A.default.weight.detach().clone()
#             init_weight_v_B = layer.self_attn.v_proj.lora_B.default.weight.detach().clone()
#             self.init_weights.append((init_weight_q_A, init_weight_q_B, init_weight_v_A, init_weight_v_B))


import torch.nn.init as init
# Raza
class LoRA_AB:
    """
    LoRA module to reset the weights of LoRA's A and B wts after every inference. 
    """
    def __init__(self, model, layer_range, init_method='xavier', lora_encoder='text'):
        self.model = model # model can either be text or image encoder
        self.layer_range = layer_range
        self.init_method = init_method
        self.lora_encoder = lora_encoder
        self.init_weights = []

        self.initialize_weights()

    def initialize_weights(self):
        if (self.init_method == 'xavier') or (self.init_method == None):
            init_function = init.xavier_normal_
            print("Initializing LoRA A and B weights with Xavier")
        elif self.init_method == 'gaussian':
            init_function = init.normal_
            print("Initializing LoRA A and B weights with Gaussian")
            # init_function = init.trunc_normal_
            # print("Initializing LoRA A and B weights with T. Gaussian")
        elif self.init_method == 'kaiming':
            init_function = init.kaiming_normal_
            print("Initializing LoRA A and B weights with Kaiming")
        elif self.init_method == 'pretrained':
            init_function = None
        else:
            raise ValueError(f"Unsupported init_method: {self.init_method}")

        if self.lora_encoder == 'image':
            for layer in self.model.vision_model.encoder.layers:
                self.initialize_layer_weights(layer, init_function)
        elif self.lora_encoder == 'text':
            for layer in self.model.text_model.encoder.layers:
                self.initialize_layer_weights(layer, init_function)

    def initialize_layer_weights(self, layer, init_function):
        lora_A_q_weights = layer.self_attn.q_proj.lora_A.default.weight
        lora_B_q_weights = layer.self_attn.q_proj.lora_B.default.weight
        lora_A_v_weights = layer.self_attn.v_proj.lora_A.default.weight
        lora_B_v_weights = layer.self_attn.v_proj.lora_B.default.weight
        # lora_A_k_weights = layer.self_attn.k_proj.lora_A.default.weight
        # lora_B_k_weights = layer.self_attn.k_proj.lora_B.default.weight

        if self.init_method != (None or 'pretrained'): #None bcoz, the weights are already initialized with Xavier
            init_function(lora_A_q_weights) #Initializing the weights of A matrices only
            # init_function(lora_B_q_weights) #Not initializing B, bcoz they are already 0
            init_function(lora_A_v_weights)
            # init_function(lora_B_v_weights)
            # init_function(lora_A_k_weights)
        else:
            print("Saving the initial pretrained weights")

        self.init_weights.append((
            lora_A_q_weights.detach().clone(),
            lora_B_q_weights.detach().clone(),
            lora_A_v_weights.detach().clone(),
            lora_B_v_weights.detach().clone(),
            # lora_A_k_weights.detach().clone(),
            # lora_B_k_weights.detach().clone(),
        ))

    def reset(self):
        # setting the current weights to the initial weights for layers within the specified range
        if self.lora_encoder == 'image':
            layers = self.model.vision_model.encoder.layers
        elif self.lora_encoder == 'text':
            layers = self.model.text_model.encoder.layers
            
        for i, layer in enumerate(layers):
            if i in range(self.layer_range[0], self.layer_range[1]+1):
                init_weight_q_A, init_weight_q_B, init_weight_v_A, init_weight_v_B = self.init_weights[i]
                layer.self_attn.q_proj.lora_A.default.weight.data.copy_(init_weight_q_A)
                layer.self_attn.q_proj.lora_B.default.weight.data.copy_(init_weight_q_B)
                layer.self_attn.v_proj.lora_A.default.weight.data.copy_(init_weight_v_A)
                layer.self_attn.v_proj.lora_B.default.weight.data.copy_(init_weight_v_B)
                # layer.self_attn.k_proj.lora_A.default.weight.data.copy_(init_weight_k_A)
                # layer.self_attn.k_proj.lora_B.default.weight.data.copy_(init_weight_k_B)



class PromptLearner(nn.Module):
    """
    Prepare the prompt and include tunable tokens to learn. 

    """
    def __init__(self, clip_model, classnames, batch_size=None, 
                 n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        
        super().__init__()
        #number of classes in the dataset
        n_cls = len(classnames)
        #this is False by default.
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        #get the embedding dimension (512)
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        #batch size
        self.batch_size = batch_size

        if ctx_init:
            # use given words to initialize context vectors. "a photo of a"
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            #make it a photo of a rather than a_photo_of_a
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                #remove [CLS] from the ctx_init if its there.
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            #index of [CLS] if it was there in the prompt. in our case its not there in "a photo of a"
            self.split_idx = split_idx
            #get the number of token, which is 4 in our case
            n_ctx = len(ctx_init.split(" "))
            #prompt now after tokenize will be in shape (1, L) where L is seq len = 77
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                #get the embedding for the prompt, will be in shape (1,77,512)
                embedding = clip_model.token_embedding(prompt).type(dtype)
            #get the embedding for the first 4 tokens (a photo of a), we exclude 1 because its 
            #start_of_token token, we dont need.
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :] #(4,512)
            #ctx vector have a shape of (n_ctx, embed) (4,512). We need to tune this
            prompt_prefix = ctx_init #just the init sentence "a photo of a"
        else:
            # if we want to randomly intiliaze the prompt. Not from "a photo of a"
            print("Random initialization: initializing a generic context")
            # initiliaze random prompt embedding, (4, 512)
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            # the prompt itself
            prompt_prefix = " ".join(["X"] * n_ctx)
        #prompt itself, for example (a photo of a) if ctx_init not random/none
        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            #repeat the vectores batch_size time. if we have more than one sample in the batch_size
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        #Get a copy for the initial embedding of "a photo of a". because we will tune the rest
        ### FIXME: just to keep a copy for the initial embeddings.
        self.ctx_init_state = ctx_vectors.detach().clone()
        # make the vectors as parameter to tune.
        self.ctx = nn.Parameter(ctx_vectors) 

        if not self.learned_cls: #learned_cls is False by default so we will enter here.
            #if the class name has _, replace it by space
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            # retunr "a photo of a {classname}." (baby cray instead of baby_cry)"
            # ['a photo of a dog.', 'a photo of a cat.', 'a photo of a plane.']
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames] # [1 1 1 ...... ] number of classes
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized
        # ['a photo of a dog.', 'a photo of a cat.', 'a photo of a plane.'] take 
        # every prompt and tokenize it for example here we will have (num_classes, 77) 
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            #get embeddings (num_classes, 77, 512). num of classes because we have prompts
            #equal to number of classes
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1,:])  # SOS (start of sentence token)
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS (class name token/s and end of sentence tokens)
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position #default end
        self.n_cls = n_cls #number of classes
        self.n_ctx = n_ctx #number of class tokens (we have it 4 here)
        self.classnames = classnames #class names

    def reset(self):
        ctx_vectors = self.ctx_init_state #embeddings of the "a photo of a" (tunable tokens)
        self.ctx.copy_(ctx_vectors) # to be optimized (we copy the embeddings, changing ctx_vectors will not change self.ctx)
        if self.learned_cls: #default = False
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        #similar to code above.
        self.n_cls = len(classnames) #number of classes
        if not self.learned_cls: # we will enter here
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        #get clip model
        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init 
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls: #will not enter
            assert self.class_token_position == "end"
        if self.class_token_position == "end": #default will enter here
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                ) # we will have the prompt as (n_classes, 1 + n_ctx + *, embed = 512)
                # for each class we have prompt, each prompt SOS+tunable+CLS+EOS
        elif self.class_token_position == "middle": #will not enter here.
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts #retunr the prompts now. (n_classes, SOS + n_ctx + CLS + *, 512)
    
    
class PromptExtractor(nn.Module):
    def __init__(self, n_ctx, ctx_init, classnames, clip_model):
        super().__init__()
        n_ctx = n_ctx
        ctx_init = ctx_init
        dtype = clip_model.dtype
        # ctx_dim = clip_model.ln_final.weight.shape[0]
        # Default is 1, which is compound shallow prompting

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.get_text_features(prompt.to(clip_model.device)).type(dtype)
            prompt_prefix = ctx_init
            
        else:
            # random initialization
            prompt_prefix = " ".join(["X"] * n_ctx)
            print(prompt_prefix)
        self.prompt_prefix = prompt_prefix
            
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        # self.proj = nn.Linear(ctx_dim, 768)
        # self.proj.half()
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.get_text_features(tokenized_prompts.to(clip_model.device)).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding)  # SOS

        # self.n_cls = n_cls
        # self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, prefix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        # print(prefix.shape)

        if label is not None:
            prefix = prefix[label]
        # print(prefix.shape)
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
            ],
            dim=1,
        )
        # print(prompts.shape)
        return prompts

    def forward(self):

        prefix = self.token_prefix
        prompts = self.construct_prompts(prefix)
        # print(prompts.shape)
        return prompts  
    
    def reset_classnames(self, classnames, clip_model):
        #similar to code above.
        self.n_cls = len(classnames) #number of classes
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
            
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(clip_model.device)

        #get clip model

        with torch.no_grad():
            embedding = clip_model.text_encoder.get_text_features(tokenized_prompts).type(clip_model.text_encoder.dtype)
 
        self.token_prefix = embedding

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames 

# Raza
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType #https://medium.com/@manyi.yim/more-about-loraconfig-from-peft-581cf54643db
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel

class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', 
                 arch="ViT-B/16", n_ctx=16, ctx_init=None, ctx_position='end', 
                 learned_cls=False, layer_range=[9, 11], init_method=None, lora_encoder='text',
                 rank=16):
        
        super(ClipTestTimeTuning, self).__init__()
        self.device = device
        self.lora_encoder = lora_encoder
        #load original clip model
        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="a_random_string",
        )
        lora_config.inferece_mode = False

        self.image_encoder = VisionEncoder(clip_model)
        self.text_encoder = PromptEncoder(clip_model)
        
        if lora_encoder == 'image':
            #Apply LoRA to image encoder
            print("Fine-tuning Vision encoder, applying LoRA")
            self.image_encoder = prepare_model_for_int8_training(self.image_encoder)
            self.image_encoder = get_peft_model(self.image_encoder, lora_config) 
            self.image_encoder = self.image_encoder.base_model.model
        elif lora_encoder == 'text':
            #Apply LoRA to text encoder
            print("Fine-tuning Text encoder, applying LoRA")
            self.text_encoder = prepare_model_for_int8_training(self.text_encoder)
            self.text_encoder = get_peft_model(self.text_encoder, lora_config) 
            self.text_encoder = self.text_encoder.base_model.model
        else:
            print("Fine-tuning Prompt Learner, applying LoRA")


        # initializing LoRA_AB for resetting A and B wts after every inference.
        if lora_encoder == 'image':
            self.LoRA_AB = LoRA_AB(self.image_encoder, layer_range=layer_range, init_method=init_method, lora_encoder=lora_encoder)
        if lora_encoder == 'text':
            self.LoRA_AB = LoRA_AB(self.text_encoder, layer_range=layer_range, init_method=init_method, lora_encoder=lora_encoder)
            
        #take the logit scale (learnable) from original CLIP.
        self.logit_scale = clip_model.logit_scale.data
        # prepare prompt.
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        # self.prompt_learner = PromptExtractor(n_ctx, ctx_init, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.criterion = criterion
        
        
        # pretrained_state_dict = torch.load('/home/raza.imam/Documents/TPT/weights/output/imagenet/MaPLe/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots/seed1/MultiModalPromptLearner/model.pth.tar-2')
        # model.load_state_dict(pretrained_state_dict['state_dict'])
    
    def get_model(self):
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        return clip_model
    
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    def LoRA_reset(self):
        self.LoRA_AB.reset()

    # restore the initial state of the prompt_learner (tunable prompt)
    #copy the initial embeddings of "a photo of a" to a seld.ctx
    def reset(self):
        self.prompt_learner.reset()

    #get the prompts embedding from class names. 
    # save the tokenized prompts in self.tokenized_prompts.
    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self):
        # clip_model = self.get_model()
        text_features = []
        # prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        # t_features = self.text_encoder(prompts, tokenized_prompts)
        # t_features = clip_model.get_text_features(tokenized_prompts)
        t_features = self.text_encoder(tokenized_prompts)
        # t_features = self.prompt_learner()
        
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)
        return torch.mean(text_features, dim=0)

    def inference(self, image, label=None, coeff=None):
        # clip_model = self.get_model().requires_grad_(True)
        with torch.no_grad():
            # image_features = self.image_encoder(image.type(self.dtype))
            # image_features = clip_model.get_image_features(image)
            if self.lora_encoder == 'image': # When 'image', text features are computed with NO grad
                self.text_features = self.get_text_features()
            elif self.lora_encoder == 'text': # When 'text', image features are computed with NO grad
                image_features = self.image_encoder(image)

        if self.lora_encoder == 'image': # When 'image', image features are computed with grad enabled
            image_features = self.image_encoder(image)
        elif self.lora_encoder == 'text': # When 'text', text features are computed with grad enabled
            self.text_features = self.get_text_features()
                        
        self.image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        if coeff is not None:
            self.image_features = self.image_features * coeff.view(-1, 1)
            self.image_features = self.image_features.mean(dim=0, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * self.image_features @ self.text_features.t()
        #logits here is the score between prompts embeddings and the image feature
        #scaled logits (n_samples, n_samples) (b_size, b_size)
        
        if self.prompt_learner.training:
            return F.cross_entropy(logits,label)
        
        return logits 

    def forward(self, input, label=None, coeff=None):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input, label, coeff)


def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False, layer_range=[0, 11], init_method=None, lora_encoder='text', rank=16):
    #imagenet not in fewshot.
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    #we are not using bongard dataset
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    #we are here ####
    else:
        classnames = imagenet_classes

    model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls, layer_range=layer_range, init_method=init_method, lora_encoder=lora_encoder)

    return model


class MyClip(torch.nn.Module):
    def __init__(self, clip_model, class_names, templetaes, ensemble, device) -> None:
        super(MyClip, self).__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = clip_model.transformer
        self.class_names = class_names
        self.ensemble = ensemble
        self.device = device
        self.templetaes = templetaes
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.dtype = clip_model.visual.conv1.weight.dtype
        self.text_projection = clip_model.text_projection
        self.prompts_embedding = self.zeroshot_classifier()
        self.logit_scale = clip_model.logit_scale
    def tokenizer(self, texts, context_length = 77, truncate = False): 
        if isinstance(texts, str):
            texts = [texts]

        sot_token = _tokenizer.encoder["<|startoftext|>"]
        eot_token = _tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result

    def zeroshot_classifier(self):
        print(f'Get the embedding of all text prompts')
        with torch.no_grad():
            if self.ensemble:
                    zeroshot_weights = []
                    for classname in tqdm(self.class_names):
                        texts = [template.format(classname) for template in self.templetaes] #format with class
                        texts = self.tokenizer(texts).to(self.device) #tokenize
                        class_embeddings = self.encode_text(texts) #embed with text encoder
                        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                        class_embedding = class_embeddings.mean(dim=0)
                        class_embedding /= class_embedding.norm()
                        zeroshot_weights.append(class_embedding)
                    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
                    return zeroshot_weights
            else: 
                    texts = [f'a photo of a {class_}.' for class_ in  self.class_names ] #format with class
                    texts = self.tokenizer(texts).to(self.device) #tokenize
                    class_embeddings = self.encode_text(texts) #embed with text encoder
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    return class_embeddings.t()
    
    def encode_image(self, image):
        return self.image_encoder(image)

    def encode_text(self, text):
        with torch.no_grad():
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.text_encoder(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)
            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
            return x
    
    def forward(self, images):
        image_features = self.encode_image(images)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale * image_features_norm @ self.prompts_embedding
        return logits
    
    def inference(self, images):
        with torch.no_grad():
            image_features = self.encode_image(images)
            image_features_norm = image_features/ image_features.norm(dim=-1, keepdim=True)
            logits = self.logit_scale * image_features_norm @ self.prompts_embedding
            return logits


# class MyCLIP(CLIP):
#     def __init__(self, original_clip_model, classes, tokenizer, device) -> None:
#         embed_dim = original_clip_model.state_dict()["text_projection"].shape[1]
#         context_length = original_clip_model.state_dict()["positional_embedding"].shape[0]
#         vocab_size = original_clip_model.state_dict()["token_embedding.weight"].shape[0]
#         transformer_width = original_clip_model.state_dict()["ln_final.weight"].shape[0]
#         transformer_heads = transformer_width // 64
#         transformer_layers = len(set(k.split(".")[2] for k in original_clip_model.state_dict() if k.startswith("transformer.resblocks")))
#         vision_width = original_clip_model.state_dict()["visual.conv1.weight"].shape[0]
#         vision_layers = len([k for k in original_clip_model.state_dict().keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
#         vision_patch_size = original_clip_model.state_dict()["visual.conv1.weight"].shape[-1]
#         grid_size = round((original_clip_model.state_dict()["visual.positional_embedding"].shape[0] - 1) ** 0.5)
#         image_resolution = vision_patch_size * grid_size

#         super().__init__(embed_dim= embed_dim, image_resolution=image_resolution, vision_layers=vision_layers, 
#                          vision_width= vision_width, vision_patch_size=vision_patch_size, context_length=context_length, 
#                          vocab_size=vocab_size, transformer_width=transformer_width, transformer_heads=transformer_heads, 
#                          transformer_layers=transformer_layers
#                          )
        
#         self.classes = classes
#         self.tokenizer = tokenizer
#         self.device = device
#         self.original_clip_model = original_clip_model

#     def tokenize(self, texts, context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
        
#         if isinstance(texts, str):
#             texts = [texts]

#         sot_token = self.tokenizer.encoder["<|startoftext|>"]
#         eot_token = self.tokenizer.encoder["<|endoftext|>"]
#         all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
#         result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

#         for i, tokens in enumerate(all_tokens):
#             if len(tokens) > context_length:
#                 if truncate:
#                     tokens = tokens[:context_length]
#                     tokens[-1] = eot_token
#                 else:
#                     raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
#             result[i, :len(tokens)] = torch.tensor(tokens)

#         return result

#     def get_text_features(self):
#         tokenized_prompts = self.tokenize(texts= self.classes)
#         text = tokenized_prompts.to(device= self.device)
#         return self.encode_text(text)
    
#     def forward(self, image):
#         image_features = self.encode_image(image)
#         text_features = self.get_text_features()

#         # normalized features
#         self.image_features = image_features / image_features.norm(dim=1, keepdim=True)
#         self.text_features = text_features / text_features.norm(dim=1, keepdim=True)

#         # cosine similarity as logits
#         logit_scale = self.original_clip_model.logit_scale.exp()
#         logits_per_image = logit_scale * self.image_features @ self.text_features.t()
        
#         # shape = [global_batch_size, global_batch_size]
#         return logits_per_image
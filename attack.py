
import clip_attack
import foolbox as fb
import torch.nn as nn
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
    
class Attack(): 
    """
    This class used to generate adversarial images.
    when create object specify  epsilon: float, attack_type: 'FGSM, CW, BIM, L2PGD, PGD, LinfBIM'. 

    generate method return images and success tensors. 
    test_model method, give the accuracy of the model after passing the adversarial examples. 

    succecces tensor shows whether the example succed to fool the model or not

    """
    def __init__(self, epsilon, attack_type, model, bounds, device, preprocess, clip_flag = False, steps = 40) :
        self.epsilon= epsilon
        self.attack_type = attack_type
        self.model_fool = fb.models.PyTorchModel(model , bounds=bounds, device= device, 
                                                 preprocessing= preprocess
                                                 ) 
        
        if self.attack_type == 'FGSM':
            if not clip_flag:
                self.attack_func = fb.attacks.FGSM()
            else: 
                self.attack_func = clip_attack.FGSM_clip()
        elif self.attack_type == 'L2PGD': 
            self.attack_func = fb.attacks.L2PGD()
        elif self.attack_type == 'CW': 
            self.attack_func = fb.attacks.L2CarliniWagnerAttack(6,1000,0.01,0)
        elif self.attack_type == 'BIM': 
            self.attack_func = fb.attacks.L2BasicIterativeAttack()
        elif self.attack_type == 'PGD':
            if not clip_flag:
                self.attack_func = fb.attacks.PGD(steps= steps)
            else: 
                self.attack_func = clip_attack.PGD_clip(steps= steps)
        elif self.attack_type =='LinfBIM':         
            self.attack_func = fb.attacks.LinfBasicIterativeAttack() 

    def FGSM(self, samples, labels):
        """
        Generate FGSM attacks. 
        Args: 
            samples -> clean images 
            labels -> labels of clean images  

        return:
            adversarial images generated from the clean images 
            success tensor shows whether the attack succeded in fooling the model or not

        """
        _, adv_images, success = self.attack_func(self.model_fool,
                                            samples,
                                            labels,
                                            epsilons = self.epsilon)
        return adv_images, success   
        
    def L2PGD(self, samples, labels): 
        """
        Generate L2 PGD attacks. 
        Args: 
            samples -> clean images 
            labels -> labels of clean images  

        return:
            adversarial images generated from the clean images 
            success tensor shows whether the attack succeded in fooling the model or not

        """

        _, adv_images, success = self.attack_func(self.model_fool,
                                            samples,
                                            labels,
                                            epsilons = self.epsilon)
        return adv_images, success
                                            

    def CW(self, samples, labels): 
        """
        Generate Carlini & Wagner attacks. 
        Args: 
            samples -> clean images 
            labels -> labels of clean images  

        return:
            adversarial images generated from the clean images 
            success tensor shows whether the attack succeded in fooling the model or not

        """

        _, adv_images, success = self.attack_func(self.model_fool,
                                            samples,
                                            labels,
                                            epsilons= self.epsilon)
        print(f'Sum = {sum(success)}')  
        return adv_images, success

    def BIM(self, samples, labels):
        """
        Generate BIM attacks. 
        Args: 
            samples -> clean images 
            labels -> labels of clean images  

        return:
            adversarial images generated from the clean images 
            success tensor shows whether the attack succeded in fooling the model or not

        """

        _, adv_images, success = self.attack_func(self.model_fool,
                                            samples,
                                            labels,
                                            epsilons = self.epsilon)
        return adv_images, success 
    
    def PGD(self, samples, labels):
        """
        Generate Linf PGD attacks. 
        Args: 
            samples -> clean images 
            labels -> labels of clean images  

        return:
            adversarial images generated from the clean images 
            success tensor shows whether the attack succeded in fooling the model or not

        """

        _, adv_images, success = self.attack_func(self.model_fool,
                                            samples,
                                            labels,
                                            epsilons = self.epsilon)
        return adv_images, success

    def LinfBIM(self, samples, labels):
        """
        Generate Linf BIM attacks. 
        Args: 
            samples -> clean images 
            labels -> labels of clean images  

        return:
            adversarial images generated from the clean images 
            success tensor shows whether the attack succeded in fooling the model or not

        """

        _, adv_images, success = self.attack_func(self.model_fool,
                                            samples,
                                            labels,
                                            epsilons = self.epsilon)
        return adv_images, success  

    def generate_attack(self, samples, labels):
        """
        Generate attacks. 
        Args: 
            samples -> clean images 
            labels -> labels of clean images  

        return:
            adversarial images -> generated from the clean images 
            success tensor -> shows whether the attack succeded in fooling the model or not

        """

        if self.attack_type == 'FGSM': 
            adv_img, success = self.FGSM(samples, labels)
        elif self.attack_type == 'CW': 
            adv_img, success = self.CW(samples, labels)           
        elif self.attack_type == 'L2PGD': 
            adv_img, success = self.L2PGD(samples, labels)
        elif self.attack_type == 'BIM': 
            adv_img, success = self.BIM(samples, labels)
        elif self.attack_type == 'PGD':
            adv_img, success = self.PGD(samples, labels)
        elif self.attack_type =='LinfBIM': 
            adv_img, success = self.LinfBIM(samples, labels)
        else: 
            print(f'Attacks of type {self.attack_type} is not supported') 
        return adv_img, success


class AttackART(): 
    def __init__(self, epsilon, attack_type, bounds, device) :
        self.epsilon= epsilon
        self.attack_type = attack_type
        self.criterion = nn.CrossEntropyLoss()
        self.bounds = bounds
        self.device = device

    def PGD(self, samples, labels, model):
        """
        Generate Linf PGD attacks. 
        Args: 
            samples -> clean images 
            labels -> labels of clean images  

        return:
            adversarial images generated from the clean images 
            success tensor shows whether the attack succeded in fooling the model or not

        """

        classifier = PyTorchClassifier(
            model=model.cpu(),
            clip_values=(self.bounds[0].cpu(), self.bounds[1].cpu()),
            loss=self.criterion,
            optimizer=None,
            input_shape=(3, 224, 224),
            nb_classes=10000,
            device_type= self.device
        )
        attack = ProjectedGradientDescentPyTorch(estimator= classifier, eps= self.epsilon, batch_size=1)
        x_test_adv = attack.generate(x= samples)

        return x_test_adv

    def generate_adv(self, samples, labels, model):
        if self.attack_type == 'PGD': 
            adv_sample = self.PGD(samples=samples.cpu().numpy(), labels=labels.cpu().numpy(), model=model.cpu())
        return adv_sample
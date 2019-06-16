from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from data_loader import adversarial_data_loaders
from models.vgg import vgg19_bn
from tqdm import tqdm
import os

# hyper-parameters
epsilons = [.0, .001, .005, .01, .05]
pretrained_model = "results/vgg.pth"
save_path = "adversarial_sample/"
use_cuda = True

uploader = transforms.ToPILImage()
train_loader, val_loader, test_loader = adversarial_data_loaders()
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# load pre-trained VGG model
model = vgg19_bn(num_classes=7)
model = model.to(device)
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
model.eval()

# generate an director to store adversarial samples
if not os.path.exists("adversarial_sample"):
    os.mkdir("adversarial_sample")
    for i in range(7):
        os.mkdir("adversarial_sample/"+str(i))

# perform FGSM attack
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

# generate adversarial samples of the training dataset
def generate_samples(model, device, train_loader, epsilon):
    cnt = 0
    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        adv_ex = perturbed_data.squeeze().detach().cpu()
        image = uploader(adv_ex)
        plt.imsave(save_path+str(target.item())+'/image'+str(cnt)+'.png',image)
        cnt += 1

# pick some results of FGSM attack as demo
def demo_samples(model, device, train_loader, epsilons):
    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        for epsilon in epsilons:
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
            adv_ex = perturbed_data.squeeze().detach().cpu()
            image = uploader(adv_ex)
            plt.imshow(image)
            plt.show()
        break

eps = 0.01
generate_samples(model, device, train_loader, eps)
# demo_samples(model,device,train_loader,epsilons)

# SJTU_CS420_MachineLearning
final project

author: Wang Lingdong, Dai Jiacheng

# Download

You can download structured fer2013 dataset and our VGG19bn model here:

链接：https://pan.baidu.com/s/1n5WNvj3IebnBdpaHTpmQ-Q 
提取码：spyl 


# Description
## Model Structure
*models/* 
contains model structures including AlexNet, MobileNet, ResNet and VGG

## Utilization
*data_loader.py* 
provides dataset loaders

*train.py* 
provides training and testing processes


## Baseline model
*train_alexnet.py*
trains and tests AlexNet

*train_mobilenet.py*
trains and tests MobileNet

*train_vgg.py*
trains and tests VGG19 with batch normalization

*train resnet.py*
trains and tests ResNet152

*train_resnet18.py*
trains and tests ResNet18

## Learning Rate Scheduler
*schedule_vgg.py*
explores the effect of multi-step learning rate scheduler

## Optimizer
*sgd_vgg.py*
explores the effect of SGD optimizer

*rms_vgg.py*
explores the effect of RMSprop optimizer

## Data Augmentation
*augment_vgg.py*
explores the effect of data augmentation

## Adversarial Sample
*fgsm_vgg.py*
tests the performance of FGSM attack

*adversarial_sample_vgg.py*
generates FGSM adversarial samples 

*retrain_vgg.py*
retains VGG model with adversarial samples to defend FGSM attack

## Pre-trained Model
*pretrained_vgg.py*
trains and tests VGG based on an ImageNet pre-trained model 

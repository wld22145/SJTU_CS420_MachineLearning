import torch
from models.vgg import vgg19_bn
from data_loader import augment_data_loaders
import copy
from train import solve
from torch.optim import Adam

# hyper-parameters
epoch = 200
test_epoch = 5
batch_size = 64

train_loader, val_loader, test_loader = augment_data_loaders(batch_size)
model = vgg19_bn(pretrained=False,num_classes=7)
model = model.cuda()
optimizer = Adam(model.parameters())

# train and test
train_loss, test_loss, test_accuracy = solve(model,train_loader,val_loader,optimizer,epoch,test_epoch)

# output log
file = open("results/augment_vgg19bn_log.txt","w")

content = ''
content += "train loss: ["
for element in train_loss:
    content += str(element)
    content += ", "
content = content[:-2]
content += "]\n"
file.write(content)

content = ''
content += "test loss: ["
for element in test_loss:
    content += str(element)
    content += ", "
content = content[:-2]
content += "]\n"
file.write(content)

content = ''
content += "test acc: ["
for element in test_accuracy:
    content += str(element)
    content += ", "
content = content[:-2]
content += "]\n"
file.write(content)

file.close()
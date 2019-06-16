import torch
from models.vgg import vgg19_bn
from data_loader import get_data_loaders
import copy
from train import solve
from torch.optim import SGD

# hyper-parameters
epoch = 200
test_epoch = 5
batch_size = 64

train_loader, val_loader, test_loader = get_data_loaders(batch_size)
model = vgg19_bn(pretrained=False,num_classes=7)
model = model.cuda()

# use SGD optimizer
optimizer = SGD(model.parameters(),lr=0.001)

# train and test
train_loss, test_loss, test_accuracy = solve(model,train_loader,val_loader,optimizer,epoch,test_epoch)

# output log
file = open("results/adam_vgg19bn_log.txt","w")

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
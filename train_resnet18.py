import torch
from models.resnet import resnet18
from data_loader import get_data_loaders
import copy
from train import solve,test
from torch.optim import Adam

# hyper-parameters
epoch = 200
test_epoch = 5
batch_size = 64

train_loader, val_loader, test_loader = get_data_loaders(batch_size)
model = resnet18(num_classes=7)
model = model.cuda()
optimizer = Adam(model.parameters())

# train on training dataset
# test on validation dataset
train_loss, val_loss, val_accuracy = solve(model, train_loader, val_loader, optimizer, epoch, test_epoch)

# final test on testing dataset
test_loss, test_accuracy = test(model, test_loader)
test_loss = [test_loss]
test_accuracy = [test_accuracy]
file = open("results/resnet18_log.txt","w")

# output log
content = ''
content += "train loss: ["
for element in train_loss:
    content += str(element)
    content += ", "
content = content[:-2]
content += "]\n"
file.write(content)

content = ''
content += "val loss: ["
for element in val_loss:
    content += str(element)
    content += ", "
content = content[:-2]
content += "]\n"
file.write(content)

content = ''
content += "val acc: ["
for element in val_accuracy:
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


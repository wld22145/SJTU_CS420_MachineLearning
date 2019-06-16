import torch
from models.vgg import vgg19_bn
from data_loader import get_data_loaders, adversarial_data_loaders
import copy
from train import train, test
from torch.optim import Adam
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# hyper-parameters
epoch = 50
test_epoch = 1
batch_size = 64
epsilon = 0.01

# initialization
_, _, adv_test_loader = adversarial_data_loaders()
pretrained_model = "results/vgg.pth"
use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
train_loader, val_loader, test_loader = get_data_loaders(batch_size)
model = vgg19_bn(num_classes=7)
model = model.to(device)
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
model.eval()
optimizer = Adam(model.parameters())
adversarial_set = torchvision.datasets.ImageFolder("adversarial_sample/", transform=transforms.Compose([transforms.ToTensor()]))
adversarial_loader = DataLoader(adversarial_set, batch_size=batch_size, shuffle=True)

# FGSM attack
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

# perform FGSM attack to test dataset
def adv_test(model, device, test_loader, epsilon):
    correct = 0
    adv_examples = []
    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        # generate gradient for input image
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        # perform FGSM attack with gradient of input image
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            # save some figures as demo
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    return final_acc, adv_examples

# retrain VGG model with adversarial samples
train_loss = []
test_loss = []
adversarial_loss = []
test_accuracy = []
test_cnt = 0
adv_test_acc = []
for e in range(epoch):
    print("epoch",e)
    # retrain with original data
    avg_loss = train(model,train_loader,optimizer)
    train_loss.append(avg_loss)
    # retrain with adversarial samples
    avg_loss = train(model,adversarial_loader,optimizer)
    adversarial_loss.append(avg_loss)
    test_cnt += 1
    if test_cnt == test_epoch:
        test_cnt = 0
        # test on original data
        avg_loss,accuracy = test(model,val_loader)
        test_loss.append(avg_loss)
        test_accuracy.append(accuracy)
        # test with FGSM attack
        adv_acc,_ = adv_test(model,device,adv_test_loader,epsilon)
        adv_test_acc.append(adv_acc)

# output log
file = open("results/vgg_log.txt","w")

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


content = ''
content += "adv train loss: ["
for element in adversarial_loss:
    content += str(element)
    content += ", "
content = content[:-2]
content += "]\n"
file.write(content)

content = ''
content += "adv test acc: ["
for element in adv_test_acc:
    content += str(element)
    content += ", "
content = content[:-2]
content += "]\n"
file.write(content)

file.close()
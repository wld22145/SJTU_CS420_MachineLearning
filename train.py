import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm

# train and test for lr scheduler
def schedule_solve(model,train_loader,test_loader,optimizer,epoch,test_epoch,scheduler):
    train_loss = []
    test_loss = []
    test_accuracy = []
    test_cnt = 0
    for e in range(epoch):
        print("epoch", e)
        avg_loss = train(model, train_loader, optimizer)
        train_loss.append(avg_loss)
        scheduler.step()
        test_cnt += 1
        if test_cnt == test_epoch:
            test_cnt = 0
            avg_loss, accuracy = test(model, test_loader)
            test_loss.append(avg_loss)
            test_accuracy.append(accuracy)
    return train_loss, test_loss, test_accuracy

# train and test
def solve(model,train_loader,test_loader,optimizer,epoch,test_epoch):
    train_loss = []
    test_loss = []
    test_accuracy = []
    test_cnt = 0
    for e in range(epoch):
        print("epoch",e)
        avg_loss = train(model,train_loader,optimizer)
        train_loss.append(avg_loss)
        test_cnt += 1
        if test_cnt == test_epoch:
            test_cnt = 0
            avg_loss,accuracy = test(model,test_loader)
            test_loss.append(avg_loss)
            test_accuracy.append(accuracy)
    return train_loss, test_loss, test_accuracy

# train for an epoch
def train(model,data_loader,optimizer):
    model.train()
    loss_func = nn.CrossEntropyLoss().cuda()
    loss_record = []
    for image,label in tqdm(data_loader):
        optimizer.zero_grad()
        image, label = image.cuda(), label.cuda()
        output = model(image)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()
        loss_record.append(float(loss.cpu().detach().item()))
    avg_loss = torch.mean(torch.Tensor(loss_record))
    avg_loss = avg_loss.item()
    print("Train Loss:",avg_loss)
    return avg_loss

# test
def test(model,data_loader):
    model.eval()
    loss_func = nn.CrossEntropyLoss().cuda()
    loss_record = []
    correct = 0
    for image,label in tqdm(data_loader):
        image, label = image.cuda(), label.cuda()
        output = model(image)
        loss = loss_func(output, label)
        loss_record.append(float(loss.cpu().detach().item()))
        pred = output.detach().max(1)[1]
        correct += pred.eq(label.view_as(pred)).sum()
    accuracy = float(correct) / len(data_loader.sampler)
    avg_loss = torch.mean(torch.Tensor(loss_record))
    avg_loss = avg_loss.item()
    print("Test Loss:", avg_loss)
    print("Test Acc:",accuracy)
    return avg_loss,accuracy
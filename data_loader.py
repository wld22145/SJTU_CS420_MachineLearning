import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

path = "datasets/datasets/"
train_path = path + "train"
val_path = path + "val"
test_path = path + "test"

# provide data loaders for train, validation and test datasets
def get_data_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_set = torchvision.datasets.ImageFolder(train_path,transform=transform)
    val_set = torchvision.datasets.ImageFolder(val_path,transform=transform)
    test_set = torchvision.datasets.ImageFolder(test_path,transform=transform)

    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False)

    return train_loader,val_loader,test_loader

# provide data loaders for data augmentation
# with random vertical filp and normalization
def augment_data_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_set = torchvision.datasets.ImageFolder(train_path, transform=transform)
    val_set = torchvision.datasets.ImageFolder(val_path, transform=transform)
    test_set = torchvision.datasets.ImageFolder(test_path, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# provide data loaders for pre-trained model
def resized_data_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_set = torchvision.datasets.ImageFolder(train_path, transform=transform)
    val_set = torchvision.datasets.ImageFolder(val_path, transform=transform)
    test_set = torchvision.datasets.ImageFolder(test_path, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# provide data loaders for adversarial sample experiment
def adversarial_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = torchvision.datasets.ImageFolder(train_path, transform=transform)
    val_set = torchvision.datasets.ImageFolder(val_path, transform=transform)
    test_set = torchvision.datasets.ImageFolder(test_path, transform=transform)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader

# unit test
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders()
    for img,label in train_loader:
        print(img.size())
        print(label.size())
        break
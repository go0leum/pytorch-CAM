import torch, os
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
import numpy as np


def dataSetLoad(DATA_SET_DIR, TRAIN_TEST_SPLIT, IMG_SIZE, BATCH_SIZE):
    
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose([
        #transforms.RandomResizedCrop(IMG_SIZE),
        transforms.CenterCrop((IMG_SIZE,IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0, saturation=0, hue=0),
        transforms.ToTensor(),
        normalize
    ])
    
    data_set = datasets.ImageFolder(DATA_SET_DIR, transform=transform)
    
    test_size = int(TRAIN_TEST_SPLIT * len(data_set))
    train_size = len(data_set) - test_size

    train_dataset, test_dataset = random_split(data_set, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle = True)
    
    return data_set, train_loader, test_loader

def checkpointLoad(CHECK_POINT_DIR, RESUME):
    print("===> Resuming from checkpoint.")
    assert os.path.isfile(CHECK_POINT_DIR+'/'+ str(RESUME) + '.pt'), 'Error: no checkpoint found!'
    checkpoint = torch.load(CHECK_POINT_DIR+'/' + str(RESUME) + '.pt')
    
    return checkpoint
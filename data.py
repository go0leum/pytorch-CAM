import torch, os
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from itertools import product


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

def batch_window(img_pil, batch, stepSize, windowSize):
    width, height= img_pil.size
    
    n_y = (height//stepSize)+(height%stepSize!=0)
    n_x = (width//stepSize)+(width%stepSize!=0)
    
    total_iter = (n_y*n_x//batch)+(n_y*n_x%batch!=0)
    last_batch = n_y*n_x%batch
    
    windows = []
    y_list = []
    windows = []
    i = 1
    
    totensor = transforms.ToTensor()
    
    for y, x in product(range(0, n_y), range(0, n_x)):
        x_i = x*stepSize
        y_i = y*stepSize
        x_f = x_i+windowSize
        y_f = y_i+windowSize
        
        pil_crop = img_pil.crop((x_i,y_i,x_f,y_f))
        tensor_crop = totensor(pil_crop)
        y_list.append(tensor_crop)
        
        if len(y_list)==batch or (i==total_iter and len(y_list)==last_batch):
            y_tensor = torch.stack(y_list)
            windows.append(y_tensor)
            y_list = []
            i+=1
            
    return windows
            

def WindowDataLoad(cam_data_dir, batch, step_size, window_size):
    
    pad = transforms.Pad(padding=step_size, padding_mode='reflect')
    
    for _, _, f in os.walk(cam_data_dir):
        for file in f:
            if '.png' not in file:
                continue
            root = os.path.join(cam_data_dir, file)
            img = Image.open(root)
            
            img_pil=pad(img)
            
            windows = batch_window(img_pil, batch, step_size, window_size)
            
            yield(root, img, windows)
                
def camDataLoad(cam_data_dir):
    
    for _, _, f in os.walk(cam_data_dir):
        for file in f:
            if '.png' not in file:
                continue
            root = os.path.join(cam_data_dir, file)
            img = Image.open(root)
            
            yield(root, img)
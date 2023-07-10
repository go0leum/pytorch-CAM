"""
Class Activation Mapping
Googlenet, Kaggle data
"""

from update import *
from data import *
from train import *
import torch, os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from inception import inception_v3


# functions
CAM             = 1
USE_CUDA        = 1
RESUME          = 10
PRETRAINED      = 1


# hyperparameters
BATCH_SIZE      = 32
IMG_SIZE        = 256
LEARNING_RATE   = 0.0001
EPOCH           = 5


# prepare data
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
)

transform_train = transforms.Compose([
    #transforms.RandomResizedCrop(IMG_SIZE),
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    #transforms.Resize(256),
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    normalize
])

train_data = datasets.ImageFolder('fire_detection/train/', transform=transform_train)
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

test_data = datasets.ImageFolder('fire_detection/test/', transform=transform_test)
testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# class
classes = {0: 'fire', 1: 'nonfire'}


# fine tuning
if PRETRAINED:
    net = inception_v3(pretrained=PRETRAINED)
    for param in net.parameters():
        param.requires_grad = False
    net.fc = torch.nn.Linear(2048, len(classes))
else:
    net = inception_v3(pretrained=PRETRAINED, num_classes=len(classes))
final_conv = 'Mixed_7c'

net.cuda()


# load checkpoint
if RESUME != 0:
    print("===> Resuming from checkpoint.")
    assert os.path.isfile('checkpoint/'+ str(RESUME) + '.pt'), 'Error: no checkpoint found!'
    net.load_state_dict(torch.load('checkpoint/' + str(RESUME) + '.pt'))


# retrain
criterion = torch.nn.CrossEntropyLoss()

if PRETRAINED:
    #optimizer = torch.optim.SGD(net.fc.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(net.fc.parameters(), lr=LEARNING_RATE)
else:
    #optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

for epoch in range (1, EPOCH + 1):
    retrain(trainloader, net, USE_CUDA, epoch, criterion, optimizer)
    retest(testloader, net, USE_CUDA, criterion, epoch, RESUME)


# hook the feature extractor
features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(final_conv).register_forward_hook(hook_feature)


# CAM
if CAM:
    #root = 'sample.jpg'
    #img = Image.open(root)
    #get_cam(net, features_blobs, img, classes, root)
    fire_sum=0
    nonfire_sum=0
    nonfire=-1;
    for _, _, f in os.walk('data'):
        for file in f:
            if '.png' not in file:
                continue
            root = os.path.join('data', file)
            img = Image.open(root)
            features_blobs = []
            net._modules.get(final_conv).register_forward_hook(hook_feature)
            nonfire = get_cam(net, features_blobs, img, classes, root, IMG_SIZE)

            #count fire, nonfire
            if nonfire==0:
                fire_sum+=1
            elif nonfire==1:
                nonfire_sum+=1
    
    print('output CAM.jpg total fire : {0}, nonfire : {1}'.format(fire_sum, nonfire_sum))
"""
Class Activation Mapping
Googlenet, Kaggle data
"""

from inception import inception_v3
from data import *
from train import *
from update import *
import torch, os
import time
from PIL import Image


#data directory
DATA_SET_DIR         = 'fire_data'          #train+test 데이터셋 디렉토리
CHECK_POINT_DIR      = 'fire_checkpoint'    #checkpoint저장할 디렉토리
PREDCICTION_DATA_DIR = 'cam_fire_data'      #cam 할 데이터셋 디렉토리, cam 결과는 res_cam_fire_data
# FAULT_DATA_DIR       = 'cam_fault_data'     #predition 실패한 cam 결과 저장할 디렉토리

#hyperparameter
TRAIN_TEST_SPLIT     = 0.2
BATCH_SIZE           = 64
IMG_SIZE             = 224
LEARNING_RATE        = 0.001
EPOCH                = 100

# functions
CAM                  = 1     #CAM 사용 여부 CAM종류(CAM:1, Grad CAM:2, other CAM:3)|0
USE_CUDA             = 1     #GPU 사용 여부 1|0
RESUME               = 0     #checkpoint 중 사용할 번호
PRETRAINED           = 1     #finetuning 사용 여부 1|0
K_FOLD               = 0
SLIDING_WINDOW       = 1

#sliding window
WINDOW_SIZE          = 224
STEP_SIZE            = 112

#class
classes = {0: 'fire', 1: 'nonfire'}

#data load
total_loader, train_loader, test_loader = dataSetLoad(DATA_SET_DIR, TRAIN_TEST_SPLIT, IMG_SIZE, BATCH_SIZE)

#fine tuning
if PRETRAINED:
    net = inception_v3(pretrained=PRETRAINED)
    net.fc = torch.nn.Linear(2048, len(classes))
else:
    net = inception_v3(pretrained=PRETRAINED, num_classes=len(classes))
final_conv = 'Mixed_7c'

net.cuda()

#load checkpoint
if RESUME !=0:
    checkpoint = checkpointLoad(CHECK_POINT_DIR, RESUME)
    net.load_state_dict(checkpoint)

#train_test
if PRETRAINED:
    #optimizer = torch.optim.SGD(net.fc.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(net.fc.parameters(), lr=LEARNING_RATE)
else:
    #optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
criterion = torch.nn.CrossEntropyLoss()

if K_FOLD>0:
    k_fold(K_FOLD, total_loader,net, USE_CUDA, EPOCH, criterion, optimizer, BATCH_SIZE, CHECK_POINT_DIR)
else:
    writer = SummaryWriter('loss_accuracy')
    
    for epoch in range (1, EPOCH + 1):
        
        acc_avg, loss_avg = retrain(train_loader, net, USE_CUDA, epoch, criterion, optimizer, writer)
        dict, test_acc, test_loss= retest(test_loader, net, USE_CUDA, epoch, criterion, writer)
        
        # Save checkpoint.
        torch.save(dict, CHECK_POINT_DIR+'/' + str(RESUME + epoch) + '.pt')
        
    writer.close()


# hook the feature extractor
features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(final_conv).register_forward_hook(hook_feature)

#CAM(prediction)
if CAM==1:
    #root = 'sample.jpg'
    #img = Image.open(root)
    #get_cam(net, features_blobs, img, classes, root)
    fire_sum=0
    nonfire_sum=0
    res=-1
    
    start = time.process_time()
    for _, _, f in os.walk(PREDCICTION_DATA_DIR):
        for file in f:
            if '.png' not in file:
                continue
            root = os.path.join(PREDCICTION_DATA_DIR, file)
            img_pil = Image.open(root)
            
            if SLIDING_WINDOW:
                fire=1
                for (x,y,window_pil) in sliding_window(img_pil, STEP_SIZE, WINDOW_SIZE):
                    
                    if window_pil.width<WINDOW_SIZE or window_pil.height<WINDOW_SIZE:
                        continue
                    features_blobs = []
                    net._modules.get(final_conv).register_forward_hook(hook_feature)
                    
                    fire, result_img = get_cam(net, features_blobs, window_pil, classes, 'window_('+str(x)+','+str(y)+')', IMG_SIZE)
                    
                    root_img=root.replace('.png','_('+str(x)+','+str(y)+')_'+classes[fire]+'.png')
                    
                    if fire==0:
                        res=0
                    
                    cv2.imwrite('res_' + root_img, result_img)
            else:
                features_blobs = []
                net._modules.get(final_conv).register_forward_hook(hook_feature)

                #nonfire = get_cam(net, features_blobs, img, classes, root, IMG_SIZE)
                res, result_img = get_cam(net, features_blobs, img_pil, classes, root, IMG_SIZE)
                
                cv2.imwrite('res_' + root, result_img)

            #count fire, nonfire
            if res==0:
                fire_sum+=1
            else :
                nonfire_sum+=1
                
    time = time.process_time()-start
    print('\nprocess time for inference : %d ms'%(time*1000/100))
    print('\noutput CAM.jpg total fire : {0}, nonfire : {1}'.format(fire_sum, nonfire_sum))
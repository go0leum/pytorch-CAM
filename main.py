"""
Class Activation Mapping
Googlenet, Kaggle data
"""

from inception import inception_v3
from data import *
from train import *
from update import *
import time
from datetime import datetime

#data directory
DATA_SET_DIR         = 'fire_data'          #train+test 데이터셋 디렉토리
CHECK_POINT_DIR      = 'checkpoint'         #checkpoint저장할 디렉토리
PREDCICTION_DATA_DIR = 'data'               #cam 할 데이터셋 디렉토리, cam 결과는 res_cam_fire_data
VAL_VIDEO_DIR        = 'data/test.mov'

#hyperparameter
TRAIN_TEST_SPLIT     = 0.2
BATCH_SIZE           = 64
IMG_SIZE             = 224
LEARNING_RATE        = 0.001
EPOCH                = 0

# functions
CAM                  = 1     #CAM 사용 여부 CAM종류(CAM:1, Grad CAM:2, other CAM:3)|0
USE_CUDA             = 1     #GPU 사용 여부 1|0
RESUME               = 57     #checkpoint 중 사용할 번호
PRETRAINED           = 0     #finetuning 사용 여부 1|0
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

if EPOCH != 0:
    writer = SummaryWriter('loss_accuracy')
    
    for epoch in range (1, EPOCH + 1):
        
        acc_avg, loss_avg = retrain(train_loader, net, USE_CUDA, epoch, criterion, optimizer, writer)
        dict, test_acc, test_loss= retest(test_loader, net, USE_CUDA, epoch, criterion, writer)
        
        # Save checkpoint.
        torch.save(dict, CHECK_POINT_DIR+'/' + str(RESUME + epoch) + '.pt')
        
    writer.close()
        
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
net._modules.get(final_conv).register_forward_hook(hook_feature)

fps = 15
cam_index = 0
frame_width = 1280
frame_height = 720

#CAM(prediction)
if CAM==1:
    fire_sum=0
    nonfire_sum=0
    if SLIDING_WINDOW:
        for (img, windows) in VideoInput(cam_index, frame_width, frame_height, fps, BATCH_SIZE, STEP_SIZE, WINDOW_SIZE):
            
            res=-1
            features_blobs = []
            img_set = windows
            
            start = datetime.now()
            res, result_img = get_cam_window(net, features_blobs, img, windows, classes, WINDOW_SIZE)
            end = datetime.now()
            elapsed_time = end-start
            max_sleep = 1/fps
            
            cv2.imshow("fire detection test", result_img)
            
            if max_sleep > elapsed_time.total_seconds():
                cv2.waitKey((max_sleep - elapsed_time.total_seconds())*10)
            else:
                cv2.waitKey(1)
            
            if res==0:
                cv2.imwrite('res_data/fire_%06.d.png'%fire_sum, result_img)
                fire_sum+=1
            else :
                nonfire_sum+=1
    else:
        for (root, img) in camDataLoad(PREDCICTION_DATA_DIR):
            start = datetime.now()
            
            features_blobs = []
            net._modules.get(final_conv).register_forward_hook(hook_feature)

            res, result_img = get_cam(net, features_blobs, img, classes, root, IMG_SIZE)
            
            root_img=root.replace('.png','_'+classes[res]+'.png')
            cv2.imwrite('res_' + root_img, result_img)
            
            end = datetime.now()
            elapsed_time = end-start
            print('\nprocess time for inference : %d ms'%(elapsed_time*1000/100))
            
            if res==0:
                fire_sum+=1
            else :
                nonfire_sum+=1
    print('\noutput CAM.jpg total fire : {0}, nonfire : {1}'.format(fire_sum, nonfire_sum))
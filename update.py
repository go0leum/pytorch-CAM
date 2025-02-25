from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import torch
from itertools import product
import scipy.signal as signal
        
def spline_window(wnd_sz, power=2):
    '''
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    '''
    intersection = int(wnd_sz/4)
    wnd_outer = (abs(2*(signal.triang(wnd_sz))) ** power)/2
    wnd_outer[intersection:-intersection] = 0

    wnd_inner = 1 - (abs(2*(signal.triang(wnd_sz) - 1)) ** power)/2
    wnd_inner[:intersection] = 0
    wnd_inner[-intersection:] = 0

    wnd = wnd_inner + wnd_outer
    wnd = wnd / np.average(wnd)
    return wnd

cached_2d_windows = dict()

def window_2d(wnd_sz, power=2):
    '''
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    '''
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(wnd_sz, power)
    if key in cached_2d_windows:
        wnd = cached_2d_windows[key]
    else:
        wnd = spline_window(wnd_sz, power)
        wnd = np.expand_dims(np.expand_dims(wnd, -1), -1)
        wnd = wnd * wnd.transpose(1, 0, 2)
        cached_2d_windows[key] = wnd
    return wnd

def merge_window(height, width, cam_set, window_size):
    height += window_size
    width += window_size
    
    weights_each = window_2d(window_size, power=2).squeeze()
    CAM = np.zeros((height, width))
    divisor = np.zeros((height, width))
    i=0
    
    n_y = (height//112)+(height%112!=0)
    n_x = (width//112)+(width%112!=0)
    
    for y, x in product(range(0, n_y), range(0, n_x)):
        x_i = x*112
        y_i = y*112
        x_f = x_i+window_size
        y_f = y_i+window_size
        
        h, w = CAM[y_i:y_f, x_i:x_f].shape
        CAM[y_i:y_f, x_i:x_f] += (cam_set[i,0:h,0:w]*weights_each[0:h,0:w])
        divisor[y_i:y_f,x_i:x_f] += weights_each[0:h,0:w]
        i+=1
    
    CAM = np.uint8(CAM/divisor)
    CAM = CAM[112:height-112, 112:width-112]
    return CAM

# generate class activation mapping for the top1 prediction
def return_cam(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam[cam<0] = 0
        cam_img = cam / 18.5
        cam[cam>1] = 1
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def get_cam_window(net, features_blobs, img, windows, classes, window_size):
    net.eval()
    
    height, width, _= img.shape
    res_list = []
    res = 0
    
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    for i in range(len(windows)):
        img_tensor = normalize(windows[i])
        img_variable = Variable(img_tensor).cuda()
        logit = net(img_variable)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(1, True)
        
        res += (idx[:,0].size()[0] - int(torch.count_nonzero(idx[:,0])))
        
        for batch in range(img_tensor.size()[0]):
            features_conv = np.expand_dims(features_blobs[i][batch], axis=0)
            CAMs = return_cam(features_conv, weight_softmax, [0])
            res_list.append(CAMs[0])
            
    cam_set = np.stack(res_list)
    CAM = merge_window(height, width, cam_set, window_size)
    
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_TURBO)
    cam_res = np.uint8(heatmap * 0.5 + img * 0.5)
    
    
    blue = (255,0,0)
    red = (0,0,255)
    font =  cv2.FONT_HERSHEY_PLAIN
    
    if res < 2:
        fire = 1
        result = cv2.putText(cam_res, classes[fire], (40,40), font, 2, blue, 2, cv2.LINE_AA)
    else:
        fire = 0
        result = cv2.putText(cam_res, classes[fire], (40,40), font, 2, red, 2, cv2.LINE_AA)
    
    return fire, result

def get_cam(net, features_blobs, img_pil, classes, root_img, IMG_SIZE):
    net.eval()
    
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    logit = net(img_variable)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    
    #output: the prediction
    for i in range(0, 2):
        line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])
        print(line)
    
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    height, width, _ = img_cv.shape
    
    CAMs = return_cam(features_blobs[0], weight_softmax, [0])
    print('output %s for the top1 prediction: %s' %(root_img, classes[idx[0].item()]))
    
    CAM = cv2.resize(CAMs[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_TURBO)
    result = heatmap * 0.5 + img_cv * 0.5
        
    return idx[0].item(), result
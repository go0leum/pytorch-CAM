from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2, torch
import os
import PIL

#sliding window
def sliding_window(img_pil, stepSize, windowSize):
    width, height= img_pil.size
    for y in range(0, height, stepSize):
        for x in range(0, width,stepSize):
            yield(x,img_pil.crop((x,y,x+windowSize,y+windowSize)))

# generate class activation mapping for the top1 prediction
def return_cam(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def get_cam(net, features_blobs, img_pil, classes, root_img, WINDOW_SIZE, STEP_SIZE):
    net.eval()
    
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # )
    # preprocess = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    #     normalize
    # ])

    # img_tensor = preprocess(img_pil)
    # img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    # logit = net(img_variable)
    # h_x = F.softmax(logit, dim=1).data.squeeze()
    # probs, idx = h_x.sort(0, True)
    
    # #output: the prediction
    # for i in range(0, 2):
    #     line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])
    #     print(line)
        
    # CAMs = return_cam(features_blobs[0], weight_softmax, [0])
    # print('output %s for the top1 prediction: %s' %(root_img, classes[idx[0].item()]))
    # img = cv2.imread(root_img)
    # height, width, _ = img.shape
    # CAM = cv2.resize(CAMs[0], (width, height))
    # heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    # result = heatmap * 0.3 + img * 0.5
    # cv2.imwrite('res_' + root_img, result)
    
    # return idx[0].item()
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    res=1
    
    for (x,window_pil) in sliding_window(img_pil, STEP_SIZE, WINDOW_SIZE):   
        
        if window_pil.width<WINDOW_SIZE or window_pil.height<WINDOW_SIZE:
            continue
        
        window_tensor=preprocess(window_pil)
        window_variable = Variable(window_tensor.unsqueeze(0)).cuda()
        logit = net(window_variable)
        
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        
        CAMs = return_cam(features_blobs[0], weight_softmax, [0])
        
        if idx[0].item()==0:
            res=0
            window_cv = cv2.cvtColor(np.array(window_pil), cv2.COLOR_RGB2BGR)
            CAM = cv2.resize(CAMs[0], (WINDOW_SIZE, WINDOW_SIZE))
            heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
            result = heatmap * 0.3 + window_cv * 0.5
            
            root=root_img.replace('png',str(x))
            
            cv2.imwrite('res_'+root+'.png',result)

    # render the CAM and output
    if res==0:
        print('output %s prediction: fire' %(root_img))
    else :
        print('output %s prediction: nonfire' %(root_img))
    
    return res

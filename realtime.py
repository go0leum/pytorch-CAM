import threading
from collections import deque
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

import cv2, torch
from torchvision import transforms
from itertools import product

import inception
from torch.nn import functional as F
import scipy.signal.windows as signal
from torch.autograd import Variable
import os, time
from datetime import datetime

import argparse
import sys

events = {
    'stop': threading.Event(),
    'record': threading.Event(),
    'detection': threading.Event()
}

q_img = deque(maxlen=1)
q_cam_in = deque(maxlen=1)
q_cam_out = deque(maxlen=1)
q_fire = deque(maxlen=1)

main_window = None
        
class MainWindow:
    def __init__(self, args):
        self.args = args
        
        self.video_input = VideoInput(args)
        self.video_input.daemon = True
        
        self.display = Display(args)
        self.display.daemon = True
        
        self.cam = FireCAM(args)
        self.cam.daemon = True
        
        self.tk = tk.Tk()
        self.tk.title('Fire Detection by LightVision Inc.')
        
        img = np.zeros((self.args.frame_height, self.args.frame_width, 3), np.uint8)
        img = Image.fromarray(img)
        self.webcam_tk = ImageTk.PhotoImage(image=img)
        self.webcam = tk.Label(self.tk, image=self.webcam_tk)
        self.webcam.pack(side='left', fill='both')
        
        self.rec_count = 0
        self.video_writer = None
        
        frame = tk.Frame(self.tk)
        frame.pack(side='top', fill='x', pady=5)
        
        self.btn_rec = tk.Button(frame, text = 'start record', overrelief="solid", command=self._video_record)
        self.btn_rec.pack(side='top', fill='x', pady=5)
        
        self.tk.after_idle(self._start_fire_detection)
        self.tk.protocol('WM_DELETE_WINDOW', self._stop_fire_detection)
        
    def update_webcam_image(self, img):
        img = Image.fromarray(img)
        self.webcam_tk = ImageTk.PhotoImage(image=img)

        self.webcam.config(image=self.webcam_tk)
        self.webcam.image = self.webcam_tk
    
    def record_webcam_image(self, img):
        if events['record'].is_set():
            self.video_writer.write(img)
    
    def _start_fire_detection(self):
        events['detection'].set()
        self.video_input.start()
        self.display.start()
        self.cam.start()

    def _stop_fire_detection(self):
        events['stop'].set()
        self.display.join(timeout=1)
        self.video_input.join(timeout=1)
        self.cam.join(timeout=1)
        self.tk.destroy()
    
    def _video_record(self):
        if self.btn_rec['text'] == 'start record':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            vid_name = datetime.now().strftime('%d-%m-%Y %H-%M-%S')
            rec_dir = self.args.record_path+'/record_'+vid_name+'.avi'
            self.video_writer = cv2.VideoWriter(rec_dir, fourcc, self.args.fps, (self.args.frame_width, self.args.frame_height))
            self.btn_rec['text'] = 'end record'
            events['record'].set()
        else :
            self.video_writer.release()
            self.rec_count+=1
            self.btn_rec['text'] = 'start record'
            events['record'].clear()

class VideoInput(threading.Thread):
    def __init__(self, args):
        threading.Thread.__init__(self)
        self.args = args
    
    def run(self):
        video_capture = cv2.VideoCapture(self.args.cam_index, cv2.CAP_DSHOW)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.frame_width)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.frame_height)
        video_capture.set(cv2.CAP_PROP_FPS, self.args.fps)
        
        while video_capture.isOpened():
            received, img = video_capture.read()
            # cv2.waitKey(17)
            if events['stop'].is_set() or not received:
                video_capture.release()
                break

            img = cv2.flip(img, 1)
            
            q_img.append(img)
            q_cam_in.append(img)

class Display(threading.Thread):
    def __init__(self,args):
        threading.Thread.__init__(self)
        self.args = args
        self.init = False
        self.org_img = None
        self.img = None
        self.fd_img = cv2.applyColorMap(np.zeros((self.args.frame_height, self.args.frame_width,3), dtype=np.uint8),cv2.COLORMAP_TURBO)
        self.fd_res = 1
        self.write = 1
    
    def run(self):
        while not events['stop'].is_set():
            start = datetime.now()
            try:
                self.org_img = q_img.pop()
            except:
                pass
            
            if len(q_cam_out)==0:
                self.write = 1
                self.fd_img = self.fd_img
                self.fd_res = self.fd_res
            else:
                self.write = 0
                try:
                    self.fd_img = q_cam_out.pop()
                except :
                    pass
                    
                try:
                    self.fd_res = q_fire.pop()
                except :
                    pass
              
            if self.org_img is not None:
                self.img = self._draw_cam(self.org_img, self.fd_img, self.fd_res, self.write)
                
                if events['record'].is_set():
                    main_window.record_webcam_image(self.img)
                    
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                main_window.update_webcam_image(self.img)
            
            end = datetime.now()
            elapsed_time = end - start
            max_sleep = 1 / self.args.fps
            if max_sleep > elapsed_time.total_seconds():
                time.sleep(max_sleep - elapsed_time.total_seconds())
    
    def _draw_cam(self, org_img, fd_img, fd_res, write):
        res_img = np.uint8(org_img * 0.6 + fd_img * 0.4)
        if fd_res == 0 :
            res_img = cv2.putText(res_img, 'fire', (40,60), cv2.FONT_HERSHEY_PLAIN, 3, (225,225,255), 12, cv2.LINE_AA)
            res_img = cv2.putText(res_img, 'fire', (40,60), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2, cv2.LINE_AA)
            if write == 0 :
                date = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                cv2.imwrite(self.args.alarm_path+'/'+date+'.png', org_img)
        
        return res_img

class FireCAM(threading.Thread):
    def __init__(self, args):
        threading.Thread.__init__(self)
        self.args = args
        self.features_blobs = []
        self.classes = {0: 'fire', 1: 'nonfire'}
        self.cached_2d_windows = dict()
        
    def run(self):
        net = self._getModel()
        
        final_conv = 'Mixed_7c'
        self.features_blobs = []
        net._modules.get(final_conv).register_forward_hook(self.hook_feature)
        
        while not events['stop'].is_set():
            try:
                start = datetime.now()
                if events['detection'].is_set():
                    
                    self.features_blobs = []
                    img = q_cam_in.pop()
                    img_set = self._batch_window(img)
                    
                    fire_count, fire_list, cam_set = self._get_cam_window(net, self.features_blobs, img_set)
                    
                    # if fire_count<2 : fire_alarm =1
                    # else : fire_alarm = 0
                    if fire_count == 0: fire_alarm = 1
                    else:
                        fire_alarm = self._second_window(net, img, fire_list)
                    
                    CAM = self._merge_window(cam_set, self.args.stepSize, self.args.windowSize)
                    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_TURBO)
                    
                    q_cam_out.append(heatmap)
                    q_fire.append(fire_alarm)
                        
                end = datetime.now()
                
                elapsed_time = (end-start)
                print(elapsed_time)
                max_sleep = 1 / self.args.camfps
                
                if max_sleep > elapsed_time.total_seconds():
                    time.sleep(max_sleep - elapsed_time.total_seconds())
                else:
                    time.sleep(1e-3)
                    
            except IndexError:
                time.sleep(1 / (2.5 * self.args.camfps))
            except Exception as e:
                print(e)
                
    def _getModel(self):
        resume = self.args.model_Epoch
        
        net = inception.inception_v3(num_classes=len(self.classes))
        net.cuda()
        
        assert os.path.isfile(self.args.pt_paht+'/'+ str(resume) + '.pt'), 'Error: no checkpoint found!'
        checkpoint = torch.load(self.args.pt_paht+'/' + str(resume) + '.pt')
        net.load_state_dict(checkpoint)
        
        return net
    
    def _batch_window(self, img):
        pad = transforms.Pad(padding=self.args.stepSize, padding_mode='reflect')
        totensor = transforms.ToTensor()
        
        img_cc = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_cc)
        img_pad = pad(img_pil)
        
        width, height = img_pad.size
        stepSize = self.args.stepSize
        windowSize = self.args.windowSize
        batch = self.args.batchSize
    
        n_y = (height//stepSize)+(height%stepSize!=0)
        n_x = (width//stepSize)+(width%stepSize!=0)
        
        total_iter = (n_y*n_x//batch)+(n_y*n_x%batch!=0)
        last_batch = n_y*n_x%batch
        
        y_list = []
        windows = []
        i = 1
        
        for y, x in product(range(0, n_y), range(0, n_x)):
            x_i = x*stepSize
            y_i = y*stepSize
            x_f = x_i+windowSize
            y_f = y_i+windowSize
            
            pil_crop = img_pad.crop((x_i,y_i,x_f,y_f))
            tensor_crop = totensor(pil_crop)
            y_list.append(tensor_crop)
            
            if len(y_list)==batch or (i==total_iter and len(y_list)==last_batch):
                y_tensor = torch.stack(y_list)
                windows.append(y_tensor)
                y_list = []
                i+=1
                
        return windows
    
    def hook_feature(self, module, input, output):
        self.features_blobs.append(output.data.cpu().numpy())
    
    def _return_cam(self, feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (224, 224)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam[cam<0] = 0
            cam[cam>18.5] = 18.5
            cam_img = cam / 18.5
            cam_img = np.uint8(255 * cam)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam
    
    def _second_window(self, net, img, fire_list):
        pad = transforms.Pad(padding=self.args.stepSize, padding_mode='reflect')
        
        transform = transforms.Compose([
            transforms.Resize((self.args.windowSize,self.args.windowSize)),
            transforms.ToTensor()
            ])
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        img_cc = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_cc)
        img_pad = pad(img_pil)
        
        w, h = img_pad.size
        step_Size = self.args.stepSize
        window_Size = self.args.windowSize
        batch = self.args.batchSize
        n_y = (h//step_Size)+(h%step_Size!=0)
        
        res = 0
        wins = []
        
        for i in range(0, len(fire_list)):
            fire_tensor = fire_list[i]
            for j in range(0, fire_tensor.shape[0]):
                n = i*batch + int(fire_tensor[j])
                i_y = n%n_y
                i_x = (n//n_y)+(n%n_y!=0)
                y = (i_y-0.5)*step_Size if (i_y-0.5)>=0 else 0
                x = (i_x-0.5)*step_Size if (i_x-0.5)>=0 else 0
                pil_crop = img_pad.crop((x, y, x+(1.5*window_Size), y+(1.5*window_Size)))
                tensor_crop = transform(pil_crop)
                wins.append(tensor_crop)
        
            wins_tensor=torch.stack(wins)
            
            win_tensor = normalize(wins_tensor)
            win_variable = Variable(win_tensor).cuda()
            
            logit = net(win_variable)
            h_x = F.softmax(logit, dim=1).data.squeeze()
            _, win_idx = h_x.sort(1, True)
            
            print(win_idx[:,0].size()[0] - int(torch.count_nonzero(win_idx[:,0])))
            
            if (win_idx[:,0].size()[0] - int(torch.count_nonzero(win_idx[:,0]))) != 0:
                res += 1
        
        if res !=0 : fire_alarm = 0
        else : fire_alarm = 1
        
        return fire_alarm
        
    
    def _get_cam_window(self, net, features_blobs, windows):
        net.eval()
        
        res_list = []
        fire_list = []
        fire_count = 0
        
        params = list(net.parameters())
        weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        for i in range(0, len(windows)):
            img_tensor = normalize(windows[i])
            img_variable = Variable(img_tensor).cuda()
            logit = net(img_variable)
            h_x = F.softmax(logit, dim=1).data.squeeze()
            _, idx = h_x.sort(1, True)
            
            fire = (idx[:,0].size()[0] - int(torch.count_nonzero(idx[:,0])))
            
            for batch in range(img_tensor.size()[0]):
                features_conv = np.expand_dims(features_blobs[i][batch], axis=0)
                CAMs = self._return_cam(features_conv, weight_softmax, [0])
                res_list.append(CAMs[0])
            
            if fire != 0:
                indexes = (idx[:,0]==0).nonzero(as_tuple = False)
                print(indexes)
                fire_count += fire
                fire_list.append(indexes)
        
        cam_set = np.stack(res_list)
        
        return fire_count, fire_list, cam_set

    def _window_2d(self, wnd_sz, power=2):
        '''
        Make a 1D window function, then infer and return a 2D window function.
        Done with an augmentation, and self multiplication with its transpose.
        Could be generalized to more dimensions.
        '''
        # Memoization
        self.cached_2d_windows = dict()
        key = "{}_{}".format(wnd_sz, power)
        if key in self.cached_2d_windows:
            wnd = self.cached_2d_windows[key]
        else:
            intersection = int(wnd_sz/4)
            wnd_outer = (abs(2*(signal.triang(wnd_sz))) ** power)/2
            wnd_outer[intersection:-intersection] = 0

            wnd_inner = 1 - (abs(2*(signal.triang(wnd_sz) - 1)) ** power)/2
            wnd_inner[:intersection] = 0
            wnd_inner[-intersection:] = 0

            wnd = wnd_inner + wnd_outer
            wnd = wnd / np.average(wnd)
            wnd = np.expand_dims(np.expand_dims(wnd, -1), -1)
            wnd = wnd * wnd.transpose(1, 0, 2)
            self.cached_2d_windows[key] = wnd
        return wnd

    def _merge_window(self, cam_set, step_Size, window_size):
        h = self.args.frame_height+step_Size*2
        w = self.args.frame_width+step_Size*2
        
        weights_each = self._window_2d(window_size, power=2).squeeze()
        CAM = np.zeros((h, w))
        divisor = np.zeros((h, w))
        i=0
        
        n_y = (h//step_Size)+(h%step_Size!=0)
        n_x = (w//step_Size)+(w%step_Size!=0)
        
        for y, x in product(range(0, n_y), range(0, n_x)):
            x_i = x*step_Size
            y_i = y*step_Size
            x_f = x_i+window_size
            y_f = y_i+window_size
            
            crop_h, crop_w = CAM[y_i:y_f, x_i:x_f].shape
            CAM[y_i:y_f, x_i:x_f] += (cam_set[i,0:crop_h,0:crop_w]*weights_each[0:crop_h,0:crop_w])
            divisor[y_i:y_f,x_i:x_f] += weights_each[0:crop_h,0:crop_w]
            i+=1
        
        CAM = np.uint8(CAM/divisor)
        CAM = CAM[step_Size:h-step_Size, step_Size:w-step_Size]
        return CAM

def main(args):
    os.environ['FIRE_DETECTION_HOME'] = os.getcwd()
    if not os.path.exits(args.alarm_path):
        os.mkdir(args.alarm_path)
    if not os.path.exits(args.pt_path):
        os.mkdir(args.pt_path)
    if not os.path.exits(args.record_path):
        os.mkdir(args.record_path)
    global main_window
    main_window = MainWindow(args)
    main_window.tk.mainloop()

def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--alarm_path', type=str, default='test_alarm')
    parser.add_argument('--pt_path', type=str, default='checkpoint')
    parser.add_argument('--record_path', type=str, default='record_video')
    parser.add_argument('--cam_index', type=int, default=0)
    parser.add_argument('--frame_width', type=int, default=1280)
    parser.add_argument('--frame_height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--camfps', type=int, default=3)
    parser.add_argument('--stepSize', type=int, default=112)
    parser.add_argument('--windowSize', type=int, default=224)
    parser.add_argument('--batchSize', type=int, default=126)
    parser.add_argument('--model_Epoch', type=int, default=200)
    parser.add_argument('--model_lr', type=float, default=0.001) #learning rate
    
    return parser.parse_args(argv)

if __name__ == '__main__':

    main(parse_arguments(sys.argv[1:]))
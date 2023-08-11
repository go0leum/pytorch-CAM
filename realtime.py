import threading
from collections import deque
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

import cv2, torch
from torchvision import transforms
from itertools import product

import inception
import CAM
import os, time
from datetime import datetime

import argparse
import sys

events = {
    'stop': threading.Event(),
    'preview': threading.Event(),
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
        
        img = np.zeros((args.frame_height, args.frame_width, 3), np.uint8)
        img = Image.fromarray(img)
        self.webcam_tk = ImageTk.PhotoImage(image=img)
        self.webcam = tk.Label(self.tk, image=self.webcam_tk)
        self.webcam.pack(side='left', fill='both')
        
        frame = tk.Frame(self.tk)
        frame.pack(side='top', fill='x', pady=5)
        
        self.tk.after_idle(self._start_fire_detection)
        self.tk.protocol('WM_DELETE_WINDOW', self._stop_fire_detection)
        
    def update_webcam_image(self, img):
        img = Image.fromarray(img)
        self.webcam_tk = ImageTk.PhotoImage(image=img)

        self.webcam.config(image=self.webcam_tk)
        self.webcam.image = self.webcam_tk
    
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

class VideoInput(threading.Thread):
    def __init__(self, args):
        threading.Thread.__init__(self)
        self.args = args
    
    def run(self):
        video_capture = cv2.VideoCapture(self.args.cam_index, cv2.CAP_DSHOW)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.frame_width)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.frame_height)
        video_capture.set(cv2.CAP_PROP_FPS, self.args.fps)
        
        while True:
            received, img = video_capture.read()
            if events['stop'].is_set() or not received:
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
        self.count = 0
        self.write = 1
    
    def run(self):
        while not events['stop'].is_set():
            start = datetime.now()
            try:
                self.org_img = q_img.pop()
            except:
                pass
            
            if len(q_cam_out)==0:
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
                self.img = self._draw_cam(self.org_img, self.fd_img, self.fd_res)
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                main_window.update_webcam_image(self.img)
            
            end = datetime.now()
            elapsed_time = end - start
            max_sleep = 1 / self.args.fps
            if max_sleep > elapsed_time.total_seconds():
                time.sleep(max_sleep - elapsed_time.total_seconds())
    
    def _draw_cam(self, org_img, fd_img, fd_res):
        res_img = np.uint8(org_img * 0.6 + fd_img * 0.4)
        
        if self.write==0 and fd_res==0:
            res_img = cv2.putText(res_img, 'fire', (40,40), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2, cv2.LINE_AA)
            cv2.imwrite('fire_alarm/%06.d.png'%self.count, res_img)
            self.count+=1
        
        return res_img
        
        

class FireCAM(threading.Thread):
    def __init__(self, args):
        threading.Thread.__init__(self)
        self.args = args
        self.features_blobs = []
        self.classes = {0: 'fire', 1: 'nonfire'}
        
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
                    
                    fire_alarm, result_img = CAM.get_cam_window(net, self.features_blobs, img, img_set, self.args.stepSize, self.args.windowSize)
                    q_cam_out.append(result_img)
                    q_fire.append(fire_alarm)
                        
                end = datetime.now()
                
                elapsed_time = (end-start)
                max_sleep = 1 / self.args.camfps
                
                if max_sleep > elapsed_time.total_seconds():
                    time.sleep(max_sleep - elapsed_time.total_seconds())
                else:
                    time.sleep(1e-3)
                    
            except IndexError:
                time.sleep(1 / (1.3 * self.args.camfps))
            except Exception as e:
                print(e)
                
    def _getModel(self):
        resume = self.args.model_Epoch
        
        net = inception.inception_v3(num_classes=len(self.classes))
        net.cuda()
        
        assert os.path.isfile('checkpoint/'+ str(resume) + '.pt'), 'Error: no checkpoint found!'
        checkpoint = torch.load('checkpoint/' + str(resume) + '.pt')
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

def main(args):
    os.environ['FIREDETECTION_HOME'] = os.getcwd()
    if not os.path.exists(args.db_path):
        os.mkdir(args.db_path)

    global main_window
    main_window = MainWindow(args)
    main_window.tk.mainloop()

def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cam_index', type=int, default=0)
    parser.add_argument('--frame_width', type=int, default=1280)
    parser.add_argument('--frame_height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--camfps', type=int, default=3)
    parser.add_argument('--stepSize', type=int, default=112)
    parser.add_argument('--windowSize', type=int, default=224)
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--model_Epoch', type=int, default=57)
    parser.add_argument('--model_lr', type=float, default=0.001) #learning rate
    parser.add_argument('--db_path', type=str, default='dataset')
    
    return parser.parse_args(argv)

if __name__ == '__main__':

    main(parse_arguments(sys.argv[1:]))
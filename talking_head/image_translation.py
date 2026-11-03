"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

from src.models.model_image_translation import ResUnetGenerator, VGGLoss
import torch
import torch.nn as nn
import time
import numpy as np
import cv2
import os, glob
from src.dataset.image_translation.image_translation_dataset import vis_landmark_on_img, vis_landmark_on_img98, vis_landmark_on_img74

from thirdparty.AdaptiveWingLoss.core import models
from thirdparty.AdaptiveWingLoss.utils.utils import get_preds_fromhm

# import face_alignment
# from tensorboardX import SummaryWriter

import subprocess as sp       ##!! LB: to replace obsolete calls to ffmpeg  !!##
from subprocess import DEVNULL 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Image_translation_block():

    def __init__(self, single_test=True):
        
        # Step 1 : define a number of parameters that are actually fixed

        self.add_audio_in = False
        self.load_G_name = 'examples/ckpt/ckpt_116_i2i_comb.pth'
        
        if(self.add_audio_in):
            self.G = ResUnetGenerator(input_nc=7, output_nc=3, num_downs=6, use_dropout=False)
        else:
            self.G = ResUnetGenerator(input_nc=6, output_nc=3, num_downs=6, use_dropout=False)
        # opt_parser.load_G_name is defined: examples/ckpt/ckpt_116_i2i_comb.pth

        if (self.load_G_name != ''):
            ckpt = torch.load(self.load_G_name, weights_only=False)
            try:
                self.G.load_state_dict(ckpt['G'])
            except:
                tmp = nn.DataParallel(self.G)
                tmp.load_state_dict(ckpt['G'])
                self.G.load_state_dict(tmp.module.state_dict())
                del tmp

        self.G.to(device)

    def single_test(self, jpg=None, fls=None):

        self.G.eval()
        collectedPicts = []
        previousFrame = np.zeros((68, 3), int)
        fl = previousFrame.astype(int)
        counts = 0
        for ii in range(0, fls.shape[0], 2):
            frame = fls[ii]
            
            shift = np.abs(((previousFrame[:, 0] - frame[:, 0]) \
                         + (previousFrame[:, 1] - frame[:, 1]))).sum()
            if shift > 75:
                img_fl = np.ones(shape=(256, 256, 3)) * 255
                fl = frame.astype(int)

                img_fl = vis_landmark_on_img(img_fl, np.reshape(fl, (68, 3)))
                frame = np.concatenate((img_fl, jpg), axis=2).astype(np.float32)/255.0

                image_in, image_out = frame.transpose((2, 0, 1)), np.zeros(shape=(3, 256, 256))
                image_in, image_out = torch.tensor(image_in, requires_grad=False), \
                                      torch.tensor(image_out, requires_grad=False)

                image_in, image_out = image_in.reshape(-1, 6, 256, 256), \
                                                  image_out.reshape(-1, 3, 256, 256)
                image_in, image_out = image_in.to(device), image_out.to(device)

                g_out = self.G(image_in)
                g_out = torch.tanh(g_out)

                g_out = g_out.cpu().detach().numpy().transpose((0, 2, 3, 1))
                g_out[g_out < 0] = 0
                ref_in = image_in[:, 3:6, :, :].cpu().detach().numpy().transpose((0, 2, 3, 1))
                fls_in = image_in[:, 0:3, :, :].cpu().detach().numpy().transpose((0, 2, 3, 1))

                previousFrame = fls[ii] 
                frame = g_out[0] * 255.0
                collectedPicts.append(frame.astype(np.uint8))
            else:
                collectedPicts.append(collectedPicts[-1])
                counts += 1

#
# Convert the sequence of pictutures together with the audio to an mp4 file
# in a single ffmpeg call, so the video is encoded only once.
#
        height, width, ch = collectedPicts[0].shape

        dimension = '{}x{}'.format(width, height)
        f_format = 'bgr24' # remember OpenCV uses bgr format
        fps = '31.25'       # the original frame rate was 62.5, corresponding to
                            # the hop-length 256 at fs=16 kHz.

        outputfile = 'tts.mp4'
        command = ['ffmpeg',
                '-y',
                '-loglevel', 'error',
                '-f', 'rawvideo',
    #            '-vcodec','rawvideo',
                '-s', dimension,
                '-pix_fmt', 'bgr24',
                '-r', fps,
                '-i', '-',
                '-i', 'examples/tts.wav',
                '-vcodec', 'mpeg4',
    #            '-b:v', '5000k',
                outputfile ]

        proc = sp.Popen(command, stdin=sp.PIPE, stderr=DEVNULL)

        for frame in collectedPicts:
            picture = np.ascontiguousarray(frame, dtype=np.uint8)
            proc.stdin.write(picture)

        proc.stdin.close()
        proc.wait()     # tts.mp4 must be complete before it is played


        '''
        writer.release()
##        np.save(picturedir + 'frame-' + str(pointer) + '.npy', np.asarray(collectedPicts))

        xxx = []
        xxx.append('ffmpeg')
        xxx.append('-loglevel')
        xxx.append('error')
        xxx.append('-y')
        xxx.append('-i')
        xxx.append('out1.mp4')
        xxx.append('-i')
        xxx.append(filename[:-16].replace('pred_fls_', '') + '.wav')
        xxx.append('-pix_fmt')
        xxx.append('yuv420p')
        xxx.append(filename[:-4].replace('pred_fls', prefix) + '.mp4')
        code = sp.run(xxx)
        '''

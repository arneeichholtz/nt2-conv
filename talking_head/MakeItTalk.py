import sys, os 
import numpy as np

import subprocess as sp

sys.path.append("thirdparty/AdaptiveWingLoss")
sys.path.append("thirdparty/resemblyer_util")
sys.path.append('../')

import glob
import cv2
import matplotlib 
import matplotlib.pyplot as plt
import face_alignment

from src.approaches.image_translation import Image_translation_block
import torch
import pickle
import shutil
import time, timeit
import util.utils as util
from scipy.signal import savgol_filter
from src.approaches.train_audio2landmark__ import Audio2landmark_model
from thirdparty.resemblyer_util.speaker_emb import get_spk_emb

## from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
## import src.autovc.AutoVC_mel_Convertor_retrain_version as convert

from src.autovc.retrain_version.model_vc_37_1 import Generator
from src.autovc.retrain_version.vocoder_spec.extract_f0_func import extract_f0_func_audiofile
from src.autovc.utils import quantize_f0_interp
from pydub import AudioSegment
from math import ceil

import yaml
###
# The chdir command is not necessary if the module is imported by the main script.
#
# os.chdir('c:/Users/LouBo/Arne')

import gtts
import yaml 
import wave
from pathlib import Path 
from piper import PiperVoice, SynthesisConfig
from scipy.signal import resample
from scipy.io.wavfile import read as readwav
from scipy.io.wavfile import write as writewav
#
# The chdir command is redundant if the module is imported by the main script.
#
# os.chdir('c:/Users/LouBo/Arne')

def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}

root = Path(__file__).resolve().parent
config = load_config(root.parent / "config.yml")
model_path = config.get("piper_model_path", config.get("tts_model_path"))
config_path = config.get("piper_config_path", config.get("tts_config_path"))
###
# Define a PIPER voice.
###
voice = PiperVoice.load(
            model_path,
            config_path,
            use_cuda=False,
        )

def make16kHz(wavfile):
    sf, data = readwav(wavfile)
    numsamplesIn = data.shape[0]
    numsamples16k = int(np.round(numsamplesIn*16000/sf))
    newdata = resample(data, numsamples16k)
    writewav('examples/tts.wav', 16000, newdata.astype(np.int16))
    print(numsamplesIn, numsamples16k)

def speak(prompt):
    with wave.open('tts.wav', 'wb') as wav_file:
        voice.synthesize_wav(
                            prompt,
                            wav_file,
                            set_wav_format=True,
                            syn_config=SynthesisConfig
                            )
    make16kHz('tts.wav')

st = time.time()

#
# determine the screen coordinates of the talking head#
def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    manager = getattr(f.canvas, "manager", None)
    window = getattr(manager, "window", None)
    if window is None:
        return

    if hasattr(window, "wm_geometry"):
        window.wm_geometry("+%d+%d" % (x, y))
        return

    if hasattr(window, "SetPosition"):
        window.SetPosition((x, y))
        return

    if hasattr(window, "move"):
        window.move(x, y)


def playVideo(videofile, baseImage):

    blank = np.ones((baseImage.shape[0], baseImage.shape[1], baseImage.shape[2]), dtype='uint8') * 255
    px = 0.5/plt.rcParams['figure.dpi']  # pixel in inches
    width, heigth = (baseImage.shape[0]*px, baseImage.shape[1]*px)
    with plt.rc_context({'toolbar': 'None'}):
        f = plt.figure(figsize=(width, heigth))
        move_figure(f, 250, 250)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(blank)
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.1)
    
    xxx = [
        'ffplay',
        '-hide_banner',
        '-noborder',
        '-autoexit',
        '-loglevel',
        'panic',
        '-x',
        '480',
        '-y',
        '540',
        '-left',
        '474',
        '-top',
        '518',
        videofile
        ]

    code = sp.run(xxx)
    plt.close(f)

    with plt.rc_context({'toolbar': 'None'}):
        f = plt.figure(figsize=(width, heigth))
        move_figure(f, 250, 250)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(baseImage)
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.1)

    return code

'''
Default hyper-parameters for the model.

Basic setup for animation.
'''
default_head_name = 'CatiaToon'           # the image name (with no .jpg) to animate
ADD_NAIVE_EYE = True                 # whether add naive eye blink
CLOSE_INPUT_FACE_MOUTH = False       # if your image has an opened mouth, put this as True, else False
AMP_LIP_SHAPE_X = 2.                 # amplify the lip motion in horizontal direction
AMP_LIP_SHAPE_Y = 2.                 # amplify the lip motion in vertical direction
AMP_HEAD_POSE_MOTION = 0.7           # amplify the head pose motion (usually smaller than 1.0, put it to 0. for a static head pose)
load_AUTOVC_name = 'examples/ckpt/ckpt_autovc.pth'


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

class AutoVC_mel_Convertor():
    

    def __init__(self, src_dir, proportion=(0., 1.), seed=0):

        self.src_dir = src_dir
        if(not os.path.exists(os.path.join(src_dir, 'filename_index.txt'))):
            self.filenames = []
        else:
            with open(os.path.join(src_dir, 'filename_index.txt'), 'r') as f:
                lines = f.readlines()
                self.filenames = [(int(line.split(' ')[0]), line.split(' ')[1][:-1]) for line in lines]

    def convert_single_wav_to_autovc_input(self, audio_filename, autovc_model_path):

        def pad_seq(x, base=32):
            len_out = int(base * ceil(float(x.shape[0]) / base))
            len_pad = len_out - x.shape[0]
            assert len_pad >= 0
            return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), len_pad
        t0 = int(time.time()) * 1000
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        G = Generator(16, 256, 512, 16).eval().to(device)

        g_checkpoint = torch.load(autovc_model_path, map_location=device)
        G.load_state_dict(g_checkpoint['model'])

        emb = np.loadtxt('src/autovc/retrain_version/obama_emb.txt')
        emb_trg = torch.from_numpy(emb[np.newaxis, :].astype('float32')).to(device)
        t1 = int(time.time()) * 1000
        aus = []
        audio_file = audio_filename

        sound = AudioSegment.from_file(audio_file, "wav")
        normalized_sound = match_target_amplitude(sound, -20.0)
        normalized_sound.export(audio_file, format='wav')

        x_real_src, f0_norm, melspectrum = extract_f0_func_audiofile(audio_file, 'F')

        f0_org_src = quantize_f0_interp(f0_norm)
        
        emb, _ = get_spk_emb(audio_file)


        l = x_real_src.shape[0]
        x_identic_psnt = []
        step = 4096
        for i in range(0, l, step):
            x_real = x_real_src[i:i + step]
            f0_org = f0_org_src[i:i + step]

            x_real, len_pad = pad_seq(x_real.astype('float32'))
            f0_org, _ = pad_seq(f0_org.astype('float32'))
            x_real = torch.from_numpy(x_real[np.newaxis, :].astype('float32')).to(device)
            emb_org = torch.from_numpy(emb[np.newaxis, :].astype('float32')).to(device)
            # emb_trg = torch.from_numpy(emb[np.newaxis, :].astype('float32')).to(device)
            f0_org = torch.from_numpy(f0_org[np.newaxis, :].astype('float32')).to(device)

            with torch.no_grad():
                x_identic, x_identic_psnt_i, code_real = G(x_real, emb_org, f0_org, emb_trg, f0_org)
                x_identic_psnt.append(x_identic_psnt_i)

        x_identic_psnt = torch.cat(x_identic_psnt, dim=1)

        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, :-len_pad, :].cpu().numpy()

        aus.append((uttr_trg, (0, audio_filename, emb)))

        return aus


def playVideo(videofile, baseImage):

    blank = np.ones((baseImage.shape[0], baseImage.shape[1], baseImage.shape[2]), dtype='uint8') * 128
    px = 2.0/plt.rcParams['figure.dpi']  # pixel in inches
    width, heigth = (baseImage.shape[0]*px, baseImage.shape[1]*px)
    with plt.rc_context({'toolbar': 'None'}):
        f = plt.figure(figsize=(width, heigth))
        move_figure(f, 200, 200)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(baseImage[:, :, ::-1])
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.1)
    
    xxx = [
        'ffplay',
        '-hide_banner',
        '-noborder',
        '-autoexit',
        '-loglevel',
        'panic',
        '-x',       # display width
        '590',
        '-y',
        '590',      # display height
        '-left',
        '400',
        '-top',
        '435',
        videofile
        ]

    code = sp.run(xxx)


    with plt.rc_context({'toolbar': 'None'}):
        f = plt.figure(figsize=(width, heigth))
        move_figure(f, 200, 200)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(baseImage[:, :, ::-1])
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.1)

    figs = plt.get_fignums()
    if len(figs) > 1:
        plt.close(figs[0])
    
    return code

def mp3TOwav(inputfile, outputfile):
    if os.path.exists(outputfile):
        os.remove(outputfile)
    if not os.path.exists(inputfile):
        print('inputfile not found')
    extractaudio = []
    extractaudio.append('ffmpeg')
    extractaudio.append('-hide_banner')
    extractaudio.append('-loglevel')
    extractaudio.append('error')
    extractaudio.append('-i')
    extractaudio.append(inputfile)
    extractaudio.append('-af')
    extractaudio.append('speechnorm=e=12.5:r=0.0001:l=1')
    extractaudio.append('-ar')
    extractaudio.append('16000')
    extractaudio.append(outputfile)

    code = sp.run(extractaudio)

        


'''
Step 2: load the image and detect its landmark

img is a 256 by 256 by 3 picture. 

face_alignment is a time-consuming procedure that returns a 63 by 3 matrix that 
depends on the picture. For each eligible picture it can be precomputed. 

'''

'''
shape_3d is still a 63 by 3 matrix. 

Step 3: Generate input data for inference based on (a set of) .mp3 file(s)
that are generated by gTTS. 
'''

class MakeItTalk: 
    
    def __init__(self,
                 headname,
                 ):
        self.headname = headname
        
        self.img = cv2.imread('examples/' + headname + '.jpg')
##
# Initialize image display
##
        baseImage = self.img
        blank = np.ones((baseImage.shape[0], baseImage.shape[1], baseImage.shape[2]), dtype='uint8') * 128
        px = 2.0/plt.rcParams['figure.dpi']  # pixel in inches
        width, heigth = (baseImage.shape[0]*px, baseImage.shape[1]*px)
        with plt.rc_context({'toolbar': 'None'}):
            f = plt.figure(figsize=(width, heigth))
            move_figure(f, 200, 200)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(baseImage[:, :, ::-1])
            plt.axis('off')
            plt.show(block=False)
            plt.pause(0.1)

        allexamples = os.listdir('examples/shapefiles')
        found = False
        search = headname.replace('Toon', '').replace('Flat', '')
        for name in allexamples:
            if search in name and not 'shift' in name:
                headfile = name
                found = True
                break 
        if found:
            print('headfile, name', headfile, name)
        else:
            print('not found')


        if found:
            shape_3d = np.loadtxt('examples/shapefiles/' + headfile)
            fin = open('examples/shapefiles/' + headfile.replace('.txt', '-scale_shift.txt'))
            regel = fin.readline().strip().split(',')
            scale = float(regel[0])
            shift = [float(regel[1]), float(regel[2])]
        else:

            predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, \
                                                     device='cpu', flip_input=True)
            shapes = predictor.get_landmarks(self.img)
            if (not shapes or len(shapes) != 1):
                print('Cannot detect face landmarks. Exit.')
                exit(-1)
            shape_3d = shapes[0]


            if CLOSE_INPUT_FACE_MOUTH:
                util.close_input_face_mouth(shape_3d)

            shape_3d[48:, 0] = (shape_3d[48:, 0] - np.mean(shape_3d[48:, 0])) * 1.05 + np.mean(shape_3d[48:, 0]) # wider lips
            shape_3d[49:54, 1] += 0.           # thinner upper lip
            shape_3d[55:60, 1] -= 1.           # thinner lower lip
            shape_3d[[37,38,43,44], 1] -=2.    # larger eyes
            shape_3d[[40,41,46,47], 1] +=2.    # larger eyes

            '''   
            Normalize face as input to audio branch
            '''

            shape_3d, scale, shift = util.norm_input_face(shape_3d)

            os.makedirs('examples/shapefiles/', exist_ok=True)
            np.savetxt('examples/shapefiles/' + opt_parser.jpg.replace('jpg', 'txt'), shape_3d)
            fout = open('examples/shapefiles/' + opt_parser.jpg.replace('.jpg', '-scale_shift.txt'), 'w')
            fout.write(str(scale) + ',' + str(shift[0]) + ',' + str(shift[1]))
            fout.close()
            
        self.shape_3d = shape_3d
        self.shift = shift 
        self.scale = scale

    def generate_prompt_gtts(self, prompt):
        tts = gtts.gTTS(prompt, lang='nl')
        tts.save('tts.mp3')
        mp3TOwav('tts.mp3', 'examples/tts.wav')

    def talking_head(self):    
        
        au_data = []
        au_emb = []


        ain = 'examples/tts.wav'
        wavname = 'tts.wav'

        au_data = []
        au_emb = []

        wavname = 'tts.wav'

        time_00 = int(time.time()) * 1000
        me, ae = get_spk_emb('examples/' + wavname)
        au_emb.append(me.reshape(-1))
        time_01 = int(time.time()) * 1000

        c = AutoVC_mel_Convertor('examples')

        au_data_i = c.convert_single_wav_to_autovc_input(audio_filename='examples/' + wavname,
               autovc_model_path=load_AUTOVC_name)
        au_data += au_data_i

        time_1 = int(time.time()) * 1000
        ###################################
        # landmark fake placeholder
        fl_data = []
        rot_tran, rot_quat, anchor_t_shape = [], [], []
        for au, info in au_data:
            au_length = au.shape[0]
            fl = np.zeros(shape=(au_length, 68 * 3))
            fl_data.append((fl, info))
            rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
            rot_quat.append(np.zeros(shape=(au_length, 4)))
            anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))

        if(os.path.exists(os.path.join('examples', 'dump', 'random_val_fl.pickle'))):
            os.remove(os.path.join('examples', 'dump', 'random_val_fl.pickle'))
        if(os.path.exists(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))):
            os.remove(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))
        if(os.path.exists(os.path.join('examples', 'dump', 'random_val_au.pickle'))):
            os.remove(os.path.join('examples', 'dump', 'random_val_au.pickle'))
        if (os.path.exists(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))):
            os.remove(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))

        with open(os.path.join('examples', 'dump', 'random_val_fl.pickle'), 'wb') as fp:
            pickle.dump(fl_data, fp)
        with open(os.path.join('examples', 'dump', 'random_val_au.pickle'), 'wb') as fp:
            pickle.dump(au_data, fp)
        with open(os.path.join('examples', 'dump', 'random_val_gaze.pickle'), 'wb') as fp:
            gaze = {'rot_trans':rot_tran, 'rot_quat':rot_quat, 'anchor_t_shape':anchor_t_shape}
            pickle.dump(gaze, fp)
        time_2 = int(time.time()) * 1000

        '''
        Step 4: Audio-to-Landmarks prediction

        the default case is that opt_parser.reuse_train_emb_list = []
        so we always execute model.test(au_emb=au_emb

        model.test() dumps the result in a .txt file in the examples directory (default)

        It also creates a line drawing cartoon that moves lip-synchronously with 
        the audio. This cartoon is (by default) stored in the examples directory
        with the file name of the audio file with .wav replaced by _av.mp4. 

        A substantial part of the processing in done by util/vis.py, where the 
        use of ffmpeg has been rewritten to avoid lots of uninformative screen output.
        '''

        model = Audio2landmark_model(jpg_shape=self.shape_3d)

        model.test(au_emb=au_emb)

        fls = glob.glob('examples/*.txt')
        fls.sort()


        for i in range(0,len(fls)):
        #for i in range(1):
            fl = np.loadtxt(fls[i]).reshape((-1, 68,3))
            fl[:, :, 0:2] = -fl[:, :, 0:2]
            fl[:, :, 0:2] = fl[:, :, 0:2] / self.scale - self.shift
            if (ADD_NAIVE_EYE) and fl.shape[0] > 47:
                fl = util.add_naive_eye(fl)

            # additional smooth
            fl = fl.reshape((-1, 204))
            fl[:, :48 * 3] = savgol_filter(fl[:, :48 * 3], 15, 3, axis=0)
            fl[:, 48*3:] = savgol_filter(fl[:, 48*3:], 5, 3, axis=0)
            fl = fl.reshape((-1, 68, 3))

            ''' STEP 6: Imag2image translation '''
            model = Image_translation_block(single_test=True)
            with torch.no_grad():
                model.single_test(jpg=self.img, fls=fl, filename=fls[i], \
                                  prefix=self.headname)

            time_4 = int(time.time()) * 1000
            videofile = fls[i].replace('pred_fls', self.headname).replace('txt', 'mp4')
            result = playVideo(videofile, self.img)

talkingHead = MakeItTalk(default_head_name)

def talkingHead_gtts(prompt):
    tts = gtts.gTTS(prompt, lang='nl')
    tts.save('tts.mp3')
    mp3TOwav('tts.mp3', 'examples/tts.wav')
    talkingHead.talking_head()


def talkingHead_piper(prompt):
    speak(prompt)
    talkingHead.talking_head()
    time4 = timeit.default_timer()

if __name__ == "__main__":
    
    prompts = ['Welkom. Ik ben vandaag jouw gastvrouw.', 'Wat kan ik voor je doen?', 
                'De soep van de dag is vandaag tomatensoep.']
    for prompt in prompts:
        talkingHead_gtts(prompt)

    sys.exit('success')

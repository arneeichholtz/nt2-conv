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

from image_translation import Image_translation_block
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

import gtts
import yaml 
import wave
from pathlib import Path 
from piper import PiperVoice, SynthesisConfig
from scipy.signal import resample
from scipy.io.wavfile import read as readwav
from scipy.io.wavfile import write as writewav

def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}

# config.yml and the voice paths in it live in the project root (one level up)
root = Path(__file__).resolve().parent.parent
config = load_config(root / "config.yml")
model_path = config.get("piper_model_path", config.get("tts_model_path"))
config_path = config.get("piper_config_path", config.get("tts_config_path"))

if not model_path or not config_path:
    raise FileNotFoundError(f"piper_model_path / piper_config_path missing in {root / 'config.yml'}")
model_path = Path(model_path)
config_path = Path(config_path)
if not model_path.is_absolute():
    model_path = root / model_path
if not config_path.is_absolute():
    config_path = root / config_path

###
# Define a PIPER voice.
###
voice = PiperVoice.load(
            model_path,
            config_path,
            use_cuda=False,
        )

def trim_silence(data, sf, threshold=0.02, keep_ms=100):
    """Remove leading/trailing silence (keeping keep_ms), so the head starts talking sooner and the video ends when the speech ends."""
    amplitude = np.abs(data.astype(np.float32))
    voiced = np.where(amplitude > threshold * amplitude.max())[0]
    if len(voiced) == 0:
        return data
    keep = int(sf * keep_ms / 1000)
    return data[max(0, voiced[0] - keep):min(len(data), voiced[-1] + keep)]

def make16kHz(wavfile):
    sf, data = readwav(wavfile)
    data = trim_silence(data, sf)
    # keep the trimmed original (22 kHz) as the audio of the video; the models need the 16 kHz copy
    writewav(wavfile, sf, data)
    numsamplesIn = data.shape[0]
    numsamples16k = int(np.round(numsamplesIn*16000/sf))
    newdata = resample(data, numsamples16k)
    writewav('examples/tts.wav', 16000, newdata.astype(np.int16))

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


'''
Display settings for the talking head window.

The video (ffplay) is placed exactly over the still image, so only these
values need to be changed to resize or move the head on the screen.
'''
DISPLAY_SCALE = 3.0                  # size of the head: 2.0 is the original size, 3.0 is 1.5x larger, etc.
DISPLAY_X, DISPLAY_Y = 200, 200      # screen position of the upper left corner of the window
VIDEO_OFFSET_X, VIDEO_OFFSET_Y = 0, 0  # fine-tune the video position (pixels) if it does not overlap the image exactly

'''
Default hyper-parameters for the model.

Basic setup for animation.
'''
default_head_name = 'male_cartoon'          # the image name (with no .jpg) to animate
ADD_NAIVE_EYE = True                 # whether add naive eye blink
CLOSE_INPUT_FACE_MOUTH = False       # if your image has an opened mouth, put this as True, else False
AMP_LIP_SHAPE_X = 2.                 # amplify the lip motion in horizontal direction
AMP_LIP_SHAPE_Y = 2.                 # amplify the lip motion in vertical direction
AMP_HEAD_POSE_MOTION = 0.7           # amplify the head pose motion (usually smaller than 1.0, put it to 0. for a static head pose)
load_AUTOVC_name = 'examples/ckpt/ckpt_autovc.pth'


def image_screen_rect(f, im):
    """Return the screen rectangle (left, top, width, height) in pixels of image im in figure f."""
    if not hasattr(f.canvas, 'get_tk_widget'):
        # not a Tk window: fall back to the values that were tuned for DISPLAY_SCALE = 2.0
        size = int(round(590 * DISPLAY_SCALE / 2.0))
        return 400, 435, size, size
    widget = f.canvas.get_tk_widget()
    widget.update_idletasks()
    bbox = im.get_window_extent()    # pixels inside the canvas, origin at the bottom left
    # round the edges (not the size) so the video covers every partially drawn pixel row/column
    left = widget.winfo_rootx() + int(np.floor(bbox.x0))
    right = widget.winfo_rootx() + int(np.ceil(bbox.x1))
    top = widget.winfo_rooty() + int(np.floor(f.bbox.height - bbox.y1))
    bottom = widget.winfo_rooty() + int(np.ceil(f.bbox.height - bbox.y0))
    return left, top, right - left, bottom - top


def show_image(baseImage):
    """Show the still image and return the figure and the screen rectangle of the image."""
    px = DISPLAY_SCALE/plt.rcParams['figure.dpi']  # pixel in inches
    width, heigth = (baseImage.shape[1]*px, baseImage.shape[0]*px)
    with plt.rc_context({'toolbar': 'None'}):
        f = plt.figure(figsize=(width, heigth))
        move_figure(f, DISPLAY_X, DISPLAY_Y)
        plt.xticks([])
        plt.yticks([])
        im = plt.imshow(baseImage[:, :, ::-1])
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.1)
    return f, im


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

class AutoVC_mel_Convertor():
    

    def __init__(self, src_dir, autovc_model_path, proportion=(0., 1.), seed=0):

        self.src_dir = src_dir
        if(not os.path.exists(os.path.join(src_dir, 'filename_index.txt'))):
            self.filenames = []
        else:
            with open(os.path.join(src_dir, 'filename_index.txt'), 'r') as f:
                lines = f.readlines()
                self.filenames = [(int(line.split(' ')[0]), line.split(' ')[1][:-1]) for line in lines]

        # load the AutoVC model once, it is reused for every sentence
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G = Generator(16, 256, 512, 16).eval().to(self.device)

        g_checkpoint = torch.load(autovc_model_path, map_location=self.device)
        self.G.load_state_dict(g_checkpoint['model'])

        embed = np.loadtxt('src/autovc/retrain_version/obama_emb.txt')
        self.emb_trg = torch.from_numpy(embed[np.newaxis, :].astype('float32')).to(self.device)

    def convert_single_wav_to_autovc_input(self, audio_filename):

        def pad_seq(x, base=32):
            len_out = int(base * ceil(float(x.shape[0]) / base))
            len_pad = len_out - x.shape[0]
            assert len_pad >= 0
            return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), len_pad
        device = self.device
        G = self.G
        emb_trg = self.emb_trg

        aus = []
        audio_file = audio_filename

        sound = AudioSegment.from_file(audio_file, "wav")
        normalized_sound = match_target_amplitude(sound, -20.0)
        normalized_sound.export(audio_file, format='wav')

        x_real_src, f0_norm, melspectrum = extract_f0_func_audiofile(audio_file, 'F')

        f0_org_src = quantize_f0_interp(f0_norm)
        time_0 = timeit.default_timer()
        emb, _ = get_spk_emb(audio_file)
        time_1 = timeit.default_timer()
        # print('time in get_spk_emb in converrt_single_wav', time_1 - time_0)


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


def playVideo(videofile, f, im):
    """Play the video with ffplay exactly over the still image im that is already shown in figure f."""
    left, top, width, height = image_screen_rect(f, im)

    xxx = [
        'ffplay',
        '-hide_banner',
        '-noborder',
        '-autoexit',
        '-loglevel',
        'panic',
        '-x',       # display width
        str(width),
        '-y',
        str(height),      # display height
        '-left',
        str(left + VIDEO_OFFSET_X),
        '-top',
        str(top + VIDEO_OFFSET_Y),
        videofile
        ]

    code = sp.run(xxx)

    # the still image is still on screen underneath the video, so only redraw it
    # (building a new figure here delayed the start of listening)
    f.canvas.draw_idle()
    f.canvas.flush_events()

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
        self.img = plt.imread('examples/' + headname + '.jpg')[:, :, ::-1]
        self.fig, self.im = show_image(self.img)

        allexamples = os.listdir('examples/shapefiles')
        found = False
        search = headname.replace('Toon', '').replace('Flat', '')
        for name in allexamples:
            if search in name and not 'shift' in name:
                headfile = name
                found = True
                break 


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
            np.savetxt('examples/shapefiles/' + headname + '.txt', shape_3d)
            fout = open('examples/shapefiles/' + headname + '-scale_shift.txt', 'w')
            fout.write(str(scale) + ',' + str(shift[0]) + ',' + str(shift[1]))
            fout.close()
            
        self.shape_3d = shape_3d
        self.shift = shift
        self.scale = scale

        # load all models once; talking_head() reuses them for every sentence
        self.autovc = AutoVC_mel_Convertor('examples', autovc_model_path=load_AUTOVC_name)
        self.audio2landmark = Audio2landmark_model(jpg_shape=self.shape_3d)
        self.image_translation = Image_translation_block(single_test=True)


    def talking_head(self, video_audio='examples/tts.wav'):
        """video_audio is the audio put in the video; the models always use the 16 kHz examples/tts.wav."""

        ain = 'examples/tts.wav'
        wavname = 'tts.wav'
        
        time_00 = timeit.default_timer()
        au_data = []
        au_emb = []

        me, ae = get_spk_emb('examples/' + wavname)
        au_emb.append(me.reshape(-1))

        au_data_i = self.autovc.convert_single_wav_to_autovc_input(audio_filename='examples/' + wavname)
        au_data += au_data_i

        time_01 = timeit.default_timer()
        # print('time in AutoVC_mel', time_01 - time_00)
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
        time_02 = timeit.default_timer()
        # print('time in step 3', time_02 - time_01)

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
        time_03 = timeit.default_timer()
        self.audio2landmark.test(au_emb=au_emb)
        time_04 = timeit.default_timer()
        # print('time in Audio2Landmark', time_04 - time_03)
        
        fls = 'examples/pred_fls_tts_audio_embed.txt'

        fl = np.loadtxt(fls).reshape((-1, 68,3))
        fl[:, :, 0:2] = -fl[:, :, 0:2]
        fl[:, :, 0:2] = fl[:, :, 0:2] / self.scale - self.shift
        if (ADD_NAIVE_EYE) and fl.shape[0] > 47:
            fl = util.add_naive_eye(fl)

        # # additional smooth
        # np.save('fl_before_filter.npy', fl)
        # fl = fl.reshape((-1, 204))
        # np.save('fl_reshaped_before.npy', fl)
        # fl[:, :48*3] = savgol_filter(fl[:, :48*3], 15, 3, axis=0)
        # fl[:, 48*3:] = savgol_filter(fl[:, 48*3:], 5, 3, axis=0)
        # np.save('fl_reshaped_after.npy', fl)
        # fl = fl.reshape((-1, 68, 3))
        # np.save('fl_after_filter.npy', fl)

        ''' STEP 6: Imag2image translation '''
        with torch.no_grad():
            self.image_translation.single_test(jpg=self.img, fls=fl, audio_file=video_audio)

        time_05 = timeit.default_timer()
        # print('time in step 6', time_05 - time_04)
        return_code = playVideo('tts.mp4', self.fig, self.im)

talkingHead = MakeItTalk(default_head_name)

def talkingHead_gtts(prompt):
    tts = gtts.gTTS(prompt, lang='nl')
    tts.save('tts.mp3')
    mp3TOwav('tts.mp3', 'examples/tts.wav')
    talkingHead.talking_head()


def talkingHead_piper(prompt):
    speak(prompt)
    talkingHead.talking_head(video_audio='tts.wav')    # original 22 kHz Piper audio

if __name__ == "__main__":
    
    prompts = ['Welkom. Ik ben vandaag jouw gastheer.', 'Wat kan ik voor je doen?', 
                'De soep van de dag is vandaag tomatensoep.']
    for prompt in prompts:
        talkingHead_piper(prompt)

    sys.exit('success')

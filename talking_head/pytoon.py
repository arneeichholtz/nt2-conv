import torch
import torchaudio
import matplotlib 
import matplotlib.pyplot as plt
from PIL import Image
import cv2 

import torchaudio.functional as F

import sys, os
import numpy as np 
import random 
from copy import deepcopy 
import yaml 
import wave
from pathlib import Path 
from piper import PiperVoice, SynthesisConfig
import gtts 
import json
from dataclasses import dataclass
import subprocess as sp 
from subprocess import DEVNULL, call
import timeit


def checkGPUmem():
    stdout_orig = sys.stdout.fileno()
    outf = open('gpuList.txt', 'w')
    call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"],\
        stdout=outf)
    outf.close()
    stdout = stdout_orig
    print('Active CUDA device: GPU', torch.cuda.current_device())

    fin = open('gpuList.txt', 'r')
    gpuList = []
    usedMem = []
    freeMem = []
    for line in fin:
        gpuList.append(line.strip())
    fin.close()
    for line in gpuList:
        print(line)
        try:
            totalMem = float(line.split(',')[3].split()[0])
            print('total memory:', np.round(totalMem/1000))
        except:
            continue
    for ii in range(1, len(gpuList)):
        usedMem.append(int(gpuList[ii].split(',')[4].split()[0]))
        freeMem.append(int(gpuList[ii].split(',')[5].split()[0]))
    print(usedMem, freeMem)
    os.remove('gpuList.txt')
'''
print('before model load')
checkGPUmem()
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bundle = torchaudio.pipelines.MMS_FA
'''
This pipeline differs from what is used in pytoon in a number of parameters.
'''
model = bundle.get_model(with_star=False).to(device)
sample_rate = bundle.sample_rate
LABELS = bundle.get_labels(star=None)

DICTIONARY = bundle.get_dict(star=None)
'''
print('after model load')
checkGPUmem()
'''
###
# Most probaby, the os.chdir is redundant, because it has already been performed by the
# script that imports this module.
###
#
# os.chdir('c:/Users/LouBo/Arne')

## Get the tabels needed to simplify the transcription 

fin = open('assets/phone2label.txt', 'r', encoding = 'utf-8')
lab_tokens = [[], []]
for regel in fin:
    parts = regel.strip().split('/ ')
    lab_tokens[0].append(parts[0])
    lab_tokens[1].append(parts[1])
fin.close()
fin = open('assets/toremove.txt', 'r', encoding='utf-8')
toremove = fin.readline().strip().split()
fin.close()
toremove = list(np.unique(toremove))
#
## get the tabel that maps transcription symbols to visemes.
#
toPytoon = [[], []]
fin = open('assets/letter2Pytoon.txt', 'r')
for regel in fin:
    parts = regel.strip().split()
    toPytoon[0].append(parts[0])
    toPytoon[1].append(parts[1])
fin.close()

#
# determine the screen coordinates of the talking head#
def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    manager = getattr(f.canvas, "manager", None)
    window = getattr(manager, "window", None)
    if window is None:
        return

    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        if hasattr(window, 'wm_geometry'):
            window.wm_geometry("+%d+%d" % (x, y))
        elif hasattr(window, 'geometry'):
            window.geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        if hasattr(window, 'SetPosition'):
            window.SetPosition((x, y))
    elif hasattr(window, 'move'):
        # This works for QT and GTK when the method exists.
        window.move(x, y)

##
# add audio to a video clip.
##
def add_audio(imagefile, speechfile, outputfile):
    if os.path.exists(outputfile):
        os.remove(outputfile)
    xxx = [
        'ffmpeg',
        '-loglevel',
        'error',
        '-i',
        imagefile,
        '-i',
        speechfile,
        outputfile ]
#    print(xxx)
    code = sp.run(xxx)
    return code


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
    
    baseImage = np.load('assets/happypose.npy')
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

def picts2video(pict_list, framerate):
    height, width, ch = pict_list[0].shape 

    dimension = '{}x{}'.format(width, height)
    f_format = 'bgr24' # remember OpenCV uses bgr format
    fps = str(framerate)

    command = ['ffmpeg',
            '-y',
            '-f', 'rawvideo',
#            '-vcodec','rawvideo',
            '-s', dimension,
            '-pix_fmt', 'bgr24',
            '-r', fps,
            '-i', '-',
            '-an',
            '-vcodec', 'mpeg4',
#            '-b:v', '5000k',
            'temp.mp4' ]

    proc = sp.Popen(command, stdin=sp.PIPE, stderr=DEVNULL)

    for frame in pict_list:
        picture = rgba2rgb(bgra_to_rgba(frame)).astype(np.uint8)
        proc.stdin.write(picture)
        
    proc.stdin.close()
#    proc.stderr.close()
#    proc.wait()


'''
We use Piper stuf for implementing a (too) detailed G2P.
'''
def load_config(path: Path) -> dict:
	if not path.exists():
		return {}
	data = yaml.safe_load(path.read_text(encoding="utf-8"))
	return data or {}

root = Path(__file__).resolve().parent
config = load_config(root.parent / "config.yml")
model_path = config.get("piper_model_path", config.get("tts_model_path"))
config_path = config.get("piper_config_path", config.get("tts_config_path"))


def read_json(file: str) -> dict:
    """Reads a json file to dictionary

    Args:
        path (str): Path to the json file.

    Returns:
        dict: Python dictionary with json data.
    """
    path = f"{os.path.dirname(__file__)}/assets/{file}"
    with open(path, "r") as file:
        data = json.load(file)
    return data

def bgra_to_rgba(image):
    # Swap blue and red channels
    b, g, r, a = np.rollaxis(image, axis=-1)
    return np.dstack([r, g, b, a])

##
# remove fourth dimension from picture frame
##
def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'
    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]
    a = np.asarray( a, dtype='float32' ) / 255.0
    R, G, B = background
    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B
    return np.asarray( rgb, dtype='uint8' )

@dataclass
class MouthCoordinates:
    """Data class for mouth image coordinate and transformation data"""

    x: float  # Distance (pxls) from top border of mouth img to top border of pose img.
    y: float  # Distance (pxls) from left border of mouth img to left border of pose img.
    scale_x: float  # Multiple by which the image should be scaled along x-axis (width)
    scale_y: float  # Multiple by which the image should be scaled along y-axis (height)
    flip_x: bool  # Image should be flipped horizontally (about the center y-axis)
    rotation: float  # Counter clockwise degrees that the image should be rotated

@dataclass
class Pose:
    """Data class for storing data for specific character poses."""

    image_files: dict  # Dictionary containing paths to variations of the pose image.
    mouth_coordinates: MouthCoordinates  # Mouth coordinates / transformations.


@dataclass
class Emotions:
    """Data class for storing emotion specific poses"""

    explain: list[Pose]  # List of poses for the explain emotion.
    happy: list[Pose]  # List os poses for the happy emotion.
    rhetorical: list[Pose]  # List of poses for the sad emotion.

def get_assets() -> Emotions:
    """Loads pose data from json file and returns as a dictionary.

    Returns:
        dict: Pose data, including paths to images, emotion specific poses, and mouth coords.
    """
    pose_data = read_json(file="frontpose_data.json")["emotions"]

    emotions = {}
    for emotion in pose_data.keys():
        if emotion not in ["sad", "angry", "confused"]:
            poses = []
            for i, _ in enumerate(pose_data[emotion]):
                images = deepcopy(pose_data[emotion][i]["image_files"])
                coords = deepcopy(pose_data[emotion][i]["mouth_coordinates"])
                pose = {
                    "image_files": images,
                    "mouth_coordinates": MouthCoordinates(**coords),
                }
                poses.append(Pose(**pose))
            emotions[emotion] = poses
    return Emotions(**emotions)
###
# Get a bunch of additional data
###

assets = get_assets()
VISEMES = read_json("visemes.json")

def random_emotion(assets):
    """Generates a random emotion to use in sequence

    Returns:
        list[Pose]: List of poses from a random emotion
    """
    emotions_list = list(assets.__dict__.keys())
    emotion = random.choice(emotions_list)
    return getattr(assets, emotion)

class phonemize():
    
    def __init__(self,
        model_path: str | Path | None,
        config_path: str | Path | None,
        ) -> None:
        if model_path == None:
            self.model_path = 'voices/nl_NL-pim-medium.onnx'
            self.config_path = 'voices/nl_NL-pim-medium.onnx.json'
        else:
            self.model_path = Path(model_path)
            self.config_path = Path(config_path)

        use_cuda = False
        self.voice = PiperVoice.load(
            self.model_path,
            self.config_path,
            use_cuda=use_cuda,
        )

    def transform(self, text):
        phones = self.voice.phonemize(text)
        phones = str(phones).replace('[[',"").replace(']]', '').replace("', '", "").replace("'], ['", " ")
        return phones
    
        
    def speak(self, prompt):
        with wave.open('tts.wav', 'wb') as wav_file:
            self.voice.synthesize_wav(
                                    prompt,
                                    wav_file,
                                    set_wav_format=True
                                  )

###
# Define a phonemizer.
###
g2p = phonemize(
    model_path = model_path,
    config_path = config_path
    )

#####
# Define a set of functions used by pytoon.
#####

def blink_manager(idx, fps):

    blink_rate = 3.0
    BLINK_DURATION = 0.16
    SUB_BLINKS = ["middle", "shut", "middle"]

    frames_between_blinks = int(blink_rate * fps)
    frames_per_blink = int(BLINK_DURATION * fps)
    frames_per_sub_blink = int(frames_per_blink / len(SUB_BLINKS)) + 1

    full_cycle = frames_between_blinks + (frames_per_sub_blink * len(SUB_BLINKS))

    start_1 = frames_between_blinks
    start_2 = start_1 + frames_per_sub_blink
    start_3 = start_2 + frames_per_sub_blink
    end_3 = start_3 + frames_per_sub_blink

    if start_1 <= (idx % full_cycle) < start_2:
        eyes = "middle"

    elif start_2 <= (idx % full_cycle) < start_3:
        eyes = "shut"

    elif start_3 <= (idx % full_cycle) < end_3:
        eyes = "middle"

    else:
        eyes = "open"

    return eyes

def mouth_transformation(mouth_file, mouth_coord) -> Image:
    """Transforms mouth image with scaling, flipping, and rotation.
        This transformation is applied because, the same mouth shape images
        are used for different pose images, but the size, angle, and position
        of a mouth image will depend on which pose image is being used.

    Args:
        mouth_path (str): .png file path pointing to mouth image
        transformation (np.array): image transformation data for mouth

    Returns:
        Image: PIL Image object of mouth image with applied transformations
    """
    mouth = deepcopy(Image.open(mouth_file))
    # Flip mouth horizontally if necessary

    if mouth_coord.flip_x is True:
        mouth = mouth.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    # Scale mouth image if necessary
    if mouth_coord.scale_y != 1:
        og_width, og_height = mouth.size
        new_width = int(abs(og_width * mouth_coord.scale_x))
        new_height = int(og_height * mouth_coord.scale_y)
        try:
            mouth = mouth.resize(new_width, new_height, Image.Resampling.LANCZOS)
        except:
            pass
    # Apply image rotation if necessary
    if mouth_coord.rotation != 0:
        mouth = mouth.rotate(-mouth_coord.rotation, resample=Image.Resampling.BICUBIC)
    return mouth

def render_frame(pose_img: Image, mouth_img: Image, mouth_coord):
    pose_img = bgra_to_rgba(pose_img)  # convert to rgba
    pose_img = Image.fromarray(pose_img)
    mouth_width, mouth_height = mouth_img.size

    # Location in pose image where mouth / viseme image will be added
    paste_coordinates = (
        int(mouth_coord.x - (mouth_width / 2)),
        int(mouth_coord.y - (mouth_height / 2)),
    )

    # Paste the mouth image onto the face image at the specified coordinates
    pose_img.paste(im=mouth_img, box=paste_coordinates, mask=mouth_img)
    np_image = np.array(pose_img)
    return np_image

#####################################################
#####################################################

def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores

'''
# Do forced alignment of speech and verbatim transcription.
# For that purpose, the verbatim transcription is transformed into a simplified 
# phonemic transcription.
# The function returns a list of mouth shapes and a sequence of frame numbers
# at which the mouth shapes change (only for debugging).
'''

def forced_align(speechfile, transliteration, VISEMES):
    try:
        waveform, sf = torchaudio.load(speechfile, backend="soundfile")
    except Exception:
        # Fallback for environments where torchaudio routes through torchcodec.
        import librosa

        audio, sf = librosa.load(speechfile, sr=None, mono=True)
        waveform = torch.from_numpy(audio).unsqueeze(0)
    if sf != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sf, sample_rate)

    with torch.inference_mode():
        emission, _ = model(waveform.to(device))

#    Convert the transliteration to a simplified phone sequence, using
#    only the 28 labels available in the bundle.

    transcriptie = g2p.transform(transliteration)

    transcrtrans = ''
    for kar in transcriptie:
        if kar in toremove:
            continue
        elif kar == ' ':
            transcrtrans = transcrtrans + ' '
        else:
            index = lab_tokens[0].index(kar)
            transcrtrans = transcrtrans + lab_tokens[1][index]
    TRANSCRIPT = transcrtrans.split()
     
    tokenized_transcript = [DICTIONARY[c] for word in TRANSCRIPT for c in word]

    aligned_tokens, alignment_scores = align(emission, tokenized_transcript)
    total_length = len(aligned_tokens)
    token_spans = F.merge_tokens(aligned_tokens, alignment_scores)

#    print("Token\tTime\tScore")
    alignment = [[], [], []]
    for s in token_spans:
        viseme = toPytoon[1][toPytoon[0].index(LABELS[s.token])]
#        print(f"{LABELS[s.token]}\t{viseme}\t[{s.start:3d}, {s.end:3d})\t{s.score:.2f}")
        alignment[0].append(viseme)
        alignment[1].append(s.start)
        alignment[2].append(s.end)
#
## alignment2visemes
#
    visemes = []
    startframes = [[], []]
    speechStart = alignment[1][0]
    for ii in range(speechStart):
        visemes.append('9.png')
    for label in range(len(alignment[0]) - 1):
        one, two = alignment[1][label], alignment[2][label]
        nextlbl = alignment[1][label+1]
        visms = VISEMES[alignment[0][label]]
        span = nextlbl - one
        startframes[0].append(one)
        startframes[1].append(len(visemes))

        toAdd = min(14, span)
        for kk in range(toAdd // 2):
            visemes.append(visms[0])
        for kk in range(toAdd//2, toAdd):
            visemes.append(visms[1])
        if span > 14:
            for kk in range(span - 14):
                visemes.append('9.png')

    visms = VISEMES[alignment[0][-1]]
    span = total_length - alignment[1][-1]
    if span > 14:
        for kk in range(7):
            visemes.append(visms[0])
        for kk in range(7):
            visemes.append(visms[1])
        for kk in range(span - 14):
            visemes.append('9.png')
    else:
        for kk in range(span // 2):
            visemes.append(visms[0])
        for kk in range(span//2, span):
            visemes.append(visms[1])
            
    return visemes, startframes

#####
# Create the sequence of pose files that form the basis for the animation.
##### 

def build_pose_sequence(sequence, assets, fps):
    """Creates the sequence of pose images for the video"""
    emotion = random_emotion(assets)
    pose = random.choice(emotion)

    pose_files = []
    mouth_coordinates = []
    mouth_images = []

    # Add a character pose frame for every frame of a mouth
    for i, _ in enumerate(sequence):

        eyes = blink_manager(i, fps)
        pose_files.append(pose.image_files[eyes])
        mouth_coordinates.append(pose.mouth_coordinates)

    
    # Create mouth PIL image for every frame, with image transformations based on pose
    transformed_image = mouth_transformation('assets/visemes/positive/' + sequence[0], mouth_coordinates[0])
    mouth_images.append(transformed_image)
    for ii in range(1, len(sequence)):
        if sequence[ii] == sequence[ii-1]:
            mouth_images.append(transformed_image)
        else:
            transformed_image = mouth_transformation('assets/visemes/positive/' + sequence[ii], mouth_coordinates[ii])
            mouth_images.append(transformed_image)

    return pose_files, mouth_images, mouth_coordinates


#######################

def compile_animation(pose_files, image_files, mouth_coords, audio_file):
    final_frames = []
    previous_pose = None
    previous_mouth = None
    for i, _ in enumerate(pose_files):
        if not pose_files[i] == previous_pose:
            previous = pose_files[i]
            newframe = cv2.imread(pose_files[i][1:], cv2.IMREAD_UNCHANGED)
#        frame = newframe.copy()
            previous_mouth = image_files[i]
            final_frame = render_frame(
                pose_img=newframe,
                mouth_img=image_files[i],
                mouth_coord=mouth_coords[i],
            )
            final_frames.append(final_frame)
        elif pose_files[i] == previous_pose and image_files[i] == previous_mouth:
            final_frames.append(final_frame)
        else:
            previous_mouth = image_files[i]
            final_frame = render_frame(
                pose_img=newframe,
                mouth_img=image_files[i],
                mouth_coord=mouth_coords[i],
            )           

    #
    # dump the animation in form of a video
    #
##    np.save('numpyPictures.npy', np.asarray(final_frames))
    fps = 48
    picts2video(final_frames, fps)

    withspeech = audio_file.split('/')[-1].split('.')[0] + '.mp4'
    add_audio('temp.mp4', audio_file, withspeech)
    
    code = playVideo(withspeech, newframe)
    np.save('picture.npy', newframe)
    return len(final_frames) / fps


def talkingHead_gtts(prompt):
    tts = gtts.gTTS(prompt, lang='nl')
    tts.save('tts.mp3')
    result, start_frames = forced_align('tts.mp3', prompt, VISEMES)
    poses, images, coordinates = build_pose_sequence(result, assets, 48)
    duration = compile_animation(poses, images, coordinates, 'tts.mp3')
    

def talkingHead_piper(prompt):
    g2p.speak(prompt)
    result, start_frames = forced_align('tts.wav', prompt, VISEMES)
    poses, images, coordinates = build_pose_sequence(result, assets, 48)
    duration = compile_animation(poses, images, coordinates, 'tts.wav')
    


if __name__ == "__main__":

    prompts = ["goedemiddag. hoe gaat het met jou vandaag", 
               "waar woon je in Nederland?", "had je ook werk in Finland?"]
    for prompt in prompts[:]:
        talkingHead_piper(prompt)

    sys.exit()

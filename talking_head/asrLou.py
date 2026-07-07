import sys, os
from subprocess import call 
import collections
import time
import torch 
from typing import Optional

import numpy as np
import pyaudio

from faster_whisper import WhisperModel

#####################
####
# Define the speech input process.
####
#####################

###
# Define and initialize VAD-controlled microphone input
###

import collections, queue
from queue import Queue
import webrtcvad, tqdm
import torchaudio, pyaudio
from scipy.io.wavfile import write as writewav

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS):
        def proxy_callback(in_data, frame_count, time_info, status):
            #pylint: disable=unused-argument
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None):
        super().__init__(device=device, input_rate=input_rate)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            raise Exception("Resampling required")

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. 
            Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None: frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        max_num_frames = 10000 / self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        speech_frames = []
        frameCounter = 0
        pbar = tqdm.tqdm(frameCounter, total=max_num_frames, colour='green')
        for frame in frames:
            if len(frame) < 640:
                return
            frameCounter += 1
            if frameCounter % 5 == 0:
                pbar.update(5)
            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        speech_frames.append(f)
                    ring_buffer.clear()
                    ratio = 0.9

            else:
                speech_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])

                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    for f, s in ring_buffer:
                        speech_frames.append(f)
                    ring_buffer.clear()
                    pbar.close()
                    return speech_frames



class FasterWhisperASR:
	"""Microphone ASR using Whisper with VAD-based endpointing."""

	def __init__(
		self,
		model_name: str = "small",		# Meeting: vervangen door FastWhisper.
		language: str = "nl",
        directory: str = None,
		sample_rate: int = 16000, 
		download_root: Optional[str] = "c:/Users/LouBo/Whisper/model_dir/",  ### MUST BE ADAPTED
	) -> None:
		self.language = language
		self.sample_rate = sample_rate
#		self.vad = webrtcvad.Vad(vad_aggressiveness)
###
# use cuda if available.
###
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.fp16 = self.device != "cpu"
		self.directory = directory
        
		self.model = WhisperModel(
			model_name,
			device=self.device,
			download_root=download_root,
			compute_type="float16",
		)


	def listen_and_transcribe(self, directory, prompt=None, hotwords=None):
        ###
        # Record some speech 
        ###

        # Start audio with VAD
		vad_audio = VADAudio(aggressiveness=3,
                             device=None,
                             input_rate=16000)

		frames = vad_audio.vad_collector()
		wav_data = bytearray()
		for frame in frames:
			wav_data.extend(frame)
		audio = np.frombuffer(wav_data, np.int16)
		if self.directory != None:
			ds = time.localtime()
			filename = str(ds.tm_hour) + '-' + str(ds.tm_min) + '-' + str(ds.tm_sec) + '.wav'
			writewav(self.directory + filename, 16000, audio)
       
		text = self.transcribeSpeech(audio,
                            prompt=prompt, 
                            hotwords=hotwords
                            )

		return text                            


        
	def transcribeSpeech(self, audio, prompt=None, hotwords=None):

		segments, info = self.model.transcribe(audio, 
                                       language='nl',
                                       beam_size=5,
                                       vad_filter=False,
                                       initial_prompt=prompt,
                                       hotwords=hotwords)
		fragments = []
		for segment in segments:
			fragments.append(segment.text)
		text = fragments[0]
		for ii in range(1, len(fragments)):
			text = text + ' ' + fragments[ii]

		return text


if __name__ == "__main__":
    
	ds = time.localtime()
	dumpdir = 'audioDir-' + str(ds.tm_mday) + '-' + str(ds.tm_mon) + '-' + str(ds.tm_hour) + '-' + str(ds.tm_min) + '/'
	os.makedirs(dumpdir, exist_ok=True)
	asr = FasterWhisperASR(model_name="large-v3", language="nl", directory=dumpdir)
	while True:
		print("Speak now. Recording will stop after silence or timeout.")
		text = asr.listen_and_transcribe(dumpdir)
		print(f"Transcription: {text}")
		if 'stop' in text:
			sys.exit()



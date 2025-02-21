import alsaaudio
import threading
import numpy as np
import librosa

def process_input_audio(data, gain=1.0):
    buf = np.frombuffer(data, dtype=np.int16)
    buf = buf.astype(np.float32)/float(2**15)
    buf *= gain
    return np.clip(buf, -1, 1)

class AudioCapture(threading.Thread):
    def __init__(self, output_audio):
        threading.Thread.__init__(self)
        self.output_audio = output_audio
        self.running = False
        
    def run(self, periodsize=1600):
        # Settings compatible with Whisper
        inp = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE,
            channels=1,
            rate=16000,
            format=alsaaudio.PCM_FORMAT_S16_LE,
            periodsize=periodsize,
            device="hw:Device,0"
        )
        
        self.running = True
        while self.running:
            length, data = inp.read()
            if length > 0:
                self.output_audio(process_input_audio(data))
        
    def stop(self):
        self.running = False

def process_output_audio(data):
    data = np.clip(data, -1, 1)
    data = (data*float(2**15)).astype(np.int16)
    data = np.column_stack((data, data))
    return data

def output_audio(data):
    data = process_output_audio(data)
    out = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK, channels=2, rate=16000,
                        format=alsaaudio.PCM_FORMAT_S16_LE, periodsize=160, device='hw:APE,0')
    if out.write(data) < 0:
        print("Playback buffer underrun! Continuing nonetheless ...")
    out.close()


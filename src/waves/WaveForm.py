import torch
from torchaudio.utils import download_asset
import torchaudio
from IPython.display import Audio
import matplotlib.pyplot as plt
from torchaudio.transforms import Resample  # If Resample is the correct function

class WaveForm:

    def __init__(self, path):
        self.wave, self.rate = torchaudio.load(download_asset(path))

    def play(self):
        waveform = self.wave
        if self.rate != 16000:
            waveform = Resample(self.rate, 16000)(waveform)
        return Audio(waveform.numpy()[0], rate=16000)
    
    def plot(self):
        waveform = self.wave.numpy()
        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / self.rate
        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c+1}")
        figure.suptitle("Waveform")

    def spectrogram(self):
        waveform = self.wave.numpy()
        num_channels, num_frames = waveform.shape
        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            # Replace the following line with the correct spectrogram function
            axes[c].plot_spectrogram(waveform[c], Fs=self.rate)  # Example placeholder
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c+1}")
        figure.suptitle("Spectrogram")

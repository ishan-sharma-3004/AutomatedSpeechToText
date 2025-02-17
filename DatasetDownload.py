import os
import wave
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import librosa.display
import seaborn as sns
import pyloudnorm as pyln
from glob import glob
from df.enhance import enhance, init_df, load_audio, save_audio

# Save normalized audio
def save_normalized_audio(audio, sample_rate, output_path):
    sf.write(output_path, audio, sample_rate)

# Normalize audio
def normalize(audioPath, outputDir):
    audio, sr = librosa.load(audioPath, sr=None)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    normalized_audio = pyln.normalize.loudness(audio, loudness, -12.0)
    peak_normalized_audio = pyln.normalize.peak(audio, -1.0)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    filename = os.path.basename(audioPath)
    outputPath = os.path.join(outputDir, f"normalized_{filename}")
    save_normalized_audio(normalized_audio, sr, outputPath)
    return outputPath

# Check if audio is noisy
def isNoisy(audioPath, threshold=0.01):
    try:
        y, sr = librosa.load(audioPath)
    except Exception as e:
        print(f"Error Loading Audio File {e}")
        return False
    rms = np.sqrt(np.mean(y**2))
    return rms > threshold

# Using glob to find all WAV files
folder = '/Users/ishansharma/Downloads/Projectsubmission/*/*.wav'
audio_files = glob(folder)

# Iterate over each file in the folder
for filename in audio_files:
    if filename.endswith(".wav"):
        with wave.open(filename, "rb") as f:
            channels = f.getnchannels()
            sampleRate = f.getframerate()
            bitDepth = f.getsampwidth() * 8
            frames = f.getnframes()

        # Load audio file
        audio, sr = librosa.load(filename, sr=None, mono=True)

        # # Plot waveform
        # plt.figure(figsize=(10, 6))
        # plt.plot(audio, label="Mono channel", color="Blue")
        # plt.title(f"Waveform - {filename}\nSample Rate: {sampleRate} Hz, Bit Depth: {bitDepth}-bit")
        # plt.xlabel("Time (samples)")
        # plt.ylabel('Amplitude')
        # plt.legend()
        # plt.show()

        # Plot time-domain signal
        # data, sr = sf.read(filename)
        # time = np.linspace(0, len(data) / sr, num=len(data))
        # plt.figure(figsize=(10, 5))
        # plt.plot(time, data, label=f"Sample Rate: {sr} Hz", color="green")
        # plt.xlabel("Time (seconds)")
        # plt.ylabel("Amplitude")
        # plt.title(f"Audio Signal with Sample Rate {sr} Hz")
        # plt.legend()
        # plt.show()

        # # Plot Mel spectrogram
        # melSpectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        # melDeb = librosa.power_to_db(melSpectrogram, ref=np.max)
        # plt.figure(figsize=(10, 4))
        # librosa.display.specshow(melDeb, x_axis='time', y_axis='mel', sr=sr)
        # plt.colorbar(label='dB')
        # plt.title("Mel Spectrogram")
        # plt.show()

        # If audio is noisy, enhance and save
        if isNoisy(filename):
            model, df_state, _ = init_df()
            audio, _ = load_audio(filename, sr=df_state.sr())
            enhanced = enhance(model, df_state, audio)  # Replace 'mode' with the actual mode
            save_audio(f"enhanced_{os.path.basename(filename)}", enhanced, df_state.sr())

            try:
                fileData, sampleRate = librosa.load(filename, sr=None)
                if fileData.size == 0:
                    print("Error: Audio File is Empty")
                absoluteValue = np.abs(fileData)
                peakIndex = np.argmax(absoluteValue)
                peakAmplitude = fileData[peakIndex]
            except Exception as e:
                print(f"An error occurred: {e}")
            outputDir = '/Users/ishansharma/Downloads/Projectsubmission/processed_data'
            if peakAmplitude > -8:
                normalizedFile = normalize(filename, outputDir)

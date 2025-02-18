import os
import warnings
import wave
from glob import glob

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from df.enhance import enhance, init_df, load_audio, save_audio

# Suppress warnings related to the df library
warnings.filterwarnings("ignore", category=UserWarning, module="df")

def normalize(audio_path, output_dir):
    '''Peak Normalization'''
    try:
        print(f"Starting normalization for: {audio_path}")
        audio, sr = librosa.load(path=audio_path, sr=None)
        
        # Ensure audio is not empty
        if len(audio) == 0:
            print(f"Error: Audio file {audio_path} is empty")
            return None
            
        # Convert to float32 if not already
        audio = audio.astype(np.float32)
        
        # Check for invalid values
        if np.isnan(audio).any() or np.isinf(audio).any():
            print(f"Error: Audio file {audio_path} contains invalid values")
            return None
            
        print("Calculating loudness...")
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        print(f"Original loudness: {loudness:.2f} LUFS")
        
        print("Applying loudness normalization...")
        normalized_audio = pyln.normalize.loudness(audio, input_loudness=loudness, target_loudness=-14.0)
        
        print("Applying peak normalization...")
        peak_normalized_audio = pyln.normalize.peak(normalized_audio, target=-1.0)
        
        # Verify the normalized audio
        if np.isnan(peak_normalized_audio).any() or np.isinf(peak_normalized_audio).any():
            print(f"Error: Normalization produced invalid values for {audio_path}")
            return None
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        filename = os.path.basename(audio_path)
        output_path = os.path.join(output_dir, f"normalized_{filename}")
        
        print(f"Saving normalized audio to: {output_path}")
        sf.write(output_path, peak_normalized_audio, sr)
        
        # Verify the file was created
        if os.path.exists(output_path):
            # Verify the saved file
            try:
                check_audio, check_sr = librosa.load(output_path, sr=None)
                if len(check_audio) > 0 and not np.isnan(check_audio).any():
                    print(f"Successfully normalized and saved: {output_path}")
                    return output_path
            except Exception as e:
                print(f"Error verifying saved file: {e}")
                
        print(f"Failed to save or verify normalized file: {output_path}")
        return None
    except Exception as e:
        print(f"Error normalizing audio {audio_path}: {str(e)}")
        return None

def is_noisy(audio_path, threshold=0.01):
    '''Checks for audio file to be noisy based on threshold'''
    try:
        print(f"Checking noise level for: {audio_path}")
        y, sr = librosa.load(audio_path)
        rms = np.sqrt(np.mean(y**2))
        is_noisy = rms > threshold
        print(f"Noise level: {rms:.4f} (threshold: {threshold})")
        return is_noisy
    except Exception as e:
        print(f"Error checking noise level: {str(e)}")
        return False

def process_audio_file(filename, output_dir, df_model=None, df_state=None):
    """Process a single audio file"""
    try:
        print(f"\nProcessing {filename}...")
        
        # First check if file exists and is valid
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return False
            
        # Read audio file properties
        with wave.open(filename, "rb") as f:
            channels = f.getnchannels()
            sample_rate = f.getframerate()
            bit_depth = f.getsampwidth() * 8
            frames = f.getnframes()
            print(f"Audio properties: {channels} channels, {sample_rate} Hz, {bit_depth}-bit")

        # Load audio for processing
        audio_series, sr = librosa.load(path=filename, sr=None, mono=True)
        print(f"Loaded audio with sample rate: {sr} Hz")

        if is_noisy(filename):
            print("Audio is noisy, applying enhancement...")
            if df_model is None or df_state is None:
                df_model, df_state, _ = init_df()
            
            audio, _ = load_audio(path=filename, sr=df_state.sr())
            enhanced = enhance(df_model, df_state, audio)
            
            # Save enhanced audio
            enhanced_filename = os.path.join(output_dir, f"enhanced_{os.path.basename(filename)}")
            save_audio(enhanced_filename, enhanced, df_state.sr())
            
            # Verify enhanced file
            if not os.path.exists(enhanced_filename):
                print(f"Failed to save enhanced file: {enhanced_filename}")
                return False
                
            print(f"Successfully saved enhanced file: {enhanced_filename}")
            
            # Normalize the enhanced audio
            print("Normalizing enhanced audio...")
            normalized_file = normalize(audio_path=enhanced_filename, output_dir=output_dir)
            
            if normalized_file:
                print(f"Successfully normalized file: {normalized_file}")
                return True
            else:
                print("Normalization failed")
                return False
        else:
            print("Audio is not noisy enough for processing")
            # Still normalize non-noisy files
            normalized_file = normalize(audio_path=filename, output_dir=output_dir)
            return normalized_file is not None
            
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return False

def main():
    # Set up directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_pattern = '/Users/ishansharma/Downloads/Projectsubmission/dataset/*/*.wav'
    output_dir = os.path.join(base_dir, "dataset", "processed_files")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of audio files
    audio_files = glob(input_pattern)
    if not audio_files:
        print(f"No WAV files found matching pattern: {input_pattern}")
        return
        
    print(f"Found {len(audio_files)} WAV files to process")
    
    # Initialize DeepFiltering model once
    print("Initializing DeepFiltering model...")
    df_model, df_state, _ = init_df()
    
    # Process files
    successful = 0
    for filename in audio_files:
        if process_audio_file(filename, output_dir, df_model, df_state):
            successful += 1
            
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed {successful} out of {len(audio_files)} files")
    print(f"Processed files can be found in: {output_dir}")
    
    # List all files in output directory
    output_files = os.listdir(output_dir)
    print(f"\nFiles in output directory ({len(output_files)} files):")
    for file in output_files:
        print(f"- {file}")

if __name__ == "__main__":
    main()
            
            
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
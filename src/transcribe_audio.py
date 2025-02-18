import os
import time
import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import (Speech2TextForConditionalGeneration,
                          Speech2TextProcessor)


def load_model():
    """Load the speech-to-text model and processor with error handling"""
    try:
        print("Loading speech-to-text model and processor...")
        model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
        processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
        print("Model loaded successfully!")
        return model, processor
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def transcribe_audio(audio, model, processor):
    '''Transcribes audio segment using speech2Text model.'''
    try:
        audio_array = np.array(audio)
        if np.std(audio_array) == 0 or np.isnan(audio_array).any():
            print("[WARNING] Audio segment has zero variance or contains NaN values. Skipping...")
            return ""
        
        # Normalize audio and ensure it's the right type
        normalized_audio = librosa.util.normalize(audio_array.astype(np.float32))
        
        # Process in smaller chunks if the audio is too long
        max_length = 200000  # Adjust this value based on your GPU memory
        if len(normalized_audio) > max_length:
            print("[INFO] Long audio detected, processing in chunks...")
            chunks = [normalized_audio[i:i + max_length] 
                     for i in range(0, len(normalized_audio), max_length)]
            transcription = ""
            for chunk in chunks:
                inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    generated_ids = model.generate(inputs["input_features"])
                chunk_transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
                transcription += " " + chunk_transcription[0]
            return transcription.strip()
        else:
            inputs = processor(normalized_audio, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                generated_ids = model.generate(inputs["input_features"])
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
            return transcription[0]
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        return ""

def split_audio_on_silence(audio, sample_rate, silence_threshold=-40, min_speech_length=0.5):
    '''Splits an Audio file into segments based on silence'''
    try:
        print("[INFO] Splitting audio on silence...")
        non_silent_intervals = librosa.effects.split(
            audio, top_db=abs(silence_threshold), frame_length=1024, hop_length=512
        )
        segments = [audio[start:end] for start, end in non_silent_intervals]
        print(f"[INFO] Found {len(segments)} segments")
        return segments, non_silent_intervals
    except Exception as e:
        print(f"[ERROR] Failed to split audio: {e}")
        return [], []

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    try:
        model, processor = load_model()
        model = model.to(device)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    folder = "/Users/ishansharma/Downloads/Projectsubmission/src/dataset/processed_files/"
    transcript_dict = {}
    
    # Get list of WAV files
    wav_files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    print(f"\nFound {len(wav_files)} WAV files to process")

    # Process each file with progress bar
    for file in tqdm(wav_files, desc="Processing files"):
        filename = os.path.join(folder, file)
        print(f"\n[INFO] Processing file: {filename}")
        start_time = time.time()
        
        try:
            # Load audio with progress indication
            print("[INFO] Loading audio file...")
            audio, sample_rate = librosa.load(filename, sr=16000)
            
            if len(audio) == 0 or np.isnan(audio).any():
                print(f"[WARNING] Skipping {file} due to invalid or empty audio.")
                continue

            # Split audio into segments
            segments, intervals = split_audio_on_silence(audio, sample_rate)
            transcript_dict[file] = []

            # Process segments with progress bar
            for i, (segment, (start, end)) in enumerate(tqdm(
                zip(segments, intervals), 
                desc="Processing segments", 
                total=len(segments)
            )):
                segment_transcription = transcribe_audio(segment, model, processor)
                if segment_transcription:  # Only add if transcription is not empty
                    transcript_dict[file].append({
                        "segment": i + 1,
                        "transcription": segment_transcription,
                        "start_time": round(start / sample_rate, 2),
                        "end_time": round(end / sample_rate, 2),
                    })
                    # Print immediate results
                    print(f"\nSegment {i+1} Transcription: {segment_transcription}")

            processing_time = time.time() - start_time
            print(f"[INFO] Completed processing {file} in {processing_time:.2f} seconds")

        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")
            continue

    # Save results to a file
    output_file = "transcription_results.txt"
    print(f"\nSaving results to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for file, transcriptions in transcript_dict.items():
            f.write(f"\n\nTranscriptions for {file}:\n")
            for entry in transcriptions:
                f.write(f"\nSegment {entry['segment']} - {entry['transcription']}\n")
                f.write(f"Start Time: {entry['start_time']}s, End Time: {entry['end_time']}s\n")

    # Print final results
    print("\nFinal Results:")
    for file, transcriptions in transcript_dict.items():
        print(f"\n[RESULT] Transcriptions for {file}:")
        for entry in transcriptions:
            print(f"\nSegment {entry['segment']} - {entry['transcription']}")
            print(f"Start Time: {entry['start_time']}s, End Time: {entry['end_time']}s")
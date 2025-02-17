import os
import librosa
import torch
import numpy as np
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

# Load the model and processor
model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

def transcribe_audio(audio):
    """
    Transcribes an audio segment using Facebook's Speech2Text model.
    Handles silent or corrupted audio to prevent runtime errors.
    """
    audio = np.array(audio)

    # Check for invalid or silent audio
    if np.std(audio) == 0 or np.isnan(audio).any():
        print("[WARNING] Audio segment has zero variance or contains NaN values. Skipping...")
        return ""

    # Normalize audio to prevent numerical instability
    audio = librosa.util.normalize(audio)

    # Process the audio using the processor
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    # Generate transcription
    with torch.no_grad():
        generated_ids = model.generate(inputs["input_features"])

    # Decode the transcription
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return transcription[0]

def split_audio_on_silence(audio, sample_rate, silence_threshold=-40, min_speech_length=0.5):
    """
    Splits an audio file into segments based on silence.
    Uses librosa's `effects.split()` to identify non-silent intervals.
    """
    non_silent_intervals = librosa.effects.split(audio, top_db=abs(silence_threshold), frame_length=1024, hop_length=512)

    # Extract audio segments
    segments = [audio[start:end] for start, end in non_silent_intervals]
    return segments, non_silent_intervals

# Path to folder containing .wav files
folder = "/Users/ishansharma/Downloads/Projectsubmission/processed_data/"
transcript_dict = {}

for file in os.listdir(folder):
    if file.endswith(".wav"):
        filename = os.path.join(folder, file)

        print(f"[INFO] Processing file: {filename}")

        try:
            # Load the audio file
            audio, sample_rate = librosa.load(filename, sr=16000)

            # Check if audio is empty or invalid
            if len(audio) == 0 or np.isnan(audio).any():
                print(f"[WARNING] Skipping {file} due to invalid or empty audio.")
                continue

            # Split the audio based on silence
            segments, intervals = split_audio_on_silence(audio, sample_rate)

            # Store transcriptions
            transcript_dict[file] = []
            for i, (segment, (start, end)) in enumerate(zip(segments, intervals)):
                segment_transcription = transcribe_audio(segment)

                transcript_dict[file].append({
                    "segment": i + 1,
                    "transcription": segment_transcription,
                    "start_time": round(start / sample_rate, 2),
                    "end_time": round(end / sample_rate, 2)
                })

        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")

# Print transcriptions
for file, transcriptions in transcript_dict.items():
    print(f"\n[RESULT] Transcriptions for {file}:")
    for entry in transcriptions:
        print(f" Segment {entry['segment']} - {entry['transcription']}")
        print(f" Start Time: {entry['start_time']}s, End Time: {entry['end_time']}s")

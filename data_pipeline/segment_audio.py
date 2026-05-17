import os
import subprocess
import librosa
from pathlib import Path

def segment_audio(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if not file.endswith(".wav"):
            continue
        
        input_path = os.path.join(input_dir, file)
        filename = Path(file).stem
        output_pattern = os.path.join(output_dir, f"{filename}_%03d.wav")
        
        # Check if it was already segmented by looking for the first segment
        first_segment = os.path.join(output_dir, f"{filename}_000.wav")
        if os.path.exists(first_segment):
            print(f"Skipping {file}, already segmented.")
            continue
        
        command = [
            "ffmpeg", "-y", "-i", input_path, "-af", "silencedetect=noise=-30dB:d=0.5",
            "-f", "segment", "-segment_time", "10",
            "-ar", "16000", "-ac", "1", output_pattern
        ]
        
        print(f"Segmenting {file}")
        # Capture output to avoid cluttering, or let it print
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def filter_clips(clips_dir):
    durations = []
    print(f"Filtering clips in {clips_dir}...")
    for file in os.listdir(clips_dir):
        if not file.endswith(".wav"):
            continue
            
        path = os.path.join(clips_dir, file)
        try:
            audio, sr = librosa.load(path, sr=16000)
            duration = len(audio) / sr
            
            # Remove clips outside the 1 to 20 seconds range
            if duration < 1 or duration > 20:
                print(f"Removing {file} (duration: {duration:.2f}s)")
                os.remove(path)
            else:
                durations.append(duration)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    print(f"\nTotal valid clips kept: {len(durations)}")
    if durations:
        print(f"Average clip duration: {sum(durations)/len(durations):.2f} seconds")

if __name__ == "__main__":
    segment_audio("dataset/telugu_audio", "dataset/telugu_audio_clips")
    filter_clips("dataset/telugu_audio_clips")

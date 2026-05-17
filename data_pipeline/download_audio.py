import os
import subprocess
import pandas as pd
import librosa

def download_audio(summary_csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df_titles = pd.read_csv(summary_csv_path)
    video_ids = df_titles["id"].tolist()
    
    for vid in video_ids:
        output_file = os.path.join(output_dir, vid)
        if os.path.exists(f"{output_file}.wav"):
            print(f"Skipping {vid}, already downloaded.")
            continue
        command = [
            "yt-dlp", "-f", "bestaudio/best", "-o", output_file,
            "--extract-audio", "--audio-format", "wav", "--audio-quality", "0",
            "--no-playlist", "--ppa", "ffmpeg:-ac 1 -ar 16000",
            f"https://youtu.be/{vid}"
        ]
        print(f"Downloading: {vid}")
        subprocess.run(command)

def calculate_total_duration(audio_dir):
    total_duration = 0
    for file in os.listdir(audio_dir):
        if file.endswith(".wav"):
            try:
                audio, sr = librosa.load(os.path.join(audio_dir, file), sr=16000)
                total_duration += len(audio) / sr
            except Exception as e:
                print(f"Error loading {file}: {e}")
    print(f"Total hours downloaded: {total_duration / 3600:.2f}")

if __name__ == "__main__":
    download_audio("selected_telugu_summary.csv", "dataset/telugu_audio")
    calculate_total_duration("dataset/telugu_audio")

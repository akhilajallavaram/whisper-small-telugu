import os
import pandas as pd
from tqdm import tqdm
import torch
import whisper

def generate_transcripts(clips_dir, output_csv, model_size="medium", buffer_size=50):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size, device=device)

    # Collect audio files
    audio_files = sorted([
        os.path.join(clips_dir, f)
        for f in os.listdir(clips_dir)
        if f.endswith(".wav")
    ])

    print("Total clips:", len(audio_files))

    # Resume support
    processed_files = set()
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        processed_files = set(existing_df["path"].tolist())
        print("Already processed:", len(processed_files))
    else:
        pd.DataFrame(columns=["path", "text"]).to_csv(output_csv, index=False)

    buffer = []

    for audio_path in tqdm(audio_files, desc="Transcribing"):
        if audio_path in processed_files:
            continue

        try:
            result = model.transcribe(
                audio_path,
                language="te",
                task="transcribe",
                fp16=(device != "cpu"),
                temperature=0,
                beam_size=1
            )

            text = result["text"].strip()

            buffer.append({
                "path": audio_path,
                "text": text
            })

            # Write in chunks
            if len(buffer) >= buffer_size:
                pd.DataFrame(buffer).to_csv(
                    output_csv, mode="a", header=False, index=False, encoding="utf-8"
                )
                buffer = []

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    # Write remaining buffer
    if buffer:
        pd.DataFrame(buffer).to_csv(
            output_csv, mode="a", header=False, index=False, encoding="utf-8"
        )

    print(f"Transcription process finished. Saved to {output_csv}")

if __name__ == "__main__":
    os.makedirs("dataset", exist_ok=True)
    generate_transcripts(
        clips_dir="dataset/telugu_audio_clips",
        output_csv="dataset/train.csv",
        model_size="large",
        buffer_size=50
    )

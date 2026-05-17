# Whisper Telugu - Data Pipeline & Finetuning

🔗 **Hugging Face Model:** [AkhilaJallavaram24/whisper-small-telugu-lora](https://huggingface.co/AkhilaJallavaram24/whisper-small-telugu-lora)

This project is an end-to-end orchestration pipeline to extract Telugu audio data from YouTube, process it into a clean dataset, and fine-tune an OpenAI Whisper model using Parameter-Efficient Fine-Tuning (PEFT/LoRA).

## 📁 Project Structure

The repository is divided into two main components:
- **`data_pipeline/`**: Handles scraping metadata, downloading YouTube audio, chunking the audio into small segments, filtering bad segments, and using a base Whisper model to automatically transcribe the segments.
- **`finetune/`**: Handles dataset splitting, data collating, setting up LoRA adapters, and fine-tuning the Whisper model using Hugging Face's `Seq2SeqTrainer`.

```text
whishper_telugu/
├── README.md
├── requirements.txt
├── data_pipeline/
│   ├── extract_telugu_data.py      # Extracts metadata and creates target subset
│   ├── download_audio.py           # Downloads audio via yt-dlp
│   ├── segment_audio.py            # Segments audio (10s chunks) and filters anomalies
│   ├── generate_transcripts.py     # Transcribes segments to create training CSV
│   └── pipeline.py                 # Orchestrates the execution of all the above steps
└── finetune/
    ├── run_training.py             # Entry point where hyperparameters are defined
    ├── train_pipeline.py           # The core training loop and dataset prep logic
    └── finetune_utils.py           # Custom DataCollator, metric functions, and utilities
```

---

## 🛠️ Installation

Before running any pipelines, ensure you have installed the required dependencies.

```bash
pip install -r requirements.txt
```

Additionally, the pipeline relies on the following system packages which must be installed:
- `ffmpeg` (for audio chunking and noise detection)
- `yt-dlp` (for YouTube downloads, also included in python requirements)

---

## 🚀 Phase 1: Data Pipeline

The data pipeline takes your raw dataset metadata and turns it into a perfectly formatted, transcript-paired `train.csv` file ready for fine-tuning. 

The pipeline is fully resumable; if it gets interrupted, running it again will automatically skip previously downloaded or transcribed files!

### Running the Data Pipeline
```bash
cd data_pipeline
python3 pipeline.py
```

### What happens under the hood?
1. **`extract_telugu_data.py`**: Reads `video_ids_metadata_Telugu.csv` and `video_ids_title_Telugu.csv`, shuffles them, and extracts enough videos to hit the target duration (e.g. 100 hours).
2. **`download_audio.py`**: Downloads the raw audio of the selected videos as `.wav` files at 16kHz using `yt-dlp`.
3. **`segment_audio.py`**: Uses `ffmpeg` to chop the long audio files into smaller chunks (roughly 10 seconds), and removes any chunks that are too short (<1s) or too long (>20s).
4. **`generate_transcripts.py`**: Runs the `openai/whisper-large` model locally to auto-transcribe the segmented chunks and saves the path-text pairs into `dataset/train.csv`.

---

## 🧠 Phase 2: Whisper Finetuning

Once the data pipeline finishes, you will have a `dataset/train.csv` file containing audio paths and their corresponding transcriptions. The finetuning module takes this CSV and trains a low-rank adapter (LoRA) on top of a base Whisper model.

### Running the Finetuning Pipeline
```bash
cd finetune
python3 run_training.py
```

### Configuring Hyperparameters
All training hyperparameters are completely decoupled from the training logic. Open `finetune/run_training.py` to customize your run.

```python
hyperparameters = {
    "dataset_csv": "../dataset/train.csv",
    "model_name": "openai/whisper-small",
    "language": "Telugu",
    "lora_r": 16,
    "lora_alpha": 32,
    "batch_size": 8,
    "grad_acc": 2,
    "learning_rate": 1e-4,
    "epochs": 5,
    "fp16": True,
    # ... and more
}
```

### What happens under the hood?
1. **Splits & Cleans Data**: Takes your single CSV, removes empty transcripts, and splits it into `train`, `val`, and `test` sets.
2. **Feature Extraction**: Uses `WhisperProcessor` to convert raw audio arrays into log-Mel spectrograms.
3. **PEFT/LoRA Injection**: Wraps the base Whisper model with a LoRA adapter specifically targeting the `q_proj` and `v_proj` attention matrices.
4. **Seq2Seq Training**: Uses Hugging Face's `Seq2SeqTrainer` and a custom `SafeDataCollator` to train the model while automatically calculating Word Error Rate (WER) after each epoch. The best model is saved to the `output_dir`.

---

## 🙏 Acknowledgements

This project uses the **MahaDhwani** dataset provided by **AI4Bharat**.I'm deeply grateful to AI4Bharat for open-sourcing this incredible dataset and making it available on GitHub. You can find their amazing work on their official [GitHub repository](https://github.com/AI4Bharat/MahaDhwani).

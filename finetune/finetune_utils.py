import torch
import evaluate
import librosa
from transformers import WhisperProcessor

def load_audio(batch):
    audio, sr = librosa.load(batch["path"], sr=16000)
    batch["audio"] = audio
    batch["sampling_rate"] = sr
    return batch

def filter_long_labels(sample, max_target_tokens=448):
    # Check if labels sequence is within the limit
    return len(sample['labels']) <= max_target_tokens

class SafeDataCollator:
    def __init__(self, processor, max_target_tokens=448):
        self.processor = processor
        self.max_target_tokens = max_target_tokens

    def __call__(self, features):
        input_features = [f["input_features"] for f in features]
        label_features = [f["labels"] for f in features]

        # Truncate EVERY label to max_target_tokens
        label_features = [
            (l[:self.max_target_tokens] if isinstance(l, list) else l.tolist()[:self.max_target_tokens])
            for l in label_features
        ]

        batch = self.processor.feature_extractor.pad(
            {"input_features": input_features},
            return_tensors="pt"
        )

        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_features},
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Emergency truncation
        if labels.shape[1] > self.max_target_tokens:
            labels = labels[:, :self.max_target_tokens]

        batch["labels"] = labels
        return batch

def get_compute_metrics_fn(processor):
    wer_metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
        
    return compute_metrics

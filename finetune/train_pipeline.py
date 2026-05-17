import os
import torch
import pandas as pd
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from .finetune_utils import SafeDataCollator, get_compute_metrics_fn

def split_and_clean_data(csv_path, output_dir, test_size=0.15, val_size=0.5):
    from sklearn.model_selection import train_test_split
    
    print("Splitting and cleaning dataset...")
    df = pd.read_csv(csv_path)
    
    # First split: train vs temp (val+test)
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=42, shuffle=True)
    # Second split: val vs test
    val_df, test_df = train_test_split(temp_df, test_size=val_size, random_state=42, shuffle=True)
    
    splits = {"train": train_df, "val": val_df, "test": test_df}
    cleaned_paths = {}
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_df in splits.items():
        # Clean nulls
        split_df = split_df.dropna(subset=["text"])
        split_df = split_df[split_df["text"].str.strip() != ""]
        
        path = os.path.join(output_dir, f"{split_name}_clean.csv")
        split_df.to_csv(path, index=False)
        cleaned_paths[split_name] = path
        
        print(f"{split_name} samples: {len(split_df)}")
        
    return cleaned_paths

def run_whisper_finetuning(
    dataset_csv,
    dataset_output_dir="../dataset",
    model_name="openai/whisper-small",
    language="Telugu",
    task="transcribe",
    output_dir="../whisper_telugu_output",
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    batch_size=8,
    grad_acc=2,
    learning_rate=1e-4,
    epochs=5,
    warmup_steps=100,
    fp16=True,
    max_target_tokens=448
):
    # 1. Clean and split data
    data_paths = split_and_clean_data(dataset_csv, dataset_output_dir)
    
    data_files = {
        "train": data_paths["train"],
        "validation": data_paths["val"],
        "test": data_paths["test"]
    }
    
    dataset = load_dataset("csv", data_files=data_files)
    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))
    
    # 2. Load Processor
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    
    # 3. Prepare dataset
    def prepare_dataset(batch):
        audio = batch["path"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    print("Extracting features (this might take a while)...")
    dataset = dataset.map(
        prepare_dataset, 
        remove_columns=dataset["train"].column_names,
        num_proc=1
    )
    
    # 4. Filter long sequences
    def filter_long_labels(sample):
        return len(sample['labels']) <= max_target_tokens

    print("Filtering sequences that are too long...")
    dataset["train"] = dataset["train"].filter(filter_long_labels)
    dataset["validation"] = dataset["validation"].filter(filter_long_labels)
    
    print(f"Train samples after filtering: {len(dataset['train'])}")
    print(f"Validation samples after filtering: {len(dataset['validation'])}")
    
    # 5. Load model & PEFT
    print("Loading base model and configuring LoRA...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 6. Training Setup
    data_collator = SafeDataCollator(processor, max_target_tokens=max_target_tokens)
    compute_metrics = get_compute_metrics_fn(processor)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=fp16,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        processing_class=processor,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # 7. Train
    print("Starting training...")
    trainer.train()
    
    # 8. Save
    final_model_dir = os.path.join(output_dir, "final_model")
    print(f"Saving final model to {final_model_dir}...")
    trainer.save_model(final_model_dir)
    processor.save_pretrained(final_model_dir)
    print("Training complete!")


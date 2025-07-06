# To successfully run this script, you need to install the required packages in a specific order
# due to a dependency issue with the 'flash-attn' package. Please follow these steps:
#
# 1. First, install 'torch' by itself:
#    pip install torch
#
# 2. Next, install 'flash-attn' using the '--no-build-isolation' flag. This is crucial
#    as it ensures the installer uses the version of 'torch' you just installed:
#    pip install flash-attn --no-build-isolation
#
# 3. Finally, install all the other required packages:
#    pip install transformers datasets peft trl bitsandbytes accelerate wandb numpy
#

import os
import torch
import transformers
import warnings
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import Dataset, load_dataset
import json
import wandb
import numpy as np
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer, SFTConfig

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")

class Config:
    """
    Configuration parameters for the training run (No Flash Attention).
    """
    # Model and Tokenizer
    MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

    # Training Environment
    OUTPUT_DIR_BASE = "finetuned-phi35-no-flash-attn"
    WANDB_PROJECT = "phi35-long-context-finetuning"
    
    # Dataset
    DATASET_PATH = "Instruction_tuning_phase/dataset/dataset.json"
    
    # QLoRA & Training Hyperparameters
    MAX_SEQ_LENGTH = 8192
    
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 16

    # Hyperparameters
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.05
    
    # LoRA Configuration (Adjusted for less memory)
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    
    # Evaluation and Logging - Adjusted for a larger dataset
    EVAL_STEPS = 50
    LOGGING_STEPS = 10
    SAVE_STEPS = 100
    VALIDATION_RATIO = 0.10


def format_instruction(example: dict) -> str:
    """
    Formats a data sample into the format required by Phi-3.
    NOTE: The 'input' and 'output' fields are pre-formatted into JSON strings
    during the data loading phase to ensure type consistency.
    """
    input_str = example['input']
    output_str = example['output']
    return (
        f"<|system|>\nYou are an expert assistant for generating synthetic data. "
        f"Analyze the user's request and generate a detailed, high-quality synthetic dataset entry. IMPORTANT: Generate unique objects, do not duplicate input examples, and convert all image URLs to 'img_keyword+keyword+keyword' format.<|end|>\n"
        f"<|user|>\n{example['instruction']}\n\n{input_str}<|end|>\n"
        f"<|assistant|>\n{output_str}<|end|>"
    )

def create_split(dataset: Dataset, tokenizer, validation_ratio: float = 0.1):
    """
    Creates a split for the dataset. First, it filters out examples that are
    longer than Config.MAX_SEQ_LENGTH to avoid truncation, then performs a
    random split.
    """
    original_size = len(dataset)
    print(f"\nFiltering dataset to remove examples longer than {Config.MAX_SEQ_LENGTH} tokens...")

    # The SFTTrainer requires a 'text' column. We'll create it by formatting the prompt.
    dataset = dataset.map(lambda example: {'text': format_instruction(example)})
    
    # Now, filter the dataset based on the token count of the newly created 'text' field.
    filtered_dataset = dataset.filter(
        lambda example: len(tokenizer.encode(example['text'])) <= Config.MAX_SEQ_LENGTH
    )
    
    removed_count = original_size - len(filtered_dataset)
    print(f"✅ Filtering complete. Removed {removed_count} out of {original_size} examples.")

    if len(filtered_dataset) == 0:
        raise ValueError("Dataset is empty after filtering. Check MAX_SEQ_LENGTH or your data.")

    print("\nℹ️  Performing a random split on the filtered dataset.")
    split = filtered_dataset.train_test_split(test_size=validation_ratio, seed=42)

    print(f"\n📊 Random Split: {len(split['train'])} train, {len(split['test'])} val")
    return split['train'], split['test']


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Dynamically create output directory based on the dataset name
    dataset_name = os.path.splitext(os.path.basename(Config.DATASET_PATH))[0]
    output_dir = f"{Config.OUTPUT_DIR_BASE}-{dataset_name}"
    print(f"Output directory will be: {output_dir}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
        attn_implementation="flash_attention_2",
    )

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules="all-linear",
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    print("Loading and preparing dataset...")
    # Robustly load the JSON data to handle potential inconsistencies.
    print(f"Loading data from {Config.DATASET_PATH}...")
    with open(Config.DATASET_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Pre-process data to ensure type consistency that 'datasets' expects.
    # This converts both 'input' and 'output' to formatted JSON strings.
    processed_data = []
    for item in raw_data:
        processed_data.append({
            'instruction': item.get('instruction', ''),
            'input': json.dumps(item.get('input', ''), indent=2),
            'output': json.dumps(item.get('output', {}), indent=2)
        })
    
    full_dataset = Dataset.from_list(processed_data)

    # Pass the tokenizer to create_split to handle token counting for filtering.
    train_dataset, eval_dataset = create_split(full_dataset, tokenizer, validation_ratio=Config.VALIDATION_RATIO)
    print(f"\n✅ Using {len(train_dataset)} samples for training and {len(eval_dataset)} for evaluation.")

    # CORRECT: Use SFTConfig and re-enable evaluation parameters.
    training_args = SFTConfig(
        # SFT-specific parameters
        dataset_text_field="text",
        packing=True,
        max_seq_length=Config.MAX_SEQ_LENGTH,
        
        # TrainingArguments parameters
        output_dir=output_dir,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=Config.NUM_EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        
        optim="adafactor",
        lr_scheduler_type="cosine",
        warmup_ratio=Config.WARMUP_RATIO,
        weight_decay=Config.WEIGHT_DECAY,
        
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        group_by_length=True,
        
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=Config.LOGGING_STEPS,
        save_strategy="steps",
        save_steps=Config.SAVE_STEPS,
        save_total_limit=2,
        
        # Re-enable evaluation for the full dataset run
        eval_strategy="steps",
        eval_steps=Config.EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        report_to="wandb",
        run_name=f"phi35-flash-attn-8k-full-dataset",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
    )

    print("🚀 Starting training with FlashAttention...")
    torch.cuda.empty_cache()
    trainer.train()

    final_save_path = os.path.join(output_dir, "final-checkpoint")
    print(f"✅ Training complete. Saving final model to {final_save_path}")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    
    print("All done!")


if __name__ == "__main__":
    wandb.login()
    main()
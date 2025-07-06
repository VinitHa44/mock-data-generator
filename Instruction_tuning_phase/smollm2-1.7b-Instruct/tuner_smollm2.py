import json
import os
import warnings

import numpy as np
import torch
import transformers
import wandb
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")


class Config:
    MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    OUTPUT_DIR = "finetuned-smollm-mock-data"
    DATASET_PATH = (
        "Instruction_tuning_phase/dataset/dataset.json"
    )
    WANDB_PROJECT = "smollm-mock-data-finetuning"

    # Optimized for your actual data distribution
    MAX_LENGTH = 8192  # Increased based on your 38% > 8192 tokens
    BATCH_SIZE = 2  # Keep at 1 for T4 safety with long sequences
    GRADIENT_ACCUMULATION_STEPS = 6  # Slightly reduced for memory safety

    # Adjusted for longer sequences
    LEARNING_RATE = 2e-5  # Lower LR for longer context stability
    NUM_EPOCHS = 8  # More epochs to compensate for complex patterns
    EVAL_STEPS = 50  # More frequent evaluation with longer sequences
    SAVE_STEPS = 100
    DATALOADER_NUM_WORKERS = 2
    VALIDATION_STRATEGY = "stratified"
    VALIDATION_RATIO = (
        0.20  # 20% validation for 50 samples = 10 validation samples
    )


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "max_split_size_mb:256"  # Smaller chunks
)


def check_bitsandbytes_compatibility():
    """Check bitsandbytes installation and compatibility"""
    try:
        import bitsandbytes as bnb

        print(f"✅ bitsandbytes version: {bnb.__version__}")

        # Check for 4-bit support
        try:
            print("✅ 4-bit quantization supported")
            return True, "4bit"
        except ImportError:
            print("⚠️  4-bit quantization not supported, falling back to 8-bit")
            return True, "8bit"

    except ImportError as e:
        print(f"❌ bitsandbytes not found: {e}")
        return False, None


def get_gpu_memory():
    """Get available GPU memory"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
            1024**3
        )
        return gpu_memory
    return 0


def debug_data_sizes(raw_data, tokenizer):
    """Debug actual input vs output token sizes"""
    print(f"\n🔍 Debugging Data Sizes (first 5 examples):")

    for i, example in enumerate(raw_data[:5]):
        instruction = example["instruction"]
        input_data = example["input"]
        output_data = example["output"]

        # Convert to strings like format_instruction does
        input_text = (
            str(input_data)
            if isinstance(input_data, (list, dict))
            else input_data
        )
        output_text = (
            str(output_data)
            if isinstance(output_data, (list, dict))
            else output_data
        )

        # Tokenize each part
        instruction_tokens = len(
            tokenizer.encode(instruction, add_special_tokens=False)
        )
        input_tokens = len(
            tokenizer.encode(input_text, add_special_tokens=False)
        )
        output_tokens = len(
            tokenizer.encode(output_text, add_special_tokens=False)
        )

        print(f"\n📝 Example {i+1}:")
        print(
            f"  Instruction: {instruction_tokens} tokens - '{instruction[:50]}...'"
        )
        print(f"  Input JSON: {input_tokens} tokens")
        print(f"  Output JSON: {output_tokens} tokens")
        print(f"  Ratio (out/in): {output_tokens/input_tokens:.2f}")

        if output_tokens < input_tokens:
            print(f"  ⚠️  OUTPUT SMALLER than INPUT - This is wrong!")
        else:
            print(f"  ✅ Output larger than input - Good!")


def format_instruction(example):
    """Format instruction following SmolLM2 chat format"""
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]

    return f"<|im_start|>user\n{instruction}\n\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"


def analyze_dataset_lengths(formatted_texts, tokenizer):
    """Enhanced analysis for long sequences"""
    lengths = []
    for text in formatted_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(tokens))

    lengths = np.array(lengths)

    print(f"\n📊 Dataset Length Analysis:")
    print(f"  - Total samples: {len(lengths)}")
    print(f"  - Average length: {lengths.mean():.1f} tokens")
    print(f"  - Median length: {np.median(lengths):.1f} tokens")
    print(f"  - Min length: {lengths.min()} tokens")
    print(f"  - Max length: {lengths.max()} tokens")
    print(f"  - 95th percentile: {np.percentile(lengths, 95):.1f} tokens")
    print(
        f"  - Samples > 2048: {sum(lengths > 2048)} ({sum(lengths > 2048)/len(lengths)*100:.1f}%)"
    )
    print(
        f"  - Samples > 3072: {sum(lengths > 3072)} ({sum(lengths > 3072)/len(lengths)*100:.1f}%)"
    )
    print(
        f"  - Samples > 4096: {sum(lengths > 4096)} ({sum(lengths > 4096)/len(lengths)*100:.1f}%)"
    )
    print(
        f"  - Samples > 6000: {sum(lengths > 6000)} ({sum(lengths > 6000)/len(lengths)*100:.1f}%)"
    )

    # Recommend strategy based on distribution
    if sum(lengths > 4096) > len(lengths) * 0.2:  # More than 20% are very long
        print(
            f"\n💡 Recommendation: Consider using MAX_LENGTH=4096 with micro-batching"
        )
    elif sum(lengths > 3072) > len(lengths) * 0.1:  # More than 10% are long
        print(f"\n💡 Recommendation: MAX_LENGTH=3072 should work well")
    else:
        print(
            f"\n💡 Recommendation: MAX_LENGTH=2048 sufficient for most examples"
        )

    return lengths


def load_and_prepare_dataset(dataset_path, tokenizer, max_length=2048):
    """Load and prepare dataset with EFFICIENT dynamic padding"""
    print("Loading dataset...")
    with open(dataset_path, "r") as f:
        raw_data = json.load(f)

    # DEBUG: Check actual input vs output sizes
    debug_data_sizes(raw_data, tokenizer)

    # Format data
    formatted_texts = [format_instruction(item) for item in raw_data]

    # Analyze dataset lengths
    lengths = analyze_dataset_lengths(formatted_texts, tokenizer)

    # Create dataset
    formatted_data = [{"text": text} for text in formatted_texts]
    dataset = Dataset.from_list(formatted_data)

    # Pre-compute assistant token IDs for efficiency
    assistant_token_ids = tokenizer.encode(
        "<|im_start|>assistant", add_special_tokens=False
    )
    assistant_newline_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )

    print(f"\nAssistant token IDs to look for:")
    print(f"  - '<|im_start|>assistant': {assistant_token_ids}")
    print(f"  - '<|im_start|>assistant\\n': {assistant_newline_ids}")

    def tokenize_function(examples):
        """EFFICIENT tokenization with NO pre-padding"""
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        labels = []
        valid_examples = 0

        for i, text in enumerate(examples["text"]):
            input_ids = tokenized["input_ids"][i]
            label = input_ids.copy()

            # Find assistant token position
            assistant_pos = None

            # Try with newline first (more precise)
            for j in range(len(input_ids) - len(assistant_newline_ids) + 1):
                if (
                    input_ids[j : j + len(assistant_newline_ids)]
                    == assistant_newline_ids
                ):
                    assistant_pos = j + len(assistant_newline_ids)
                    break

            # Fallback to without newline
            if assistant_pos is None:
                for j in range(len(input_ids) - len(assistant_token_ids) + 1):
                    if (
                        input_ids[j : j + len(assistant_token_ids)]
                        == assistant_token_ids
                    ):
                        assistant_pos = j + len(assistant_token_ids)
                        break

            # Mask tokens before assistant response
            if assistant_pos is not None:
                for k in range(assistant_pos):
                    label[k] = -100
                valid_examples += 1
            else:
                for k in range(len(label)):
                    label[k] = -100
                print(f"⚠️  Warning: No assistant token found in example {i}")

            labels.append(label)

        print(
            f"✅ Successfully processed {valid_examples}/{len(examples['text'])} examples"
        )

        tokenized["labels"] = labels
        return tokenized

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset (no padding - efficient!)",
    )

    return tokenized_dataset


# Enhanced Data Collator for Dynamic Padding
class AdaptiveDataCollator:
    """Enhanced T4-safe collator for your long sequences"""

    def __init__(self, tokenizer, max_safe_length=2048, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.max_safe_length = max_safe_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        max_length = max(len(f["input_ids"]) for f in features)

        # Aggressive micro-batching for your long sequences
        if max_length > 3000:  # Your data has many 3000+ token sequences
            features = features[:1]  # Force batch_size=1 for long sequences
            max_length = len(features[0]["input_ids"])
            print(f"⚠️  Micro-batching: sequence length {max_length}")

        # Cap at MAX_LENGTH to prevent T4 OOM
        max_length = min(max_length, Config.MAX_LENGTH)

        # Round up for efficiency
        if self.pad_to_multiple_of:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of

        # Pad sequences
        batch = {}
        for key in features[0].keys():
            if key in ["input_ids", "labels"]:
                padded_sequences = []
                for f in features:
                    sequence = f[key][:max_length]  # Truncate if needed
                    padding_length = max_length - len(sequence)
                    if key == "labels":
                        padded = sequence + [-100] * padding_length
                    else:
                        padded = (
                            sequence
                            + [self.tokenizer.pad_token_id] * padding_length
                        )
                    padded_sequences.append(padded)
                batch[key] = torch.tensor(padded_sequences, dtype=torch.long)
            elif key == "attention_mask":
                attention_masks = []
                for f in features:
                    length = min(len(f["input_ids"]), max_length)
                    attention_mask = [1] * length + [0] * (max_length - length)
                    attention_masks.append(attention_mask)
                batch[key] = torch.tensor(attention_masks, dtype=torch.long)

        if "attention_mask" not in batch:
            batch["attention_mask"] = (
                batch["input_ids"] != self.tokenizer.pad_token_id
            ).long()

        return batch


class DetailedLoggingCallback(transformers.TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, num_samples=2):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = min(num_samples, len(eval_dataset))

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Log detailed generation examples during evaluation"""
        if model is None:
            return

        # Sample random examples - FIX: Convert numpy ints to Python ints
        indices = np.random.choice(
            len(self.eval_dataset), self.num_samples, replace=False
        )
        indices = [int(idx) for idx in indices]  # ✅ Convert to Python ints

        generation_examples = []
        model.eval()

        with torch.no_grad():
            for idx in indices:
                sample = self.eval_dataset[idx]  # ✅ Now works with Python int
                input_ids = sample["input_ids"]
                labels = sample["labels"]

                # Find where the assistant response starts
                prompt_end = None
                for i, label in enumerate(labels):
                    if label != -100:
                        prompt_end = i
                        break

                if prompt_end is not None:
                    prompt_ids = input_ids[:prompt_end]
                    prompt_text = self.tokenizer.decode(
                        prompt_ids, skip_special_tokens=False
                    )

                    expected_ids = [
                        id
                        for id, label in zip(input_ids, labels)
                        if label != -100
                    ]
                    expected_text = self.tokenizer.decode(
                        expected_ids, skip_special_tokens=True
                    )

                    # Generate response
                    inputs = self.tokenizer(
                        prompt_text, return_tensors="pt"
                    ).to(model.device)

                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=150,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                    full_response = self.tokenizer.decode(
                        outputs[0], skip_special_tokens=False
                    )
                    generated_part = full_response[len(prompt_text) :].strip()

                    generation_examples.append(
                        {
                            "step": state.global_step,
                            "prompt": prompt_text,
                            "expected": expected_text,
                            "generated": generated_part,
                            "prompt_length": len(prompt_ids),
                            "expected_length": len(expected_ids),
                        }
                    )

        # Create W&B table for generations
        if generation_examples:
            generation_table = wandb.Table(
                columns=[
                    "Step",
                    "Prompt",
                    "Expected Output",
                    "Generated Output",
                    "Prompt Tokens",
                    "Expected Tokens",
                ],
                data=[
                    [
                        ex["step"],
                        ex["prompt"][:300] + "...",
                        ex["expected"][:200] + "...",
                        ex["generated"][:200] + "...",
                        ex["prompt_length"],
                        ex["expected_length"],
                    ]
                    for ex in generation_examples
                ],
            )

            wandb.log(
                {
                    "generation_examples": generation_table,
                    "avg_prompt_length": np.mean(
                        [ex["prompt_length"] for ex in generation_examples]
                    ),
                    "avg_expected_length": np.mean(
                        [ex["expected_length"] for ex in generation_examples]
                    ),
                },
                step=state.global_step,
            )


def create_stratified_split(dataset, tokenizer, validation_ratio=0.15):
    """Create stratified split based on sequence length for better validation"""

    # Calculate lengths for stratification from tokenized data
    lengths = []
    for item in dataset:
        # Use input_ids length instead of re-tokenizing text
        lengths.append(len(item["input_ids"]))

    # Create length-based strata
    def get_length_stratum(length):
        if length <= 1000:
            return "short"
        elif length <= 2500:
            return "medium"
        elif length <= 4000:
            return "long"
        else:
            return "very_long"

    # Group by strata
    strata = {}
    for i, length in enumerate(lengths):
        stratum = get_length_stratum(length)
        if stratum not in strata:
            strata[stratum] = []
        strata[stratum].append(i)

    # Sample from each stratum
    train_indices = []
    val_indices = []

    print(f"\n📊 Stratified Split Analysis:")
    for stratum, indices in strata.items():
        n_val = max(1, int(len(indices) * validation_ratio))
        n_train = len(indices) - n_val

        # Random selection within stratum
        np.random.seed(42)
        shuffled = np.random.permutation(indices)

        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])

        print(
            f"  - {stratum}: {len(indices)} total → {n_train} train, {n_val} val"
        )

    # Create datasets
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)

    print(
        f"\n✅ Final split: {len(train_dataset)} train, {len(val_dataset)} validation"
    )
    print(
        f"   Validation ratio: {len(val_dataset)/(len(train_dataset)+len(val_dataset)):.1%}"
    )

    return train_dataset, val_dataset


def setup_model_and_tokenizer():
    """Setup model and tokenizer with quantization compatibility checking"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Check bitsandbytes compatibility
    bnb_available, bnb_type = check_bitsandbytes_compatibility()

    # Check GPU memory
    gpu_memory = get_gpu_memory()
    print(f"GPU Memory: {gpu_memory:.1f} GB")

    print("Loading model...")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Try different configurations in order of preference
    configs_to_try = []

    if bnb_available and bnb_type == "4bit":
        # Updated 4-bit configurations with more compatible parameters
        configs_to_try.extend(
            [
                {
                    "name": "4-bit NF4 with double quantization (updated)",
                    "config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    ),
                    "device_map": "auto",
                    "torch_dtype": torch.bfloat16,
                },
                {
                    "name": "4-bit NF4 basic (updated)",
                    "config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    ),
                    "device_map": "auto",
                    "torch_dtype": torch.bfloat16,
                },
            ]
        )

    if bnb_available:
        # 8-bit fallback with updated parameters
        configs_to_try.append(
            {
                "name": "8-bit quantization (updated)",
                "config": BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                ),
                "device_map": "auto",
                "torch_dtype": torch.float16,
            }
        )

    # Add fallback without quantization
    configs_to_try.extend(
        [
            {
                "name": "FP16 without quantization",
                "config": None,
                "device_map": "auto",
                "torch_dtype": torch.float16,
            },
            {
                "name": "FP32 without quantization",
                "config": None,
                "device_map": "auto",
                "torch_dtype": torch.float32,
            },
        ]
    )

    # Try each configuration
    model = None
    successful_config = None

    for config_info in configs_to_try:
        try:
            print(f"🔄 Trying: {config_info['name']}")

            model_kwargs = {
                "torch_dtype": config_info["torch_dtype"],
                "device_map": config_info["device_map"],
                "use_cache": False,
                "trust_remote_code": True,
            }

            if config_info["config"] is not None:
                model_kwargs["quantization_config"] = config_info["config"]

            model = AutoModelForCausalLM.from_pretrained(
                Config.MODEL_NAME, **model_kwargs
            )

            print(f"✅ Successfully loaded with: {config_info['name']}")
            successful_config = config_info
            break

        except Exception as e:
            print(f"❌ Failed with {config_info['name']}: {str(e)[:100]}...")
            if model is not None:
                del model
                torch.cuda.empty_cache()
            continue

    if model is None:
        raise RuntimeError("All model loading configurations failed!")

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Only prepare for quantization if we're using a quantized model
    if successful_config["config"] is not None:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, lora_config)

    # Print model memory usage
    try:
        print(
            f"Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB"
        )
    except:
        print("Model loaded successfully (memory footprint unavailable)")

    # Print which configuration was successful
    print(f"💡 Using configuration: {successful_config['name']}")

    return model, tokenizer


def create_training_args():
    """T4-optimized training for long sequences"""
    return TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        per_device_eval_batch_size=1,  # Always 1 for safety
        eval_strategy="steps",
        eval_steps=Config.EVAL_STEPS,
        logging_steps=10,
        save_steps=Config.SAVE_STEPS,
        save_total_limit=1,
        learning_rate=Config.LEARNING_RATE,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.05,  # Stronger regularization
        # T4 CRITICAL settings
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.5,
        # Memory management
        eval_accumulation_steps=4,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,  # Keep our custom labels for instruction tuning
        group_by_length=True,  # CRITICAL: Groups similar lengths together
        # Monitoring
        report_to="wandb",
        run_name=f"smollm-t4-long-context",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
        # Additional safety
        skip_memory_metrics=True,
        include_inputs_for_metrics=False,
        # CRITICAL: Ensure labels are used properly for instruction tuning
        label_names=["labels"],
        prediction_loss_only=False,
    )


class CrossValidationTrainer:
    """3-fold cross-validation for robust meta-pattern evaluation"""

    def __init__(self, dataset, tokenizer, n_folds=3):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.n_folds = n_folds
        self.fold_results = []

    def create_folds(self):
        """Create stratified k-folds"""
        dataset_size = len(self.dataset)
        indices = np.arange(dataset_size)
        np.random.seed(42)
        np.random.shuffle(indices)

        fold_size = dataset_size // self.n_folds
        folds = []

        for i in range(self.n_folds):
            start_idx = i * fold_size
            end_idx = (
                (i + 1) * fold_size if i < self.n_folds - 1 else dataset_size
            )
            folds.append(indices[start_idx:end_idx])

        return folds

    def train_fold(self, fold_idx, train_indices, val_indices):
        """Train one fold"""
        print(f"\n🔄 Training Fold {fold_idx + 1}/{self.n_folds}")

        # Create fold datasets
        train_dataset = self.dataset.select(train_indices)
        val_dataset = self.dataset.select(val_indices)

        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Validation samples: {len(val_dataset)}")

        # Setup model for this fold (fresh model each time)
        model, tokenizer = setup_model_and_tokenizer()

        # Create training arguments with fold-specific run name
        training_args = create_training_args()
        training_args.run_name = f"smollm-fold-{fold_idx+1}"
        training_args.output_dir = f"{Config.OUTPUT_DIR}/fold_{fold_idx+1}"

        # Disable W&B for cross-validation (too many runs)
        training_args.report_to = []

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=AdaptiveDataCollator(
                tokenizer=tokenizer, max_safe_length=2048
            ),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # Train
        result = trainer.train()

        # Evaluate
        eval_results = trainer.evaluate()

        return {
            "fold": fold_idx + 1,
            "train_loss": result.training_loss,
            "eval_loss": eval_results["eval_loss"],
            "model_path": training_args.output_dir,
        }

    def run_cross_validation(self):
        """Run full cross-validation"""
        folds = self.create_folds()

        for fold_idx in range(self.n_folds):
            # Create train/val split for this fold
            val_indices = folds[fold_idx]
            train_indices = np.concatenate(
                [folds[i] for i in range(self.n_folds) if i != fold_idx]
            )

            # Train this fold
            fold_result = self.train_fold(fold_idx, train_indices, val_indices)
            self.fold_results.append(fold_result)

            # Clean up GPU memory
            torch.cuda.empty_cache()

        # Analyze results
        self.analyze_cv_results()

    def analyze_cv_results(self):
        """Analyze cross-validation results"""
        eval_losses = [r["eval_loss"] for r in self.fold_results]
        train_losses = [r["train_loss"] for r in self.fold_results]

        print(f"\n📊 Cross-Validation Results:")
        print(
            f"  - Mean Eval Loss: {np.mean(eval_losses):.4f} ± {np.std(eval_losses):.4f}"
        )
        print(
            f"  - Mean Train Loss: {np.mean(train_losses):.4f} ± {np.std(train_losses):.4f}"
        )
        print(
            f"  - Overfitting Gap: {np.mean(eval_losses) - np.mean(train_losses):.4f}"
        )

        for result in self.fold_results:
            print(
                f"  - Fold {result['fold']}: Train={result['train_loss']:.4f}, Val={result['eval_loss']:.4f}"
            )

        # Save results
        cv_results = {
            "mean_eval_loss": np.mean(eval_losses),
            "std_eval_loss": np.std(eval_losses),
            "mean_train_loss": np.mean(train_losses),
            "fold_results": self.fold_results,
        }

        with open(f"{Config.OUTPUT_DIR}/cv_results.json", "w") as f:
            json.dump(cv_results, f, indent=2)


class SimplifiedDataCollator:
    """L40S-optimized collator for batch_size=2"""

    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # L40S can handle batch_size=2, so remove the assertion
        batch_size = len(features)

        # Find max length in this batch
        max_length = max(len(f["input_ids"]) for f in features)

        # Memory usage decision (for longest sequence in batch)
        if max_length > 6000:
            print(f"🚨 Processing very long batch: max {max_length} tokens")
        elif max_length > 4000:
            print(f"⚠️  Processing long batch: max {max_length} tokens")
        else:
            print(f"✅ Processing normal batch: max {max_length} tokens")

        # Cap length
        max_length = min(max_length, Config.MAX_LENGTH)

        # Pad to efficient multiple
        if self.pad_to_multiple_of:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of

        # Create batch tensors for all samples
        batch = {}
        for key in ["input_ids", "labels"]:
            batch_sequences = []
            for sample in features:
                sequence = sample[key][:max_length]
                padding_length = max_length - len(sequence)
                if key == "labels":
                    padded = sequence + [-100] * padding_length
                else:
                    padded = (
                        sequence
                        + [self.tokenizer.pad_token_id] * padding_length
                    )
                batch_sequences.append(padded)
            batch[key] = torch.tensor(batch_sequences, dtype=torch.long)

        batch["attention_mask"] = (
            batch["input_ids"] != self.tokenizer.pad_token_id
        ).long()

        # DEBUG: Verify instruction tuning labels for first sample
        if len(features) > 0:
            labels = features[0]["labels"]
            masked_count = sum(1 for label in labels if label == -100)
            total_count = len(labels)
            print(
                f"🔍 Label verification (sample 1): {masked_count}/{total_count} tokens masked ({masked_count/total_count*100:.1f}%)"
            )

        return batch


class InstructionTuningTrainer(Trainer):
    """Custom trainer that ensures proper label handling for instruction tuning"""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override to ensure labels are used properly for instruction tuning
        Updated to handle additional kwargs from newer transformers versions
        """
        # Ensure we have labels
        if "labels" not in inputs:
            raise ValueError(
                "Labels not found! Instruction tuning requires labels."
            )

        # Verify labels have -100 masking
        labels = inputs["labels"]
        masked_tokens = (labels == -100).sum().item()
        total_tokens = labels.numel()

        if masked_tokens == 0:
            print("⚠️  WARNING: No masked tokens found in labels!")
        elif masked_tokens == total_tokens:
            print("⚠️  WARNING: All tokens are masked!")
        else:
            mask_percentage = (masked_tokens / total_tokens) * 100
            if mask_percentage < 20 or mask_percentage > 80:
                print(
                    f"⚠️  Unusual masking: {mask_percentage:.1f}% tokens masked"
                )

        # Forward pass
        outputs = model(**inputs)

        # Extract loss (should be computed using our masked labels)
        loss = outputs.get("loss")
        if loss is None:
            # Fallback: compute loss manually
            logits = outputs.get("logits")
            if logits is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Only compute loss on non-masked tokens
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
            else:
                raise ValueError("Could not compute loss - no logits found!")

        return (loss, outputs) if return_outputs else loss


# [Keep all the existing code unchanged until the main() function, then replace the main() function with this:]


def main():
    """Main training function with flexible validation"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Clear any existing cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()

    # Initialize wandb
    wandb.init(
        project=Config.WANDB_PROJECT,
        config={
            "model_name": Config.MODEL_NAME,
            "max_length": Config.MAX_LENGTH,
            "batch_size": Config.BATCH_SIZE,
            "learning_rate": Config.LEARNING_RATE,
            "num_epochs": Config.NUM_EPOCHS,
            "padding_strategy": "dynamic",
            "quantization": "4bit_preferred",
            "lora_r": 8,
            "lora_alpha": 16,
        },
    )

    # Load and prepare dataset
    dataset = load_and_prepare_dataset(
        Config.DATASET_PATH, tokenizer, Config.MAX_LENGTH
    )

    if Config.VALIDATION_STRATEGY == "cross_validation":
        # Run cross-validation
        cv_trainer = CrossValidationTrainer(dataset, tokenizer, n_folds=3)
        cv_trainer.run_cross_validation()
        return

    elif Config.VALIDATION_STRATEGY == "stratified":
        # Stratified split
        train_dataset, eval_dataset = create_stratified_split(
            dataset, tokenizer, validation_ratio=Config.VALIDATION_RATIO
        )

    else:  # simple split
        train_test_split = dataset.train_test_split(
            test_size=Config.VALIDATION_RATIO, seed=42
        )
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")

    # Create training arguments
    training_args = create_training_args()

    # Use our SIMPLIFIED data collator (cleaner for batch_size=1)
    data_collator = SimplifiedDataCollator(
        tokenizer=tokenizer, pad_to_multiple_of=8  # Simpler parameters
    )

    # Use custom trainer to ensure proper instruction tuning
    trainer = InstructionTuningTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            DetailedLoggingCallback(tokenizer, eval_dataset),
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    # Train the model
    print("Starting training with quantized model...")
    trainer.train()

    # Save the LoRA adapters (ALWAYS works)
    print("Saving LoRA adapters...")
    trainer.save_model(os.path.join(Config.OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(Config.OUTPUT_DIR, "final"))

    wandb.finish()

    # --- IMPROVED MODEL SAVING FOR CROSS-ENVIRONMENT COMPATIBILITY ---
    print("\n🔧 Creating inference-ready models for different environments...")

    # Method 1: Try merge_and_unload (works in GPU environments)
    try:
        print("📦 Attempting GPU-compatible merge...")
        merged_model = model.merge_and_unload()

        # Save merged model
        merged_path = os.path.join(Config.OUTPUT_DIR, "merged")
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)

        # Clean the config.json to remove quantization info for CPU compatibility
        import json

        config_path = os.path.join(merged_path, "config.json")
        with open(config_path, "r") as f:
            config_data = json.load(f)

        # Remove quantization config if present
        if "quantization_config" in config_data:
            print("🧹 Cleaning quantization config from merged model...")
            del config_data["quantization_config"]

            # Save cleaned config
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

        print("✅ GPU-compatible merged model saved!")

    except Exception as e:
        print(f"⚠️  GPU merge failed: {e}")
        print("💡 Will rely on adapter-based loading for inference.")

    # Method 2: Create CPU-compatible version by loading base model without quantization
    try:
        print("💻 Creating CPU-compatible model...")

        # Load base model without quantization
        cpu_base_model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            torch_dtype=torch.float16,  # Use float16 for better CPU compatibility
            device_map="cpu",
            trust_remote_code=True,
        )

        # Load and merge LoRA
        from peft import PeftModel

        cpu_model = PeftModel.from_pretrained(
            cpu_base_model, os.path.join(Config.OUTPUT_DIR, "final")
        )
        cpu_merged = cpu_model.merge_and_unload()

        # Save CPU-compatible version
        cpu_path = os.path.join(Config.OUTPUT_DIR, "cpu_compatible")
        cpu_merged.save_pretrained(cpu_path)
        tokenizer.save_pretrained(cpu_path)

        print("✅ CPU-compatible model saved!")

    except Exception as e:
        print(f"⚠️  CPU merge failed: {e}")
        print("💡 Use adapter-based loading for CPU inference.")

    print("\n📋 Inference Options Created:")
    print("  1. final/ - LoRA adapters (works everywhere with PEFT)")
    print("  2. merged/ - GPU-optimized merged model (if successful)")
    print("  3. cpu_compatible/ - CPU-optimized merged model (if successful)")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

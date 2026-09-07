import os
import sys
import json
import random
import logging
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
    set_seed,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from accelerate import Accelerator

# --- GPU CONFIGURATION ---
def detect_gpu_config() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. This pipeline requires an NVIDIA GPU.")

    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    cc = torch.cuda.get_device_capability(0)  # (major, minor)

    is_blackwell = cc[0] >= 12
    if is_blackwell:
        assert torch.version.cuda is not None and tuple(int(x) for x in torch.version.cuda.split(".")[:2]) >= (12, 8), \
            f"Blackwell GPU detected but PyTorch was built against CUDA {torch.version.cuda}. Need CUDA 12.8+."

    # 32 GB tier — RTX 5090
    if 28 <= vram_gb < 35:
        return {
            "lora_rank": 64, "batch": 4, "grad_accum": 4,
            "load_in_4bit": False, "dtype": torch.bfloat16,
            "attn_impl": "sdpa",
            "grad_checkpointing": True,
        }
    # ... (other tiers omitted for brevity as per user hardware specification)
    # Default to 32GB config as target is RTX 5090
    return {
        "lora_rank": 64, "batch": 4, "grad_accum": 4,
        "load_in_4bit": False, "dtype": torch.bfloat16,
        "attn_impl": "sdpa",
        "grad_checkpointing": True,
    }

GPU_CFG = detect_gpu_config()

# --- CONSTANTS ---
MODEL_ID    = "Qwen/Qwen2.5-VL-7B-Instruct"
PROJECT_ROOT = Path(__file__).parent
DATASET_PATH = PROJECT_ROOT / "dataset"
OUTPUT_DIR  = PROJECT_ROOT / "output"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
BEST_DIR = CHECKPOINTS_DIR / "best"
INTERRUPTED_DIR = OUTPUT_DIR / "interrupted"
HISTORY_FILE = OUTPUT_DIR / "history.json"
METRICS_FILE = OUTPUT_DIR / "training_metrics.png"

EPOCHS      = 10
LR          = 1e-4
SAVE_EVERY  = 200
EARLY_STOP  = 8
MAX_LEN     = 2048
SEED        = 42

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- DATASET ---
from qwen_vl_utils import process_vision_info

class OCRDataset(Dataset):
    def __init__(self, jsonl_path: Path, max_len: int, base_dir: Path):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.max_len = max_len
        self.base_dir = base_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        system = item["system"]
        question = item["question"]
        ocr_text = item["ocr_text"]
        answer = item["answer"]
        
        img_rel_path = item["image"]
        img_path = self.base_dir / img_rel_path
        img_path_str = f"file://{img_path}"

        # Feed the image, and still include the OCR text to help the model
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [
                # Restrict max_pixels to ~800k to balance VRAM and OCR resolution
                {"type": "image", "image": img_path_str, "max_pixels": 800000},
                {"type": "text", "text": f"{question}\n{ocr_text}"}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]

        return {"messages": messages}

def get_collate_fn(processor, max_len):
    def collate_fn(batch_items):
        batch_messages = [item["messages"] for item in batch_items]
        
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in batch_messages]
        image_inputs, video_inputs = process_vision_info(batch_messages)
        
        batch = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        
        labels = batch["input_ids"].clone()
        assistant_tag = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        tag_len = len(assistant_tag)
        
        for i in range(len(batch_messages)):
            input_ids = batch["input_ids"][i].tolist()
            start_idx = -1
            for j in range(len(input_ids) - tag_len + 1):
                if input_ids[j:j+tag_len] == assistant_tag:
                    start_idx = j + tag_len
                    break
                    
            if start_idx != -1:
                labels[i, :start_idx] = -100
            
            labels[i, batch["attention_mask"][i] == 0] = -100
        
        batch["labels"] = labels
        return batch
    return collate_fn

# --- TRAINING FUNCTIONS ---
def save_metrics(history):
    if not history: return
    steps = [h["step"] for h in history]
    train_losses = [h.get("train_loss") for h in history]
    eval_losses = [h.get("eval_loss") for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label="Train Loss")
    plt.plot(steps, [l for l in eval_losses if l is not None], label="Eval Loss", marker='o')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Fine-tuning Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig(METRICS_FILE)
    plt.close()

def train():
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_PATH))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=GPU_CFG["batch"])
    parser.add_argument("--lora-rank", type=int, default=GPU_CFG["lora_rank"])
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit quantization (QLoRA)")
    args_train = parser.parse_args()

    use_4bit = args_train.load_in_4bit
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    quantization_config = None
    if use_4bit:
        logger.info("Enabling 4-bit quantization (NF4)...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    load_kwargs = {
        "torch_dtype": GPU_CFG["dtype"],
        "attn_implementation": GPU_CFG["attn_impl"],
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if use_4bit:
        load_kwargs["quantization_config"] = quantization_config

    logger.info(f"Loading model: {MODEL_ID} (4-bit={use_4bit})")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, **load_kwargs)

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Freeze Vision Tower
    if hasattr(model, "visual"):
        logger.info("Freezing vision tower parameters.")
        for param in model.visual.parameters():
            param.requires_grad = False
    
    # LoRA Configuration
    lora_config = LoraConfig(
        r=GPU_CFG["lora_rank"],
        lora_alpha=GPU_CFG["lora_rank"] * 2,
        lora_dropout=0.05,
        use_rslora=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    if GPU_CFG["grad_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    actual_dataset_path = Path(args_train.dataset_dir)
    # The image paths in the jsonl are relative to the 'loan' root directory
    loan_root_dir = PROJECT_ROOT.parent
    train_ds = OCRDataset(actual_dataset_path / "train.jsonl", MAX_LEN, loan_root_dir)
    eval_ds = OCRDataset(actual_dataset_path / "val.jsonl", MAX_LEN, loan_root_dir)

    collate_fn = get_collate_fn(processor, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=args_train.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_ds, batch_size=args_train.batch_size, shuffle=False, collate_fn=collate_fn)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args_train.lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    grad_accum = GPU_CFG["grad_accum"]
    num_training_steps = len(train_loader) * args_train.epochs // grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps
    )

    # --- KEY FIX: Do NOT pass model through accelerator.prepare() ---
    # With device_map="auto", the model is already on GPU. Accelerator re-sharding causes a deadlock.
    logger.info("Skipping accelerator.prepare(model) — model already on device.")
    accelerator = Accelerator(gradient_accumulation_steps=grad_accum)
    optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        optimizer, train_loader, eval_loader, scheduler
    )

    # Loss Masking Verification (Batch 1)
    logger.info("Verifying loss masking on first batch...")
    first_batch = next(iter(train_loader))
    # Move batch to model's device manually
    first_batch = {k: v.to(device) for k, v in first_batch.items()}
    labels = first_batch["labels"]
    masked_count = (labels == -100).sum().item()
    total_tokens = labels.numel()
    logger.info(f"First batch masking check: {masked_count}/{total_tokens} tokens masked.")
    if masked_count == 0:
        raise RuntimeError("CRITICAL: Loss masking failed! No tokens masked in labels.")

    # --- RESUME FROM CHECKPOINT ---
    resume_step = 0
    start_epoch = 0
    history = []
    best_eval_loss = float("inf")
    no_improvement_count = 0
    global_step = 0

    # Check for existing checkpoints to resume from
    resume_dir = None
    if INTERRUPTED_DIR.exists():
        resume_dir = INTERRUPTED_DIR
        logger.info(f"Found interrupted checkpoint at {INTERRUPTED_DIR}")
    else:
        # Find latest step_N checkpoint
        existing = sorted(CHECKPOINTS_DIR.glob("step_*"), key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0)
        if existing:
            resume_dir = existing[-1]
            logger.info(f"Found checkpoint at {resume_dir}")

    if resume_dir and resume_dir.exists():
        logger.info(f"Resuming training from {resume_dir}...")
        if use_4bit:
            # For 4-bit, load LoRA weights manually instead of using accelerator.load_state
            from peft import set_peft_model_state_dict
            import safetensors.torch
            adapter_path = resume_dir / "model" / "adapter_model.safetensors"
            if not adapter_path.exists():
                adapter_path = resume_dir / "model" / "adapter_model.bin"
            if adapter_path.exists():
                if str(adapter_path).endswith(".safetensors"):
                    state_dict = safetensors.torch.load_file(str(adapter_path))
                else:
                    state_dict = torch.load(str(adapter_path), map_location="cpu")
                set_peft_model_state_dict(model, state_dict)
                logger.info("Loaded LoRA adapter weights for 4-bit resume.")
            else:
                logger.warning(f"No adapter weights found at {resume_dir}, starting fresh.")
        else:
            if (resume_dir / "model" / "adapter_model.safetensors").exists() or (resume_dir / "model" / "adapter_model.bin").exists():
                from peft import set_peft_model_state_dict
                import safetensors.torch
                adapter_path = resume_dir / "model" / "adapter_model.safetensors"
                if not adapter_path.exists():
                    adapter_path = resume_dir / "model" / "adapter_model.bin"
                if adapter_path.exists():
                    if str(adapter_path).endswith(".safetensors"):
                        state_dict = safetensors.torch.load_file(str(adapter_path))
                    else:
                        state_dict = torch.load(str(adapter_path), map_location="cpu")
                    set_peft_model_state_dict(accelerator.unwrap_model(model), state_dict)
                    logger.info("Loaded LoRA adapter weights for manual resume.")
                if (resume_dir / "optimizer.pt").exists():
                    optimizer.load_state_dict(torch.load(resume_dir / "optimizer.pt", map_location="cpu"))
                if (resume_dir / "scheduler.pt").exists():
                    scheduler.load_state_dict(torch.load(resume_dir / "scheduler.pt", map_location="cpu"))
            else:
                accelerator.load_state(resume_dir)

        # Load resume metadata
        meta_path = resume_dir / "train_meta.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            global_step = meta.get("global_step", 0)
            start_epoch = meta.get("epoch", 0)
            best_eval_loss = meta.get("best_eval_loss", float("inf"))
            no_improvement_count = meta.get("no_improvement_count", 0)
            logger.info(f"Resumed: global_step={global_step}, epoch={start_epoch}, best_eval_loss={best_eval_loss:.4f}")
        else:
            # Infer from checkpoint name
            try:
                global_step = int(resume_dir.name.split("_")[1])
                steps_per_epoch = len(train_loader)
                start_epoch = global_step // steps_per_epoch
                logger.info(f"Inferred: global_step={global_step}, start_epoch={start_epoch}")
            except:
                logger.warning("Could not infer step from checkpoint name, starting from 0")

        # Load history if exists
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)

        # Clean up interrupted dir after loading
        if resume_dir == INTERRUPTED_DIR:
            import shutil
            shutil.rmtree(INTERRUPTED_DIR)
            logger.info("Cleaned up interrupted checkpoint after loading.")
    else:
        logger.info("No checkpoint found. Starting fresh training.")

    # --- Scaler for mixed precision in 4-bit mode ---
    scaler = torch.amp.GradScaler("cuda") if use_4bit else None

    try:
        for epoch in range(start_epoch, args_train.epochs):
            model.train()
            total_train_loss = 0
            
            for step, batch in enumerate(train_loader):
                # Skip steps already completed (for resume)
                if epoch == start_epoch and step < (global_step % len(train_loader)):
                    continue

                # Move batch to device (accelerator doesn't handle this when model is not prepared)
                batch = {k: v.to(device) for k, v in batch.items()}

                if use_4bit:
                    # Manual gradient accumulation for 4-bit
                    outputs = model(**batch)
                    loss = outputs.loss / grad_accum
                    loss.backward()
                    
                    if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    actual_loss = loss.item() * grad_accum
                else:
                    with accelerator.accumulate(model):
                        outputs = model(**batch)
                        loss = outputs.loss
                        accelerator.backward(loss)
                        
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    actual_loss = loss.item()

                global_step += 1
                total_train_loss += actual_loss

                # Per-step logging for real-time monitoring
                avg_so_far = total_train_loss / (step + 1)
                lr_now = scheduler.get_last_lr()[0]
                logger.info(f"Epoch {epoch+1}/{args_train.epochs} | Step {global_step} | Loss: {actual_loss:.4f} | Avg: {avg_so_far:.4f} | LR: {lr_now:.2e}")

                if global_step % SAVE_EVERY == 0:
                    # Evaluation
                    model.eval()
                    eval_loss = 0
                    with torch.no_grad():
                        for eval_batch in eval_loader:
                            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                            eval_outputs = model(**eval_batch)
                            eval_loss += eval_outputs.loss.item()
                    
                    avg_eval_loss = eval_loss / len(eval_loader)
                    avg_train_loss = total_train_loss / (step + 1)
                    
                    logger.info(f"Step {global_step} | Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f}")
                    
                    history_item = {
                        "step": global_step,
                        "train_loss": avg_train_loss,
                        "eval_loss": avg_eval_loss
                    }
                    history.append(history_item)
                    
                    with open(HISTORY_FILE, "w") as f:
                        json.dump(history, f)
                    save_metrics(history)

                    # Checkpointing — save LoRA weights directly for 4-bit
                    snapshot_path = CHECKPOINTS_DIR / f"step_{global_step}"
                    snapshot_path.mkdir(parents=True, exist_ok=True)
                    if use_4bit:
                        model.save_pretrained(snapshot_path / "model")
                    else:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(snapshot_path / "model")
                        torch.save(optimizer.state_dict(), snapshot_path / "optimizer.pt")
                        torch.save(scheduler.state_dict(), snapshot_path / "scheduler.pt")
                    # Save training metadata for resume
                    meta = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "best_eval_loss": best_eval_loss,
                        "no_improvement_count": no_improvement_count,
                    }
                    with open(snapshot_path / "train_meta.json", "w") as f:
                        json.dump(meta, f)
                    
                    # Keep only the latest 3 checkpoints to save disk space
                    import shutil
                    MAX_CHECKPOINTS = 3
                    existing_checkpoints = sorted(CHECKPOINTS_DIR.glob("step_*"), key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0)
                    if len(existing_checkpoints) > MAX_CHECKPOINTS:
                        for old_ckpt in existing_checkpoints[:-MAX_CHECKPOINTS]:
                            logger.info(f"Removing old checkpoint to save space: {old_ckpt.name}")
                            shutil.rmtree(old_ckpt, ignore_errors=True)
                    
                    if avg_eval_loss < best_eval_loss:
                        best_eval_loss = avg_eval_loss
                        logger.info(f"New best model found at step {global_step}!")
                        if use_4bit:
                            model.save_pretrained(BEST_DIR)
                        else:
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(BEST_DIR)
                        processor.save_pretrained(BEST_DIR)
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        if no_improvement_count >= EARLY_STOP:
                            logger.info("Early stopping triggered.")
                            return
                    
                    model.train()

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving state...")
        INTERRUPTED_DIR.mkdir(parents=True, exist_ok=True)
        if use_4bit:
            model.save_pretrained(INTERRUPTED_DIR / "model")
        else:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(INTERRUPTED_DIR / "model")
            torch.save(optimizer.state_dict(), INTERRUPTED_DIR / "optimizer.pt")
            torch.save(scheduler.state_dict(), INTERRUPTED_DIR / "scheduler.pt")
        meta = {"global_step": global_step, "epoch": epoch, "best_eval_loss": best_eval_loss, "no_improvement_count": no_improvement_count}
        with open(INTERRUPTED_DIR / "train_meta.json", "w") as f:
            json.dump(meta, f)
        logger.info(f"Saved interrupt checkpoint at step {global_step}. Run train.py again to resume.")
        raise
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        INTERRUPTED_DIR.mkdir(parents=True, exist_ok=True)
        if use_4bit:
            model.save_pretrained(INTERRUPTED_DIR / "model")
        else:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(INTERRUPTED_DIR / "model")
            torch.save(optimizer.state_dict(), INTERRUPTED_DIR / "optimizer.pt")
            torch.save(scheduler.state_dict(), INTERRUPTED_DIR / "scheduler.pt")
        meta = {"global_step": global_step, "epoch": epoch, "best_eval_loss": best_eval_loss, "no_improvement_count": no_improvement_count}
        with open(INTERRUPTED_DIR / "train_meta.json", "w") as f:
            json.dump(meta, f)
        raise

    logger.info("Training complete.")

if __name__ == "__main__":
    train()

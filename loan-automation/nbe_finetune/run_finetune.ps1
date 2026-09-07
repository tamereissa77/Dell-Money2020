# PowerShell runner for Qwen2.5‑VL fine‑tuning pipeline
param(
    [string]$DataDir = "d:\projects\loan\nbe_finetune\data",
    [int]$Epochs = 5,
    [int]$BatchSize = 4,
    [int]$LoraRank = 64,
    [float]$LR = 5e-5
)

$pyExe = "D:\projects\loan\.venv\Scripts\python.exe"

$BatchSize = 2  # Reduced from 4 to prevent Windows WDDM VRAM swapping to system RAM
$DatasetDir = "D:\projects\loan\data\vl_dataset_v2"

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

Write-Host " Starting bfloat16 training at $DatasetDir (batch=$BatchSize, lr=$LR)..."
& $pyExe "${PSScriptRoot}\train.py" --dataset-dir $DatasetDir --epochs $Epochs --batch-size $BatchSize --lora-rank $LoraRank --lr $LR

Write-Host " Starting evaluation..."
& $pyExe "${PSScriptRoot}\evaluate.py"

Write-Host ' Pipeline finished – best LoRA adapter saved under nbe_finetune\output\checkpoints\best'

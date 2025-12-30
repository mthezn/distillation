# 🚀 Training Guide - Knowledge Distillation Pipeline

Complete guide for training the surgical tool segmentation model using the unified training script.

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Training Pipeline](#training-pipeline)
- [Configuration](#configuration)
- [Command Line Options](#command-line-options)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Quick Start

### 1. **Setup Environment**

```bash
# Install dependencies
pip install -r requirements.txt

# Set Weights & Biases API key
export WANDB_API_KEY="your_api_key_here"

# Create directory structure
mkdir -p configs checkpoints/{encoder,decoder,finetune}
```

### 2. **Prepare Configuration Files**

Copy the example configs from `configs/examples/` to `configs/`:

```bash
cp configs/examples/encoder_distillation.yaml configs/
cp configs/examples/decoder_distillation.yaml configs/
cp configs/examples/finetune.yaml configs/
```

**⚠️ Important**: Edit each config file and update the dataset paths!

### 3. **Run Complete Training Pipeline**

```bash
# Stage 1: Encoder Distillation (~4-6 hours on A100)
python train.py --mode encoder --config configs/encoder_distillation.yaml

# Stage 2: Decoder Distillation (~3-4 hours)
python train.py --mode decoder --config configs/decoder_distillation.yaml \
    --checkpoint checkpoints/encoder/best_encoder.pth

# Stage 3: Fine-tuning (~4-5 hours)
python train.py --mode finetune --config configs/finetune.yaml \
    --checkpoint checkpoints/decoder/best_decoder.pth
```

---

## 🔄 Training Pipeline

### **Overview**

The training follows a **3-stage distillation strategy**:

```
Stage 1: Encoder Distillation
    ├─ Teacher: SAM ViT-H encoder (frozen)
    ├─ Student: CMT encoder (trainable)
    ├─ Loss: MSE (feature matching)
    └─ Output: Aligned encoder weights

Stage 2: Decoder Distillation
    ├─ Teacher: SAM full model (frozen)
    ├─ Student: CMT encoder (frozen) + CNN decoder (trainable)
    ├─ Loss: BCE + Dice (mask matching)
    └─ Output: Complete distilled model

Stage 3: Supervised Fine-tuning
    ├─ Student: Full model (trainable)
    ├─ Datasets: MICCAI + CholecSeg8k
    ├─ Loss: Dice Loss
    └─ Output: Final optimized model
```

### **Stage 1: Encoder Distillation**

**Goal**: Align student encoder features with teacher encoder

**What happens**:
- Student encoder learns to extract features similar to SAM
- Only encoder weights are updated
- Decoder remains uninitialized

**Configuration**:
```yaml
mode: encoder
student_checkpoint: null  # Start from scratch
epochs: 50
lr: 0.0001
weight_decay: 0.1
```

**Expected results**:
- Training loss: ~0.02-0.05
- Validation loss: ~0.03-0.06
- Duration: 4-6 hours (A100 GPU)

**Run**:
```bash
python train.py --mode encoder --config configs/encoder_distillation.yaml
```

---

### **Stage 2: Decoder Distillation**

**Goal**: Train decoder to produce masks similar to SAM

**What happens**:
- Encoder is frozen (uses weights from Stage 1)
- Decoder learns to generate binary segmentation masks
- Loss combines Dice and BCE for mask quality

**Configuration**:
```yaml
mode: decoder
student_checkpoint: "checkpoints/encoder/best_encoder.pth"  # Load from Stage 1
epochs: 30
lr: 0.0001
weight_decay: 0.0001
```

**Expected results**:
- Training loss: ~0.10-0.15
- Validation IoU: ~0.85-0.88
- Duration: 3-4 hours (A100 GPU)

**Run**:
```bash
python train.py --mode decoder --config configs/decoder_distillation.yaml \
    --checkpoint checkpoints/encoder/enc_XYZ.pth
```

---

### **Stage 3: Supervised Fine-tuning**

**Goal**: Refine end-to-end model on combined datasets

**What happens**:
- Full model is trainable (encoder + decoder)
- Adds CholecSeg8k dataset for generalization
- Uses Dice loss for optimization
- Learns surgical domain specifics

**Configuration**:
```yaml
mode: finetune
student_checkpoint: "checkpoints/decoder/best_decoder.pth"  # Load from Stage 2
use_cholec: true  # Add CholecSeg8k
epochs: 30
lr: 0.0001
weight_decay: 0.000001
scheduler_type: plateau
```

**Expected results**:
- Validation IoU: ~0.89-0.92
- Validation Dice: ~0.93-0.95
- Duration: 4-5 hours (A100 GPU)

**Run**:
```bash
python train.py --mode finetune --config configs/finetune.yaml \
    --checkpoint checkpoints/decoder/dec_ABC.pth
```

---

## ⚙️ Configuration

### **Config File Structure**

```yaml
# Training mode
mode: encoder  # 'encoder', 'decoder', or 'finetune'

# Model paths
teacher_checkpoint: "checkpoints/sam_vit_h_4b8939.pth"
student_checkpoint: null  # or path to previous checkpoint

# Training parameters
epochs: 50
batch_size: 2
lr: 0.0001
weight_decay: 0.1
optimizer: adamw
scheduler_type: cosine  # or 'plateau'

# Data configuration
use_miccai: true
use_cholec: false
image_size: 1024

# Augmentation
use_augmentation: true
horizontal_flip: 0.5
vertical_flip: 0.5
rotation_limit: 45

# Paths
miccai_train_images: [...]
miccai_train_masks: [...]
miccai_val_images: [...]
miccai_val_masks: [...]

# Logging
wandb_project: "my-project"
checkpoint_dir: "checkpoints"
```

### **Key Parameters**

| Parameter | Stage 1 | Stage 2 | Stage 3 | Description |
|-----------|---------|---------|---------|-------------|
| `lr` | 1e-4 | 1e-4 | 1e-4 | Learning rate |
| `weight_decay` | 0.1 | 1e-4 | 1e-6 | Regularization |
| `epochs` | 50 | 30 | 30 | Max epochs |
| `patience` | 5 | 5 | 7 | Early stopping |
| `scheduler` | cosine | cosine | plateau | LR scheduler |
| `use_cholec` | false | false | true | Add CholecSeg8k |

---

## 🖥️ Command Line Options

### **Basic Usage**

```bash
python train.py --mode <MODE> --config <CONFIG_FILE>
```

### **Available Options**

```
--mode             Training mode: encoder, decoder, or finetune (required)
--config           Path to YAML config file
--checkpoint       Path to checkpoint to resume from
--device           Device: cuda or cpu (default: cuda)
--epochs           Override epochs from config
--batch_size       Override batch size from config
--lr               Override learning rate from config
--wandb_key        Weights & Biases API key
--experiment_name  Custom experiment name for logging
```

### **Examples**

**Basic training:**
```bash
python train.py --mode encoder --config configs/encoder_distillation.yaml
```

**Override parameters:**
```bash
python train.py \
    --mode encoder \
    --config configs/encoder_distillation.yaml \
    --epochs 100 \
    --batch_size 4 \
    --lr 0.0001
```

**Resume from checkpoint:**
```bash
python train.py \
    --mode decoder \
    --config configs/decoder_distillation.yaml \
    --checkpoint checkpoints/encoder/my_checkpoint.pth
```

**Custom experiment name:**
```bash
python train.py \
    --mode finetune \
    --config configs/finetune.yaml \
    --experiment_name "finetune_v2_higher_lr"
```

**CPU training (debugging):**
```bash
python train.py \
    --mode encoder \
    --config configs/encoder_distillation.yaml \
    --device cpu \
    --batch_size 1
```

---

## 📊 Monitoring

### **Weights & Biases Dashboard**

All training runs are logged to W&B:

1. **Login**: Set `WANDB_API_KEY` environment variable
2. **View runs**: Visit `https://wandb.ai/<your-username>/<project-name>`
3. **Tracked metrics**:
   - Training/Validation Loss
   - Learning Rate
   - IoU, Dice, Sensitivity (for decoder/finetune)
   - Epoch time
   - GPU memory usage

### **Console Output**

```
================================================================================
Starting ENCODER training
Experiment: enc_XYZ
Device: cuda
Epochs: 50
Batch size: 2
Learning rate: 0.0001
Checkpoint: checkpoints/encoder/enc_XYZ.pth
================================================================================

Epoch 1/50
--------------------------------------------------------------------------------
Train Loss: 0.0523
Validation Loss: 0.0612
✓ Validation loss improved: inf → 0.0612
✓ Model saved to checkpoints/encoder/enc_XYZ.pth

Epoch 2/50
--------------------------------------------------------------------------------
Train Loss: 0.0445
Validation Loss: 0.0589
✓ Validation loss improved: 0.0612 → 0.0589
✓ Model saved to checkpoints/encoder/enc_XYZ.pth
...
```

### **Saved Checkpoints**

```
checkpoints/
├── encoder/
│   ├── enc_XYZ.pth              # Best model weights
│   └── enc_XYZ_config.yaml      # Training configuration
├── decoder/
│   ├── dec_ABC.pth
│   └── dec_ABC_config.yaml
└── finetune/
    ├── ft_DEF.pth
    └── ft_DEF_config.yaml
```

---

## 🔧 Troubleshooting

### **Common Issues**

#### **1. CUDA Out of Memory**

```
RuntimeError: CUDA out of memory
```

**Solution**:
```bash
# Reduce batch size
python train.py --mode encoder --config config.yaml --batch_size 1

# Or add to config.yaml:
batch_size: 1
```

#### **2. Dataset Not Found**

```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solution**: Update dataset paths in config file:
```yaml
miccai_train_images:
  - "/absolute/path/to/your/dataset/left_frames"
```

#### **3. W&B Authentication Error**

```
wandb.errors.UsageError: api_key not configured
```

**Solution**:
```bash
# Set API key
export WANDB_API_KEY="your_key_here"

# Or disable W&B (not recommended)
wandb disabled
```

#### **4. Checkpoint Loading Error**

```
RuntimeError: Error(s) in loading state_dict
```

**Solution**: Check mode and checkpoint compatibility:
```bash
# Stage 2 requires Stage 1 checkpoint
python train.py --mode decoder --checkpoint checkpoints/encoder/best.pth

# Stage 3 requires Stage 2 checkpoint  
python train.py --mode finetune --checkpoint checkpoints/decoder/best.pth
```

### **Performance Tips**

**Speed up training:**
```yaml
# Config adjustments
batch_size: 4          # Increase if GPU allows
num_workers: 8         # More data loading threads
use_augmentation: false  # Disable for Stage 1 only
```

**Reduce overfitting:**
```yaml
weight_decay: 0.1      # Higher regularization
patience: 10           # More early stopping patience
use_cholec: true       # Add more data (Stage 3)
```

**Better convergence:**
```yaml
scheduler_type: plateau  # Adaptive LR
lr: 0.00005             # Lower learning rate
epochs: 100             # More training time
```

---

## 📝 Full Pipeline Example

Complete script to run all three stages:

```bash
#!/bin/bash

# Set environment
export WANDB_API_KEY="your_api_key"
export CUDA_VISIBLE_DEVICES=0

# Stage 1: Encoder Distillation
echo "Starting Stage 1: Encoder Distillation..."
python train.py \
    --mode encoder \
    --config configs/encoder_distillation.yaml \
    --experiment_name "exp1_encoder"

# Get best encoder checkpoint
ENCODER_CKPT=$(ls -t checkpoints/encoder/*.pth | head -1)
echo "Best encoder: $ENCODER_CKPT"

# Stage 2: Decoder Distillation
echo "Starting Stage 2: Decoder Distillation..."
python train.py \
    --mode decoder \
    --config configs/decoder_distillation.yaml \
    --checkpoint $ENCODER_CKPT \
    --experiment_name "exp1_decoder"

# Get best decoder checkpoint
DECODER_CKPT=$(ls -t checkpoints/decoder/*.pth | head -1)
echo "Best decoder: $DECODER_CKPT"

# Stage 3: Fine-tuning
echo "Starting Stage 3: Fine-tuning..."
python train.py \
    --mode finetune \
    --config configs/finetune.yaml \
    --checkpoint $DECODER_CKPT \
    --experiment_name "exp1_finetune"

# Get final model
FINAL_MODEL=$(ls -t checkpoints/finetune/*.pth | head -1)
echo "Final model: $FINAL_MODEL"
echo "Training complete!"
```

---

## 🎓 Tips & Best Practices

1. **Always save configs**: Each checkpoint has an associated config file
2. **Monitor early stopping**: Adjust patience based on validation curve
3. **Use validation set**: Don't train on the test set (Dataset 4 for validation)
4. **Check GPU utilization**: Use `nvidia-smi` to monitor GPU usage
5. **Backup checkpoints**: Copy important checkpoints to safe storage
6. **Log everything**: W&B tracks all hyperparameters automatically
7. **Resume training**: If interrupted, resume from last checkpoint
8. **Compare experiments**: Use W&B to compare different configurations

---

## 📚 Additional Resources

- [Training Scripts Documentation](docs/training.md)
- [Dataset Preparation Guide](docs/datasets.md)
- [Model Architecture Details](docs/architecture.md)
- [Troubleshooting FAQ](docs/faq.md)

---

**Questions?** Open an issue on GitHub or contact marco.dezen01@gmail.com

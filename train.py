import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW 
import torch.nn.functional as F 
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt

from dataloader import create_dataloaders
from model import LipNet_Attn 
from utils import plot_loss_curves, plot_roc_auc_curves 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = "/home/name/lip-reading/200_processed" 
SAVE_DIR = "/home/name/lip-reading/200_gray" 

LATEST_CHECKPOINT = os.path.join(SAVE_DIR, "latest_checkpoint.pth")
BEST_MODEL = os.path.join(SAVE_DIR, "best_model.pth")

BATCH_SIZE = 128   
NUM_WORKERS = 8   
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 80 
START_EPOCH = 0
RNN_HIDDEN_SIZE = 256
RNN_NUM_LAYERS = 2
DROPOUT = 0.5

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", unit="batch", leave=False)
    for i, (frames, labels) in enumerate(progress_bar):
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(frames)
        loss = criterion(outputs, labels)

        if torch.isnan(loss):
            print(f"NaN loss detected at batch {i}. Skipping batch.")
            torch.cuda.empty_cache() 
            continue

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * frames.size(0) 
        _, predicted = torch.max(outputs.detach(), 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        current_loss = total_loss / total_samples if total_samples > 0 else 0
        current_acc = correct_predictions / total_samples if total_samples > 0 else 0
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

    progress_bar.close()
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_acc = correct_predictions / total_samples if total_samples > 0 else 0
    return avg_loss, avg_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    all_labels = []
    all_probs = []

    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch", leave=False)
    with torch.no_grad():
        for frames, labels in progress_bar:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            probs = F.softmax(outputs, dim=1)
            
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

            total_loss += loss.item() * frames.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            current_loss = total_loss / total_samples if total_samples > 0 else 0
            current_acc = correct_predictions / total_samples if total_samples > 0 else 0
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

    progress_bar.close()
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_acc = correct_predictions / total_samples if total_samples > 0 else 0
    
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    
    return avg_loss, avg_acc, all_labels, all_probs

def plot_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    fig.suptitle('Training History', fontsize=16)

    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training loss')
    val_loss_data = [l for l in history['val_loss'] if l is not None] 
    if val_loss_data:
        ax1.plot(epochs[:len(val_loss_data)], val_loss_data, 'ro-', label='Validation loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history['train_acc'], 'bo-', label='Training accuracy')
    val_acc_data = [a for a in history['val_acc'] if a is not None] 
    if val_acc_data:
        ax2.plot(epochs[:len(val_acc_data)], val_acc_data, 'ro-', label='Validation accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plot_save_path = os.path.join(SAVE_DIR, "training_history.png")
    try:
        plt.savefig(plot_save_path)
        print(f"Training history plot saved to {plot_save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.show()

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading data...")
    try:
        train_loader, val_loader, test_loader, num_classes = create_dataloaders(
            root_dir=ROOT_DIR,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )
    except ValueError as e:
        print(f"Error loading data: {e}. Exiting.")
        return
    except FileNotFoundError:
        print(f"Error: ROOT_DIR '{ROOT_DIR}' not found. Make sure data is unzipped correctly. Exiting.")
        return


    print(f"Found {num_classes} classes.")

    print("Initializing model...")
    model = LipNet_Attn(
        num_classes=num_classes,
        rnn_hidden_size=RNN_HIDDEN_SIZE,
        rnn_num_layers=RNN_NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    print(f"Model: {model.__class__.__name__}")
    print(f"Model moved to {DEVICE}")
    print(f"Total trainable parameters: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min', 
        factor=0.2, 
        patience=5 
    )

    global START_EPOCH
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}


    if os.path.exists(LATEST_CHECKPOINT):
        print(f"Resuming training from checkpoint: {LATEST_CHECKPOINT}")
        try:
            checkpoint = torch.load(LATEST_CHECKPOINT, map_location=DEVICE)
            if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                if 'scheduler_state_dict' in checkpoint:
                     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                     print("  -> Resuming scheduler state.")

                START_EPOCH = checkpoint['epoch'] + 1
                best_val_acc = checkpoint.get('best_val_acc', 0.0)
                if 'history' in checkpoint:
                    history = checkpoint['history']
                    min_len = min(len(history['train_loss']), len(history['train_acc']), len(history['val_loss']), len(history['val_acc']))
                    history = {k: v[:min_len] for k, v in history.items()}
                    print("  -> Resuming previous training history.")
                print(f"  -> Resuming from Epoch {START_EPOCH}, Best Val Acc: {best_val_acc:.4f}")
            else:
                 print("  -> Checkpoint structure mismatch. Starting fresh.")
                 START_EPOCH = 0
                 best_val_acc = 0.0

        except Exception as e:
            print(f"  -> Error loading checkpoint ({e}). Starting fresh.")
            START_EPOCH = 0
            best_val_acc = 0.0

    else:
        print("Starting new training session (no previous model or checkpoint found).")
        START_EPOCH = 0
        best_val_acc = 0.0


    print(f"\n--- Starting Training (Epochs {START_EPOCH+1} to {EPOCHS}) ---")

    for epoch in range(START_EPOCH, EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        val_loss, val_acc = 0.0, 0.0
        if val_loader:
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step(val_loss) 
        else:
            print("Skipping validation phase.")

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(
            f"\nEpoch {epoch+1}/{EPOCHS} | "
            f"Duration: {epoch_duration:.2f}s | "
            f"LR: {optimizer.param_groups[0]['lr']:.1E} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss if val_loader else None)
        history['val_acc'].append(val_acc if val_loader else None)


        current_val_acc_for_comparison = val_acc if val_loader else train_acc
        if current_val_acc_for_comparison > best_val_acc:
            best_val_acc = current_val_acc_for_comparison
            torch.save(model.state_dict(), BEST_MODEL)
            print(f"  -> New best model saved to {BEST_MODEL} (Acc: {current_val_acc_for_comparison:.4f})")

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'history': history 
        }
        temp_checkpoint_path = LATEST_CHECKPOINT + ".tmp"
        torch.save(checkpoint_data, temp_checkpoint_path)
        os.replace(temp_checkpoint_path, LATEST_CHECKPOINT) 
        print(f"  -> Latest checkpoint saved to {LATEST_CHECKPOINT}")


    print("\n--- Training finished ---")
    
    print(f"Loading best model from {BEST_MODEL} for final analysis...")
    try:
        model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading best model: {e}. Aborting final evaluation.")
        return

    print("Running final evaluation on the validation set with the best model...")
    final_val_loss, final_val_acc, final_val_labels, final_val_probs = evaluate(
        model, val_loader, criterion, DEVICE
    )
    
    print("\n--- Best Model Final Performance ---")
    print(f"  (Loaded from {BEST_MODEL})")
    print(f"  Val Loss:   {final_val_loss:.4f} | Val Acc: {final_val_acc:.4f}")
    
    print("\nGenerating visualizations...")
    
    plot_loss_curves(
        history['train_loss'], 
        history['val_loss'],
        save_path=os.path.join(SAVE_DIR, "loss_curves.png") 
    )
    
    plot_roc_auc_curves(
        final_val_labels, 
        final_val_probs, 
        num_classes,
        save_path=os.path.join(SAVE_DIR, "roc_auc_curves.png") 
    )
    
    print("All tasks complete.")


if __name__ == "__main__":
    main()
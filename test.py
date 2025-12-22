import torch
import torch.nn as nn
from tqdm import tqdm
import os
import random
from torch.utils.data import DataLoader, Subset

try:
    from model import LipNet_Attn
    from dataloader import LRWDataset
except ImportError:
    print("Error: Make sure model.py and dataloader.py are in this folder.")
    exit()

ROOT_DIR = r"D:\lrw-v1\100_proc_rgb" 
MODEL_PATH = "best_finetuned_model(entire).pth"

RNN_HIDDEN_SIZE = 256
RNN_NUM_LAYERS = 2
DROPOUT = 0.5
NUM_CLASSES = 100 

BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Evaluating on Test Set", unit="batch")
    with torch.no_grad():
        for frames, labels in progress_bar:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(frames)
            loss = criterion(outputs, labels)

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
    return avg_loss, avg_acc

if __name__ == "__main__":
    
    print(f"--- Starting Test Set Evaluation ---")
    print(f"Using device: {DEVICE}")
    
    print(f"Loading test data from: {ROOT_DIR}")
    try:
        test_dataset = LRWDataset(root_dir=ROOT_DIR, split='test', augment=False)
        
        num_classes_loaded = len(test_dataset.classes)
        if num_classes_loaded != NUM_CLASSES:
            print(f"Warning: Dataloader found {num_classes_loaded} classes, but config expects {NUM_CLASSES}.")
            NUM_CLASSES = num_classes_loaded
            
        if len(test_dataset) == 0:
            print("Error: The test dataset is empty. Check your dataset 'test' split.")
            exit()
            
        num_test_samples = len(test_dataset)
        subset_size = int(num_test_samples * 1)
        print(f"Found {num_test_samples} test samples. Creating a 10% subset with {subset_size} samples.")
        
        indices = random.sample(range(num_test_samples), subset_size)
        test_subset = Subset(test_dataset, indices)
        
        test_loader = DataLoader(
            test_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS
        )
            
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    print(f"Initializing model (LipNet_Attn)")
    model = LipNet_Attn(
        num_classes=NUM_CLASSES,
        rnn_hidden_size=RNN_HIDDEN_SIZE,
        rnn_num_layers=RNN_NUM_LAYERS,
        dropout=DROPOUT
    )
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at: {MODEL_PATH}")
        print("Please edit the MODEL_PATH variable in this script.")
        exit()
        
    print(f"Loading weights from: {MODEL_PATH}")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("This often means your model config (RNN_HIDDEN_SIZE, etc.)")
        print("does not match the saved .pth file.")
        exit()
        
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    print("Running evaluation...")
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    
    print("\n--- Evaluation Complete (on 10% of Test Set) ---")
    print(f"  Model:        {MODEL_PATH}")
    print(f"  Test Loss:    {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc * 100:.2f}%")
    print("--------------------------------------------------\n")
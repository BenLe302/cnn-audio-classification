import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import AudioCNN
from tqdm import tqdm
import json
from datetime import datetime

class ESC50Dataset(Dataset):
    def __init__(self, audio_files, labels, audio_dir, transform=None):
        self.audio_files = audio_files
        self.labels = labels
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_sample_rate = 22050
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        
        # Load audio file
        audio_path = os.path.join(self.audio_dir, audio_file)
        try:
            audio, sr = librosa.load(audio_path, sr=self.target_sample_rate)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return zeros if file can't be loaded
            audio = np.zeros(self.target_sample_rate * 5)
            sr = self.target_sample_rate
        
        # Ensure audio is 5 seconds long
        target_length = sr * 5
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        
        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        # Convert to tensor
        mel_tensor = torch.FloatTensor(mel_spec_norm).unsqueeze(0)  # Add channel dimension
        
        return mel_tensor, torch.LongTensor([label])[0]

class AudioTrainer:
    def __init__(self, model, device, learning_rate=0.001, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_preds, all_targets
    
    def train(self, train_loader, val_loader, epochs, save_dir="results"):
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_preds, val_targets = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': best_val_acc,
                    'val_loss': val_loss
                }, os.path.join(save_dir, 'best_model.pth'))
                
                print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
        
        # Save final model
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, os.path.join(save_dir, 'final_model.pth'))
        
        # Plot training curves
        self.plot_training_curves(save_dir)
        
        # Generate confusion matrix
        self.plot_confusion_matrix(val_preds, val_targets, save_dir)
        
        return best_val_acc
    
    def plot_training_curves(self, save_dir):
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, predictions, targets, save_dir):
        # ESC-50 class names
        class_names = [
            'dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow',
            'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds', 'water_drops',
            'wind', 'pouring_water', 'toilet_flush', 'thunderstorm', 'crying_baby', 'sneezing',
            'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 'brushing_teeth',
            'snoring', 'drinking_sipping', 'door_knock', 'mouse_click', 'keyboard_typing',
            'door_wood_creaks', 'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm',
            'clock_tick', 'glass_breaking', 'helicopter', 'chainsaw', 'siren', 'car_horn',
            'engine', 'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw'
        ]
        
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

def load_esc50_metadata(meta_file):
    """Load ESC-50 metadata"""
    if os.path.exists(meta_file):
        df = pd.read_csv(meta_file)
        return df['filename'].tolist(), df['target'].tolist()
    else:
        print(f"Metadata file not found: {meta_file}")
        return [], []

def main():
    # Configuration
    config = {
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'audio_dir': 'audio',
        'meta_file': 'meta/esc50.csv',
        'save_dir': 'results_fixed',
        'test_size': 0.2,
        'random_state': 42
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load metadata
    audio_files, labels = load_esc50_metadata(config['meta_file'])
    
    if not audio_files:
        print("No audio files found. Please check the metadata file and audio directory.")
        return
    
    print(f"Loaded {len(audio_files)} audio files")
    
    # Split data
    train_files, val_files, train_labels, val_labels = train_test_split(
        audio_files, labels, test_size=config['test_size'], 
        random_state=config['random_state'], stratify=labels
    )
    
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    
    # Create datasets
    train_dataset = ESC50Dataset(train_files, train_labels, config['audio_dir'])
    val_dataset = ESC50Dataset(val_files, val_labels, config['audio_dir'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    model = AudioCNN(num_classes=50)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = AudioTrainer(model, device, config['learning_rate'], config['weight_decay'])
    
    # Train model
    print("Starting training...")
    best_acc = trainer.train(train_loader, val_loader, config['epochs'], config['save_dir'])
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    
    # Save configuration
    config['best_accuracy'] = best_acc
    config['timestamp'] = datetime.now().isoformat()
    
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    main()
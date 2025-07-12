import torch
import torchaudio
import numpy as np
import librosa
from model import AudioCNN
import os
import subprocess
import tempfile

class AudioProcessor:
    def __init__(self, target_sample_rate=22050, n_mels=128, n_fft=2048, hop_length=512):
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def convert_m4a_to_wav(self, m4a_path):
        """Convert M4A file to WAV using FFmpeg"""
        try:
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            
            # Use FFmpeg to convert M4A to WAV
            cmd = ['ffmpeg', '-i', m4a_path, '-ar', str(self.target_sample_rate), 
                   '-ac', '1', '-y', temp_wav_path]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return temp_wav_path
            else:
                print(f"FFmpeg error: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error converting M4A to WAV: {e}")
            return None
    
    def load_audio(self, file_path):
        """Load audio file and convert to the target sample rate"""
        try:
            # Handle M4A files
            if file_path.lower().endswith('.m4a'):
                wav_path = self.convert_m4a_to_wav(file_path)
                if wav_path is None:
                    raise ValueError(f"Failed to convert M4A file: {file_path}")
                
                # Load the converted WAV file
                audio, sr = librosa.load(wav_path, sr=self.target_sample_rate)
                
                # Clean up temporary file
                os.unlink(wav_path)
                
            else:
                # Load other audio formats
                audio, sr = librosa.load(file_path, sr=self.target_sample_rate)
            
            return audio, sr
            
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None, None
    
    def audio_to_melspectrogram(self, audio, sr):
        """Convert audio to mel spectrogram"""
        try:
            # Ensure audio is the right length (5 seconds)
            target_length = sr * 5  # 5 seconds
            
            if len(audio) > target_length:
                # Trim to 5 seconds
                audio = audio[:target_length]
            elif len(audio) < target_length:
                # Pad with zeros
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
            
            # Normalize to [0, 1]
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
            return mel_spec_norm
            
        except Exception as e:
            print(f"Error converting audio to mel spectrogram: {e}")
            return None
    
    def process_audio_file(self, file_path):
        """Process audio file and return mel spectrogram tensor"""
        # Load audio
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return None
        
        # Convert to mel spectrogram
        mel_spec = self.audio_to_melspectrogram(audio, sr)
        if mel_spec is None:
            return None
        
        # Convert to tensor and add batch and channel dimensions
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
        
        return mel_tensor

class AudioClassifier:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AudioCNN(num_classes=50)
        
        # Load trained weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # ESC-50 class names
        self.class_names = [
            'dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow',
            'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds', 'water_drops',
            'wind', 'pouring_water', 'toilet_flush', 'thunderstorm', 'crying_baby', 'sneezing',
            'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 'brushing_teeth',
            'snoring', 'drinking_sipping', 'door_knock', 'mouse_click', 'keyboard_typing',
            'door_wood_creaks', 'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm',
            'clock_tick', 'glass_breaking', 'helicopter', 'chainsaw', 'siren', 'car_horn',
            'engine', 'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw'
        ]
        
        self.processor = AudioProcessor()
    
    def predict(self, audio_file_path, top_k=5):
        """Predict the class of an audio file"""
        try:
            # Process audio file
            mel_tensor = self.processor.process_audio_file(audio_file_path)
            if mel_tensor is None:
                return None
            
            # Move to device
            mel_tensor = mel_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(mel_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, top_k)
                
                results = []
                for i in range(top_k):
                    class_idx = top_indices[0][i].item()
                    prob = top_probs[0][i].item()
                    class_name = self.class_names[class_idx]
                    results.append((class_name, prob))
                
                return results
                
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    model_path = "models/best_audio_cnn.pth"  # Update with your model path
    
    if os.path.exists(model_path):
        classifier = AudioClassifier(model_path)
        
        # Example prediction
        audio_file = "path/to/your/audio/file.wav"  # Update with your audio file
        if os.path.exists(audio_file):
            predictions = classifier.predict(audio_file)
            
            if predictions:
                print(f"Predictions for {audio_file}:")
                for i, (class_name, prob) in enumerate(predictions):
                    print(f"{i+1}. {class_name}: {prob:.4f}")
            else:
                print("Failed to make predictions")
        else:
            print(f"Audio file not found: {audio_file}")
    else:
        print(f"Model file not found: {model_path}")
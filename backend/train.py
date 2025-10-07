import os
import sys

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.frame_model import train_model

if __name__ == "__main__":
    clean_dir = os.path.join("dataset", "clean")
    noisy_dir = os.path.join("dataset", "noisy")
    
    # Validate directories exist
    if not os.path.exists(clean_dir):
        print(f"ERROR: Clean audio directory not found: {clean_dir}")
        sys.exit(1)
    if not os.path.exists(noisy_dir):
        print(f"ERROR: Noisy audio directory not found: {noisy_dir}")
        sys.exit(1)
    
    # Check if directories have files
    clean_files = [f for f in os.listdir(clean_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    noisy_files = [f for f in os.listdir(noisy_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    
    if not clean_files:
        print(f"ERROR: No audio files found in {clean_dir}")
        sys.exit(1)
    if not noisy_files:
        print(f"ERROR: No audio files found in {noisy_dir}")
        sys.exit(1)
    
    print(f"Training model with {len(clean_files)} clean files and {len(noisy_files)} noisy files...")
    
    try:
        model, mean, std = train_model(clean_dir, noisy_dir)
        print("✅ Training completed successfully!")
        print("Model saved to: backend/models/frame_model.keras")
        print("Stats saved to: backend/models/norm_stats.json")
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        sys.exit(1)
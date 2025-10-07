"""
Optimized model for real-time chunk-based audio enhancement.
This module provides lightweight, frame-wise processing for streaming audio.
"""
import numpy as np
import librosa
import soundfile as sf
import io
from scipy.signal import butter, lfilter
import threading
from collections import deque

# Streaming configuration constants
CHUNK_SIZE = 512
OVERLAP_SIZE = 256  # 50% overlap
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
FRAME_SIZE = (N_FFT // 2 + 1)  # 257 frequency bins
WINDOW_TYPE = 'hann'

class StreamingEnhancer:
    """
    Optimized model wrapper for real-time audio enhancement.
    Handles chunk-based processing with overlap-add reconstruction.
    """
    
    def __init__(self, model, mean, std):
        """
        Initialize streaming enhancer with trained model.
        
        Args:
            model: Trained TensorFlow model
            mean: Feature normalization mean
            std: Feature normalization standard deviation
        """
        self.model = model
        self.mean = mean
        self.std = std
        self.lock = threading.Lock()
        
        # Buffers for overlap-add processing
        self.overlap_buffer = np.zeros(OVERLAP_SIZE, dtype=np.float32)
        self.phase_buffer = deque(maxlen=10)  # Store recent phase information
        self.chunk_counter = 0
        
        # Pre-compute window for overlap-add
        self.window = np.hanning(CHUNK_SIZE)
        self.overlap_window = self.window[-OVERLAP_SIZE:]
        
        print(f"✅ StreamingEnhancer initialized:")
        print(f"   - Chunk size: {CHUNK_SIZE} samples")
        print(f"   - Overlap: {OVERLAP_SIZE} samples")
        print(f"   - Sample rate: {SAMPLE_RATE} Hz")
        print(f"   - Frame size: {FRAME_SIZE} bins")
    
    def enhance_chunk(self, audio_chunk):
        """
        Enhance a single audio chunk with overlap-add processing.
        
        Args:
            audio_chunk: numpy array of shape (chunk_size,)
            
        Returns:
            enhanced_audio: numpy array of enhanced audio
        """
        with self.lock:
            try:
                # Ensure correct input size
                if len(audio_chunk) != CHUNK_SIZE:
                    # Pad or trim to correct size
                    if len(audio_chunk) < CHUNK_SIZE:
                        audio_chunk = np.pad(audio_chunk, 
                                           (0, CHUNK_SIZE - len(audio_chunk)), 
                                           mode='constant', constant_values=0)
                    else:
                        audio_chunk = audio_chunk[:CHUNK_SIZE]
                
                # Apply windowing to reduce artifacts
                windowed_chunk = audio_chunk * self.window
                
                # Combine with overlap from previous chunk
                if self.chunk_counter > 0:
                    # Apply overlap-add for seamless reconstruction
                    windowed_chunk[:OVERLAP_SIZE] += self.overlap_buffer
                
                # Extract features using STFT
                stft = librosa.stft(windowed_chunk, 
                                   n_fft=N_FFT, 
                                   hop_length=HOP_LENGTH, 
                                   window=WINDOW_TYPE)
                
                # Get magnitude and phase
                magnitude = np.abs(stft)
                phase = np.angle(stft)
                
                # Store phase for next chunk
                self.phase_buffer.append(phase)
                
                # Convert to dB scale for model input
                db_features = librosa.amplitude_to_db(magnitude).T
                
                # Normalize features
                normalized_features = (db_features - self.mean) / (self.std + 1e-8)
                
                # Predict enhancement (batch processing for efficiency)
                enhanced_features = self.model.predict(
                    np.expand_dims(normalized_features, axis=0), 
                    verbose=0
                )[0]
                
                # Denormalize
                enhanced_features = enhanced_features * self.std + self.mean
                
                # Convert back to linear magnitude
                enhanced_magnitude = librosa.db_to_amplitude(enhanced_features.T)
                
                # Reconstruct STFT with original phase
                enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
                
                # Convert back to time domain
                enhanced_audio = librosa.istft(enhanced_stft, 
                                             hop_length=HOP_LENGTH, 
                                             window=WINDOW_TYPE,
                                             length=CHUNK_SIZE)
                
                # Apply post-processing filter
                enhanced_audio = self._apply_post_filter(enhanced_audio)
                
                # Normalize output
                max_val = np.max(np.abs(enhanced_audio))
                if max_val > 0:
                    enhanced_audio = enhanced_audio / max_val * 0.95  # Prevent clipping
                
                # Store overlap for next chunk
                self.overlap_buffer = enhanced_audio[-OVERLAP_SIZE:] * self.overlap_window
                
                self.chunk_counter += 1
                
                return enhanced_audio.astype(np.float32)
                
            except Exception as e:
                print(f"Error enhancing chunk {self.chunk_counter}: {e}")
                # Return original audio on error
                return audio_chunk.astype(np.float32)
    
    def _apply_post_filter(self, audio):
        """Apply post-processing filter to reduce artifacts"""
        try:
            # Low-pass filter to remove high-frequency artifacts
            nyquist = SAMPLE_RATE / 2
            cutoff = 8000  # 8kHz cutoff
            normal_cutoff = cutoff / nyquist
            b, a = butter(6, normal_cutoff, btype='low', analog=False)
            return lfilter(b, a, audio)
        except Exception as e:
            print(f"Post-filter failed: {e}")
            return audio
    
    def reset(self):
        """Reset internal buffers for new audio stream"""
        with self.lock:
            self.overlap_buffer = np.zeros(OVERLAP_SIZE, dtype=np.float32)
            self.phase_buffer.clear()
            self.chunk_counter = 0
            print("🔄 StreamingEnhancer buffers reset")
    
    def get_stats(self):
        """Get processing statistics"""
        return {
            'chunks_processed': self.chunk_counter,
            'buffer_size': len(self.phase_buffer),
            'overlap_size': OVERLAP_SIZE,
            'chunk_size': CHUNK_SIZE
        }

class ChunkBuffer:
    """
    Thread-safe buffer for managing audio chunks in streaming mode.
    """
    
    def __init__(self, max_size=10):
        """
        Initialize chunk buffer.
        
        Args:
            max_size: Maximum number of chunks to buffer
        """
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.chunk_id = 0
        
    def add_chunk(self, audio_data, timestamp=None):
        """
        Add audio chunk to buffer.
        
        Args:
            audio_data: numpy array of audio samples
            timestamp: optional timestamp
            
        Returns:
            chunk_id: unique identifier for this chunk
        """
        with self.lock:
            chunk_info = {
                'id': self.chunk_id,
                'data': audio_data,
                'timestamp': timestamp or time.time(),
                'processed': False
            }
            self.buffer.append(chunk_info)
            self.chunk_id += 1
            return chunk_info['id']
    
    def get_next_chunk(self):
        """
        Get next unprocessed chunk from buffer.
        
        Returns:
            chunk_info: dict with chunk data and metadata, or None if empty
        """
        with self.lock:
            for chunk in self.buffer:
                if not chunk['processed']:
                    chunk['processed'] = True
                    return chunk
            return None
    
    def clear(self):
        """Clear all chunks from buffer"""
        with self.lock:
            self.buffer.clear()
            self.chunk_id = 0

def create_streaming_enhancer(model_path=None):
    """
    Create and return a StreamingEnhancer instance.
    
    Args:
        model_path: Optional path to model file
        
    Returns:
        StreamingEnhancer instance or None if model loading fails
    """
    try:
        from frame_model import load_trained_model
        
        # Load the trained model
        model, mean, std = load_trained_model(model_path)
        
        if model is None:
            print("❌ Failed to load model for streaming")
            return None
        
        # Create streaming enhancer
        enhancer = StreamingEnhancer(model, mean, std)
        print("✅ StreamingEnhancer created successfully")
        return enhancer
        
    except Exception as e:
        print(f"❌ Error creating StreamingEnhancer: {e}")
        return None

def process_audio_stream(enhancer, input_stream, output_callback, chunk_size=CHUNK_SIZE):
    """
    Process continuous audio stream with real-time enhancement.
    
    Args:
        enhancer: StreamingEnhancer instance
        input_stream: Generator yielding audio chunks
        output_callback: Function to call with enhanced audio chunks
        chunk_size: Size of audio chunks
    """
    try:
        for chunk in input_stream:
            # Ensure chunk is correct size
            if len(chunk) != chunk_size:
                continue
            
            # Enhance the chunk
            enhanced = enhancer.enhance_chunk(chunk)
            
            # Send enhanced audio to callback
            output_callback(enhanced)
            
    except Exception as e:
        print(f"❌ Error processing audio stream: {e}")

# Pre-compute common values for efficiency
_window_cache = {}

def get_window(size, window_type='hann'):
    """Get cached window function for efficiency"""
    key = (size, window_type)
    if key not in _window_cache:
        if window_type == 'hann':
            _window_cache[key] = np.hanning(size)
        elif window_type == 'hamming':
            _window_cache[key] = np.hamming(size)
        else:
            _window_cache[key] = np.ones(size)  # Rectangular window
    return _window_cache[key]

def validate_chunk(audio_chunk):
    """
    Validate audio chunk for streaming processing.
    
    Args:
        audio_chunk: numpy array
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(audio_chunk, np.ndarray):
        return False
    
    if len(audio_chunk.shape) != 1:
        return False
    
    if audio_chunk.dtype not in [np.float32, np.float64, np.int16, np.int32]:
        return False
    
    if len(audio_chunk) == 0:
        return False
    
    # Check for reasonable audio values
    if np.max(np.abs(audio_chunk)) > 10.0:  # Likely not normalized
        return False
    
    return True

# Import time for timestamp functionality
import time
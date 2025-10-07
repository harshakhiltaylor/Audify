"""
Real-time audio streaming handler for Audify.
Handles WebSocket connections for live speech enhancement.
"""
import io
import json
import numpy as np
import soundfile as sf
import librosa
from flask_socketio import SocketIO, emit
from threading import Lock
import queue
import time
from collections import deque

from models.frame_model import load_trained_model
from data.features import extract_features, butter_lowpass_filter

# Global variables
socketio = None
model = None
mean = None
std = None
processing_lock = Lock()

# Audio configuration
CHUNK_SIZE = 512  # samples per chunk
OVERLAP_SIZE = 256  # 50% overlap
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = int((CHUNK_SIZE / SAMPLE_RATE) * 1000)

# Client sessions storage
client_sessions = {}

class AudioSession:
    """Manages audio processing session for a client"""
    
    def __init__(self, client_id):
        self.client_id = client_id
        self.audio_buffer = deque(maxlen=10)  # Keep last 10 chunks
        self.overlap_buffer = np.zeros(OVERLAP_SIZE, dtype=np.float32)
        self.chunk_counter = 0
        self.is_processing = False
        self.last_activity = time.time()
        
    def add_chunk(self, audio_chunk):
        """Add new audio chunk to buffer"""
        self.audio_buffer.append(audio_chunk)
        self.last_activity = time.time()
        
    def get_windowed_chunk(self):
        """Get audio chunk with overlap for seamless processing"""
        if not self.audio_buffer:
            return None
            
        current_chunk = self.audio_buffer[-1]
        
        # Combine with overlap from previous chunk
        if self.chunk_counter > 0:
            windowed = np.concatenate([self.overlap_buffer, current_chunk])
        else:
            windowed = current_chunk
            
        # Store overlap for next chunk
        if len(current_chunk) >= OVERLAP_SIZE:
            self.overlap_buffer = current_chunk[-OVERLAP_SIZE:]
        
        self.chunk_counter += 1
        return windowed

def init_streaming(app):
    """Initialize streaming with Flask app"""
    global socketio, model, mean, std
    
    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", 
                       async_mode='threading',
                       ping_timeout=60,
                       ping_interval=25)
    
    # Load model
    model, mean, std = load_trained_model()
    if model is None:
        print("WARNING: No trained model found for streaming!")
    
    # Register event handlers
    socketio.on_event('connect', handle_connect)
    socketio.on_event('disconnect', handle_disconnect)
    socketio.on_event('start_streaming', handle_start_streaming)
    socketio.on_event('stop_streaming', handle_stop_streaming)
    socketio.on_event('audio_chunk', handle_audio_chunk)
    
    print("✅ Streaming initialized successfully")
    return socketio

def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    client_sessions[client_id] = AudioSession(client_id)
    
    emit('streaming_ready', {
        'status': 'connected',
        'client_id': client_id,
        'chunk_size': CHUNK_SIZE,
        'sample_rate': SAMPLE_RATE,
        'chunk_duration_ms': CHUNK_DURATION_MS
    })
    
    print(f"Client connected: {client_id}")

def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    if client_id in client_sessions:
        del client_sessions[client_id]
        print(f"Client disconnected: {client_id}")

def handle_start_streaming():
    """Handle start streaming request"""
    client_id = request.sid
    
    if model is None:
        emit('streaming_error', {'error': 'Model not loaded'})
        return
    
    if client_id in client_sessions:
        session = client_sessions[client_id]
        session.is_processing = True
        session.chunk_counter = 0
        session.overlap_buffer = np.zeros(OVERLAP_SIZE, dtype=np.float32)
        
        emit('streaming_started', {
            'status': 'started',
            'timestamp': time.time()
        })
        
        print(f"Streaming started for client: {client_id}")

def handle_stop_streaming():
    """Handle stop streaming request"""
    client_id = request.sid
    
    if client_id in client_sessions:
        session = client_sessions[client_id]
        session.is_processing = False
        
        emit('streaming_stopped', {
            'status': 'stopped',
            'chunks_processed': session.chunk_counter,
            'timestamp': time.time()
        })
        
        print(f"Streaming stopped for client: {client_id}")

def handle_audio_chunk(data):
    """Handle incoming audio chunk for processing"""
    client_id = request.sid
    
    if client_id not in client_sessions:
        emit('streaming_error', {'error': 'Session not found'})
        return
    
    session = client_sessions[client_id]
    
    if not session.is_processing:
        return
    
    try:
        # Decode audio data
        audio_data = np.frombuffer(data['audio'], dtype=np.float32)
        
        # Add to session buffer
        session.add_chunk(audio_data)
        
        # Get windowed chunk for processing
        windowed_chunk = session.get_windowed_chunk()
        
        if windowed_chunk is not None and len(windowed_chunk) >= CHUNK_SIZE:
            # Process chunk in background thread
            socketio.start_background_task(process_audio_chunk, 
                                         client_id, windowed_chunk, 
                                         session.chunk_counter)
            
    except Exception as e:
        emit('streaming_error', {'error': f'Processing error: {str(e)}'})
        print(f"Error processing chunk for {client_id}: {e}")

def process_audio_chunk(client_id, audio_chunk, chunk_id):
    """Process audio chunk with model enhancement"""
    global model, mean, std
    
    try:
        with processing_lock:
            if model is None:
                return
                
            # Ensure minimum length
            if len(audio_chunk) < CHUNK_SIZE:
                audio_chunk = np.pad(audio_chunk, 
                                   (0, CHUNK_SIZE - len(audio_chunk)), 
                                   mode='constant')
            
            # Extract features
            # Convert to temporary file for librosa processing
            temp_buffer = io.BytesIO()
            sf.write(temp_buffer, audio_chunk, SAMPLE_RATE, format='WAV')
            temp_buffer.seek(0)
            
            # Load and process
            y, _ = librosa.load(temp_buffer, sr=SAMPLE_RATE)
            stft = librosa.stft(y, n_fft=512, hop_length=128, window='hann')
            mag = np.abs(stft)
            db_features = librosa.amplitude_to_db(mag).T
            
            # Normalize features
            normalized_features = (db_features - mean) / (std + 1e-8)
            
            # Predict enhancement
            enhanced_features = model.predict(normalized_features, verbose=0)
            
            # Denormalize
            enhanced_features = enhanced_features * std + mean
            
            # Convert back to magnitude
            enhanced_mag = librosa.db_to_amplitude(enhanced_features.T)
            
            # Reconstruct audio using original phase
            enhanced_stft = enhanced_mag * np.exp(1j * np.angle(stft))
            enhanced_audio = librosa.istft(enhanced_stft, 
                                         hop_length=128, 
                                         window='hann',
                                         length=len(y))
            
            # Apply post-processing filter
            enhanced_audio = butter_lowpass_filter(enhanced_audio, 8000, SAMPLE_RATE)
            
            # Normalize output
            enhanced_audio = enhanced_audio / (np.max(np.abs(enhanced_audio)) + 1e-10)
            
            # Convert to bytes for transmission
            enhanced_bytes = enhanced_audio.astype(np.float32).tobytes()
            
            # Emit enhanced audio back to client
            socketio.emit('enhanced_chunk', {
                'audio': enhanced_bytes,
                'chunk_id': chunk_id,
                'timestamp': time.time(),
                'length': len(enhanced_audio)
            }, room=client_id)
            
    except Exception as e:
        print(f"Error processing chunk {chunk_id} for {client_id}: {e}")
        socketio.emit('streaming_error', {
            'error': f'Enhancement failed: {str(e)}',
            'chunk_id': chunk_id
        }, room=client_id)

def cleanup_inactive_sessions():
    """Clean up inactive sessions (run periodically)"""
    current_time = time.time()
    inactive_sessions = []
    
    for client_id, session in client_sessions.items():
        if current_time - session.last_activity > 300:  # 5 minutes timeout
            inactive_sessions.append(client_id)
    
    for client_id in inactive_sessions:
        del client_sessions[client_id]
        print(f"Cleaned up inactive session: {client_id}")

# Import request object for session handling
from flask import request
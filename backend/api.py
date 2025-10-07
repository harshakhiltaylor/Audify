# from flask import Flask, request, jsonify, send_file, send_from_directory
# from flask_cors import CORS
# import io
# import os
# import uuid
# import threading
# import time
# import json
# import numpy as np
# from scipy import signal
# from pesq import pesq
# from pystoi import stoi
# import warnings
# from werkzeug.utils import secure_filename
# import soundfile as sf
# import numpy as np

# from models.frame_model import (
#     load_trained_model,
#     enhance_audio as enhance_func
# )
# from metrics.quality import segmental_snr, compute_pesq, compute_stoi

# app = Flask(__name__, static_folder='../outputs', static_url_path='/outputs')
# CORS(app)

# # Global storage for processing status
# processing_status = {}
# processing_lock = threading.Lock()

# # Load model on startup
# model, mean, std = load_trained_model()
# if model is None:
#     print("WARNING: No trained model found! Please run 'python backend/train.py' first.")

# # Create necessary directories
# os.makedirs('temp', exist_ok=True)
# os.makedirs('outputs', exist_ok=True)

# @app.route('/')
# def index():
#     """Serve the main HTML page"""
#     return send_from_directory('../', 'index.html')

# @app.route('/health')
# def health():
#     """Health check endpoint"""
#     return jsonify({
#         "status": "healthy",
#         "model_loaded": model is not None
#     })

# @app.route('/enhance', methods=['POST'])
# def enhance():
#     """Start audio enhancement process"""
#     if 'audio' not in request.files:
#         return jsonify({'success': False, 'error': 'No audio file provided'}), 400
    
#     if model is None:
#         return jsonify({'success': False, 'error': 'Model not loaded. Please train the model first.'}), 500
    
#     file = request.files['audio']
#     if file.filename == '':
#         return jsonify({'success': False, 'error': 'No file selected'}), 400
    
#     try:
#         # Generate unique processing ID
#         processing_id = str(uuid.uuid4())
        
#         # Save uploaded file
#         filename = secure_filename(file.filename)
#         temp_path = os.path.join('temp', f"{processing_id}_{filename}")
#         file.save(temp_path)
        
#         # Initialize processing status
#         with processing_lock:
#             processing_status[processing_id] = {
#                 'status': 'processing',
#                 'progress': 0,
#                 'error': None,
#                 'result': None
#             }
        
#         # Start processing in background thread
#         thread = threading.Thread(target=process_audio, args=(processing_id, temp_path))
#         thread.daemon = True
#         thread.start()
        
#         return jsonify({
#             'success': True,
#             'processing_id': processing_id
#         })
        
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)}), 500

# def process_audio(processing_id, input_path):
#     """Background processing function"""
#     try:
#         # Generate output filename
#         output_filename = f"enhanced_{processing_id}.wav"
#         output_path = os.path.join('/outputs', output_filename)
        
#         # Create output buffer for metrics calculation
#         output_buffer = io.BytesIO()
#         # Enhance audio
#         enhance_func(model, input_path, mean, std, 
#                     output_path=output_path,
#                     output_buffer=output_buffer, update_progress=update_progress,
#                     processing_id=processing_id)
#         # Load original and enhanced audio for metrics
#         original_audio, sr = sf.read(input_path)
#         enhanced_audio, _ = sf.read(output_path)
        
#         # Calculate quality metrics
#         metrics = calculate_metrics(original_audio, enhanced_audio, sr)
        
#         with processing_lock:
#             processing_status[processing_id] = {
#                 'status': 'completed',
#                 'progress': 100,
#                 'error': None,
#                 'result': {
#                     'success': True,
#                     'output_filename': output_filename,
#                     'metrics': metrics
#                 }
#             }
        
#         # Clean up temp file
#         try:
#             os.remove(input_path)
#         except:
#             pass
            
#     except Exception as e:
#         with processing_lock:
#             processing_status[processing_id] = {
#                 'status': 'error',
#                 'progress': 0,
#                 'error': str(e),
#                 'result': None
#             }

# def update_progress(processing_id, progress):
#     """Update processing progress"""
#     with processing_lock:
#         if processing_id in processing_status:
#             processing_status[processing_id]['progress'] = progress


# def segmental_snr(clean, enhanced, frame_len=160, overlap=0.5):
#     """
#     Calculate segmental SNR
#     """
#     try:
#         hop_len = int(frame_len * (1 - overlap))
#         n_frames = (len(clean) - frame_len) // hop_len + 1
        
#         snr_segments = []
#         for i in range(n_frames):
#             start = i * hop_len
#             end = start + frame_len
            
#             clean_frame = clean[start:end]
#             enhanced_frame = enhanced[start:end]
            
#             # Calculate noise as difference
#             noise = clean_frame - enhanced_frame
            
#             # Calculate power
#             signal_power = np.mean(clean_frame ** 2)
#             noise_power = np.mean(noise ** 2)
            
#             # Avoid division by zero
#             if noise_power > 1e-10 and signal_power > 1e-10:
#                 snr_db = 10 * np.log10(signal_power / noise_power)
#                 # Clip extreme values
#                 snr_db = np.clip(snr_db, -10, 35)
#                 snr_segments.append(snr_db)
        
#         return np.mean(snr_segments) if snr_segments else 0.0
#     except Exception as e:
#         print(f"Segmental SNR calculation failed: {e}")
#         return 0.0

# def compute_pesq_safe(clean, enhanced, sr):
#     """
#     Safely compute PESQ with proper error handling
#     """
#     try:
#         # PESQ only works with 8kHz or 16kHz
#         if sr not in [8000, 16000]:
#             # Resample to 16kHz
#             target_sr = 16000
#             clean_resampled = signal.resample(clean, int(len(clean) * target_sr / sr))
#             enhanced_resampled = signal.resample(enhanced, int(len(enhanced) * target_sr / sr))
#         else:
#             clean_resampled = clean
#             enhanced_resampled = enhanced
#             target_sr = sr
        
#         # Ensure minimum length (0.25 seconds)
#         min_samples = int(0.25 * target_sr)
#         if len(clean_resampled) < min_samples:
#             return 0.0
        
#         # Normalize to prevent clipping
#         clean_resampled = clean_resampled / (np.max(np.abs(clean_resampled)) + 1e-10)
#         enhanced_resampled = enhanced_resampled / (np.max(np.abs(enhanced_resampled)) + 1e-10)
        
#         # PESQ expects reference first, then degraded
#         score = pesq(target_sr, clean_resampled, enhanced_resampled, 'wb')
#         return score
#     except Exception as e:
#         print(f"PESQ calculation failed: {e}")
#         return 0.0

# def compute_stoi_safe(clean, enhanced, sr):
#     """
#     Safely compute STOI with proper error handling
#     """
#     try:
#         # Normalize signals
#         clean_norm = clean / (np.max(np.abs(clean)) + 1e-10)
#         enhanced_norm = enhanced / (np.max(np.abs(enhanced)) + 1e-10)
        
#         score = stoi(clean_norm, enhanced_norm, sr, extended=False)
#         return score
#     except Exception as e:
#         print(f"STOI calculation failed: {e}")
#         return 0.0

# def align_signals(reference, test, max_delay=None):
#     """
#     Align two signals using cross-correlation
#     """
#     if max_delay is None:
#         max_delay = min(len(reference), len(test)) // 4
    
#     # Compute cross-correlation
#     correlation = np.correlate(reference, test, mode='full')
    
#     # Find the delay
#     delay = np.argmax(correlation) - len(test) + 1
#     delay = np.clip(delay, -max_delay, max_delay)
    
#     # Align signals
#     if delay > 0:
#         # test is delayed
#         aligned_ref = reference[delay:]
#         aligned_test = test[:-delay] if delay < len(test) else test
#     elif delay < 0:
#         # reference is delayed
#         aligned_ref = reference[:delay]
#         aligned_test = test[-delay:]
#     else:
#         aligned_ref = reference
#         aligned_test = test
    
#     # Ensure same length
#     min_len = min(len(aligned_ref), len(aligned_test))
#     return aligned_ref[:min_len], aligned_test[:min_len]

# def calculate_metrics(original, enhanced, sr, align_signals_flag=True):
#     """
#     Calculate comprehensive audio quality metrics
#     """
#     try:
#         # Validate inputs
#         if len(original) == 0 or len(enhanced) == 0:
#             raise ValueError("Empty audio signals")
        
#         if sr <= 0:
#             raise ValueError("Invalid sample rate")
        
#         # Convert to numpy arrays
#         original = np.array(original, dtype=np.float32)
#         enhanced = np.array(enhanced, dtype=np.float32)
        
#         # Align signals if requested
#         if align_signals_flag:
#             original, enhanced = align_signals(original, enhanced)
#         else:
#             # Ensure same length
#             min_len = min(len(original), len(enhanced))
#             original = original[:min_len]
#             enhanced = enhanced[:min_len]
        
#         # Check minimum length
#         if len(original) < sr * 0.1:  # At least 0.1 seconds
#             warnings.warn("Audio too short for reliable metrics")
        
#         # Calculate metrics
#         seg_snr = segmental_snr(original, enhanced)
#         pesq_score = compute_pesq_safe(original, enhanced, sr)
#         stoi_score = compute_stoi_safe(original, enhanced, sr)
        
#         # Calculate SNR (improvement over original)
#         noise = original - enhanced
#         signal_power = np.mean(original ** 2)
#         noise_power = np.mean(noise ** 2)
        
#         if noise_power > 1e-10:
#             snr = 10 * np.log10(signal_power / noise_power)
#         else:
#             snr = 60.0  # Very high SNR if no noise
        
#         # Additional metrics
#         signal_length = len(enhanced) / sr
#         signal_rms = np.sqrt(np.mean(enhanced ** 2))
        
#         # Calculate dynamic range
#         dynamic_range = 20 * np.log10(np.max(np.abs(enhanced)) / (signal_rms + 1e-10))
        
#         return {
#             'segmental_snr': float(seg_snr),
#             'pesq': float(pesq_score),
#             'stoi': float(stoi_score),
#             'snr': float(snr),
#             'signal_length': float(signal_length),
#             'signal_rms': float(signal_rms),
#             'dynamic_range': float(dynamic_range)
#         }
        
#     except Exception as e:
#         print(f"[calculate_metrics] failed: {e}")
#         return {
#             'segmental_snr': 0.0,
#             'pesq': 0.0,
#             'stoi': 0.0,
#             'snr': 0.0,
#             'signal_length': 0.0,
#             'signal_rms': 0.0,
#             'dynamic_range': 0.0
#         }

# @app.route('/status/<processing_id>')
# def get_status(processing_id):
#     """Get processing status"""
#     with processing_lock:
#         if processing_id in processing_status:
#             return jsonify(processing_status[processing_id])
#         else:
#             return jsonify({'status': 'not_found'}), 404

# @app.route('/outputs/<filename>')
# def download_file(filename):
#     """Download enhanced audio file"""
#     try:
#         return send_from_directory('/outputs', filename, as_attachment=False)
#     except FileNotFoundError:
#         return jsonify({'error': 'File not found'}), 404

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=False)







from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import io
import os
import uuid
import threading
import time
import json
import numpy as np
from scipy import signal
from pesq import pesq
from pystoi import stoi
import warnings
from werkzeug.utils import secure_filename
import soundfile as sf
import numpy as np

from models.frame_model import (
    load_trained_model,
    enhance_audio as enhance_func
)
from metrics.quality import segmental_snr, compute_pesq, compute_stoi

# Import streaming functionality
try:
    from streaming_handler import init_streaming
    STREAMING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Streaming handler not available: {e}")
    STREAMING_AVAILABLE = False

app = Flask(__name__, static_folder='../outputs', static_url_path='/outputs')
CORS(app)

# Initialize SocketIO for streaming if available
socketio = None
if STREAMING_AVAILABLE:
    try:
        socketio = init_streaming(app)
        print("✅ SocketIO streaming initialized in API module")
    except Exception as e:
        print(f"❌ Failed to initialize streaming in API: {e}")
        STREAMING_AVAILABLE = False

# Global storage for processing status
processing_status = {}
processing_lock = threading.Lock()

# Load model on startup
model, mean, std = load_trained_model()
if model is None:
    print("WARNING: No trained model found! Please run 'python backend/train.py' first.")

# Create necessary directories
os.makedirs('temp', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('../', 'index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "streaming_available": STREAMING_AVAILABLE,
        "socketio_enabled": socketio is not None
    })

@app.route('/streaming/health')
def streaming_health():
    """Streaming-specific health check endpoint"""
    if not STREAMING_AVAILABLE:
        return jsonify({
            "status": "unavailable",
            "error": "Streaming module not loaded"
        }), 503
    
    return jsonify({
        "status": "available",
        "socketio_enabled": socketio is not None,
        "model_loaded": model is not None
    })

@app.route('/enhance', methods=['POST'])
def enhance():
    """Start audio enhancement process"""
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file provided'}), 400
    
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded. Please train the model first.'}), 500
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    try:
        # Generate unique processing ID
        processing_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        temp_path = os.path.join('temp', f"{processing_id}_{filename}")
        file.save(temp_path)
        
        # Initialize processing status
        with processing_lock:
            processing_status[processing_id] = {
                'status': 'processing',
                'progress': 0,
                'error': None,
                'result': None
            }
        
        # Start processing in background thread
        thread = threading.Thread(target=process_audio, args=(processing_id, temp_path))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'processing_id': processing_id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def process_audio(processing_id, input_path):
    """Background processing function"""
    try:
        # Generate output filename
        output_filename = f"enhanced_{processing_id}.wav"
        output_path = os.path.join('outputs', output_filename)
        
        # Create output buffer for metrics calculation
        output_buffer = io.BytesIO()
        
        # Enhance audio
        enhance_func(model, input_path, mean, std, 
                    output_path=output_path,
                    output_buffer=output_buffer, 
                    update_progress=lambda progress: update_progress(processing_id, progress),
                    processing_id=processing_id)
        
        # Load original and enhanced audio for metrics
        original_audio, sr = sf.read(input_path)
        enhanced_audio, _ = sf.read(output_path)
        
        # Calculate quality metrics
        metrics = calculate_metrics(original_audio, enhanced_audio, sr)
        
        with processing_lock:
            processing_status[processing_id] = {
                'status': 'completed',
                'progress': 100,
                'error': None,
                'result': {
                    'success': True,
                    'output_filename': output_filename,
                    'metrics': metrics
                }
            }
        
        # Clean up temp file
        try:
            os.remove(input_path)
        except:
            pass
            
    except Exception as e:
        with processing_lock:
            processing_status[processing_id] = {
                'status': 'error',
                'progress': 0,
                'error': str(e),
                'result': None
            }

def update_progress(processing_id, progress):
    """Update processing progress"""
    with processing_lock:
        if processing_id in processing_status:
            processing_status[processing_id]['progress'] = progress

def segmental_snr(clean, enhanced, frame_len=160, overlap=0.5):
    """Calculate segmental SNR"""
    try:
        hop_len = int(frame_len * (1 - overlap))
        n_frames = (len(clean) - frame_len) // hop_len + 1
        
        snr_segments = []
        for i in range(n_frames):
            start = i * hop_len
            end = start + frame_len
            
            clean_frame = clean[start:end]
            enhanced_frame = enhanced[start:end]
            
            # Calculate noise as difference
            noise = clean_frame - enhanced_frame
            
            # Calculate power
            signal_power = np.mean(clean_frame ** 2)
            noise_power = np.mean(noise ** 2)
            
            # Avoid division by zero
            if noise_power > 1e-10 and signal_power > 1e-10:
                snr_db = 10 * np.log10(signal_power / noise_power)
                # Clip extreme values
                snr_db = np.clip(snr_db, -10, 35)
                snr_segments.append(snr_db)
        
        return np.mean(snr_segments) if snr_segments else 0.0
    except Exception as e:
        print(f"Segmental SNR calculation failed: {e}")
        return 0.0

def compute_pesq_safe(clean, enhanced, sr):
    """Safely compute PESQ with proper error handling"""
    try:
        # PESQ only works with 8kHz or 16kHz
        if sr not in [8000, 16000]:
            # Resample to 16kHz
            target_sr = 16000
            clean_resampled = signal.resample(clean, int(len(clean) * target_sr / sr))
            enhanced_resampled = signal.resample(enhanced, int(len(enhanced) * target_sr / sr))
        else:
            clean_resampled = clean
            enhanced_resampled = enhanced
            target_sr = sr
        
        # Ensure minimum length (0.25 seconds)
        min_samples = int(0.25 * target_sr)
        if len(clean_resampled) < min_samples:
            return 0.0
        
        # Normalize to prevent clipping
        clean_resampled = clean_resampled / (np.max(np.abs(clean_resampled)) + 1e-10)
        enhanced_resampled = enhanced_resampled / (np.max(np.abs(enhanced_resampled)) + 1e-10)
        
        # PESQ expects reference first, then degraded
        score = pesq(target_sr, clean_resampled, enhanced_resampled, 'wb')
        return score
    except Exception as e:
        print(f"PESQ calculation failed: {e}")
        return 0.0

def compute_stoi_safe(clean, enhanced, sr):
    """Safely compute STOI with proper error handling"""
    try:
        # Normalize signals
        clean_norm = clean / (np.max(np.abs(clean)) + 1e-10)
        enhanced_norm = enhanced / (np.max(np.abs(enhanced)) + 1e-10)
        
        score = stoi(clean_norm, enhanced_norm, sr, extended=False)
        return score
    except Exception as e:
        print(f"STOI calculation failed: {e}")
        return 0.0

def align_signals(reference, test, max_delay=None):
    """Align two signals using cross-correlation"""
    if max_delay is None:
        max_delay = min(len(reference), len(test)) // 4
    
    # Compute cross-correlation
    correlation = np.correlate(reference, test, mode='full')
    
    # Find the delay
    delay = np.argmax(correlation) - len(test) + 1
    delay = np.clip(delay, -max_delay, max_delay)
    
    # Align signals
    if delay > 0:
        # test is delayed
        aligned_ref = reference[delay:]
        aligned_test = test[:-delay] if delay < len(test) else test
    elif delay < 0:
        # reference is delayed
        aligned_ref = reference[:delay]
        aligned_test = test[-delay:]
    else:
        aligned_ref = reference
        aligned_test = test
    
    # Ensure same length
    min_len = min(len(aligned_ref), len(aligned_test))
    return aligned_ref[:min_len], aligned_test[:min_len]

def calculate_metrics(original, enhanced, sr, align_signals_flag=True):
    """Calculate comprehensive audio quality metrics"""
    try:
        # Validate inputs
        if len(original) == 0 or len(enhanced) == 0:
            raise ValueError("Empty audio signals")
        
        if sr <= 0:
            raise ValueError("Invalid sample rate")
        
        # Convert to numpy arrays
        original = np.array(original, dtype=np.float32)
        enhanced = np.array(enhanced, dtype=np.float32)
        
        # Align signals if requested
        if align_signals_flag:
            original, enhanced = align_signals(original, enhanced)
        else:
            # Ensure same length
            min_len = min(len(original), len(enhanced))
            original = original[:min_len]
            enhanced = enhanced[:min_len]
        
        # Check minimum length
        if len(original) < sr * 0.1:  # At least 0.1 seconds
            warnings.warn("Audio too short for reliable metrics")
        
        # Calculate metrics
        seg_snr = segmental_snr(original, enhanced)
        pesq_score = compute_pesq_safe(original, enhanced, sr)
        stoi_score = compute_stoi_safe(original, enhanced, sr)
        
        # Calculate SNR (improvement over original)
        noise = original - enhanced
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 1e-10:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 60.0  # Very high SNR if no noise
        
        # Additional metrics
        signal_length = len(enhanced) / sr
        signal_rms = np.sqrt(np.mean(enhanced ** 2))
        
        # Calculate dynamic range
        dynamic_range = 20 * np.log10(np.max(np.abs(enhanced)) / (signal_rms + 1e-10))
        
        return {
            'segmental_snr': float(seg_snr),
            'pesq': float(pesq_score),
            'stoi': float(stoi_score),
            'snr': float(snr),
            'signal_length': float(signal_length),
            'signal_rms': float(signal_rms),
            'dynamic_range': float(dynamic_range)
        }
        
    except Exception as e:
        print(f"[calculate_metrics] failed: {e}")
        return {
            'segmental_snr': 0.0,
            'pesq': 0.0,
            'stoi': 0.0,
            'snr': 0.0,
            'signal_length': 0.0,
            'signal_rms': 0.0,
            'dynamic_range': 0.0
        }

@app.route('/status/<processing_id>')
def get_status(processing_id):
    """Get processing status"""
    with processing_lock:
        if processing_id in processing_status:
            return jsonify(processing_status[processing_id])
        else:
            return jsonify({'status': 'not_found'}), 404

@app.route('/outputs/<filename>')
def download_file(filename):
    """Download enhanced audio file"""
    try:
        return send_from_directory('outputs', filename, as_attachment=False)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

def get_app():
    """Return the Flask app instance for external use"""
    return app

def get_socketio():
    """Return the SocketIO instance for external use"""
    return socketio

if __name__ == '__main__':
    if socketio and STREAMING_AVAILABLE:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    else:
        app.run(host='0.0.0.0', port=5000, debug=False)
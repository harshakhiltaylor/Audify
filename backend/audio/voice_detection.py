"""
Voice Activity Detection (VAD) for real-time audio streaming.
Uses webrtcvad for efficient chunk-based voice detection to reduce processing load.
"""
import numpy as np
import threading
import time
from collections import deque
import logging

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False
    print("⚠️ webrtcvad not available, using energy-based VAD fallback")

# Configuration constants
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30  # 30ms frames (480 samples at 16kHz)
CHUNK_SIZE = 512  # samples per processing chunk
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 samples
OVERLAP_FRAMES = 2  # Number of overlapping frames for smoothing
MIN_SPEECH_FRAMES = 3  # Minimum consecutive frames for speech detection
MIN_SILENCE_FRAMES = 10  # Minimum consecutive frames for silence detection

class VoiceActivityDetector:
    """
    Thread-safe Voice Activity Detection using webrtcvad or energy-based fallback.
    Processes audio chunks and determines voice activity with configurable sensitivity.
    """
    
    def __init__(self, sensitivity=2, sample_rate=SAMPLE_RATE):
        """
        Initialize Voice Activity Detector.
        
        Args:
            sensitivity: VAD sensitivity level (0=least, 3=most sensitive)
            sample_rate: Audio sample rate in Hz
        """
        self.sensitivity = max(0, min(3, sensitivity))  # Clamp to 0-3
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * FRAME_DURATION_MS / 1000)
        
        # Initialize webrtcvad if available
        self.vad = None
        self.use_webrtc = False
        
        if WEBRTCVAD_AVAILABLE and sample_rate in [8000, 16000, 32000, 48000]:
            try:
                self.vad = webrtcvad.Vad()
                self.vad.set_mode(self.sensitivity)
                self.use_webrtc = True
                print(f"✅ WebRTC VAD initialized: sensitivity={sensitivity}, sr={sample_rate}")
            except Exception as e:
                print(f"⚠️ WebRTC VAD initialization failed: {e}")
        
        if not self.use_webrtc:
            print(f"📊 Using energy-based VAD fallback: sensitivity={sensitivity}")
        
        # Threading protection
        self.lock = threading.RLock()
        
        # Frame history for smoothing
        self.frame_history = deque(maxlen=OVERLAP_FRAMES * 2)
        self.speech_frames = 0
        self.silence_frames = 0
        
        # Energy-based VAD parameters (fallback)
        self.energy_threshold = self._calculate_energy_threshold()
        self.noise_floor = 0.001
        self.speech_ratio_threshold = 0.3
        
        # Statistics
        self.stats = {
            'total_chunks': 0,
            'voice_chunks': 0,
            'silence_chunks': 0,
            'processing_time': 0.0
        }
        
    def _calculate_energy_threshold(self):
        """Calculate energy threshold based on sensitivity"""
        # Higher sensitivity = lower threshold
        base_threshold = 0.01
        sensitivity_factor = (3 - self.sensitivity) / 3.0  # Invert sensitivity
        return base_threshold * (1 + sensitivity_factor * 4)
    
    def detect_voice_activity(self, audio_chunk):
        """
        Detect voice activity in audio chunk.
        
        Args:
            audio_chunk: numpy array of audio samples
            
        Returns:
            bool: True if voice activity detected, False otherwise
        """
        with self.lock:
            start_time = time.time()
            
            try:
                # Validate input
                if not isinstance(audio_chunk, np.ndarray):
                    audio_chunk = np.array(audio_chunk, dtype=np.float32)
                
                if len(audio_chunk) == 0:
                    return False
                
                # Ensure correct sample rate scaling
                if len(audio_chunk) != CHUNK_SIZE:
                    # Pad or trim to expected size
                    if len(audio_chunk) < CHUNK_SIZE:
                        audio_chunk = np.pad(audio_chunk, 
                                           (0, CHUNK_SIZE - len(audio_chunk)), 
                                           mode='constant')
                    else:
                        audio_chunk = audio_chunk[:CHUNK_SIZE]
                
                # Process chunk using frame-based detection
                voice_detected = self._process_chunk_frames(audio_chunk)
                
                # Update statistics
                self.stats['total_chunks'] += 1
                if voice_detected:
                    self.stats['voice_chunks'] += 1
                else:
                    self.stats['silence_chunks'] += 1
                
                self.stats['processing_time'] += time.time() - start_time
                
                return voice_detected
                
            except Exception as e:
                print(f"❌ VAD error: {e}")
                return True  # Process by default on error
    
    def _process_chunk_frames(self, audio_chunk):
        """Process audio chunk as overlapping frames for VAD"""
        try:
            # Convert to 16-bit PCM for webrtcvad if needed
            if self.use_webrtc:
                # Scale to int16 range
                pcm_audio = (audio_chunk * 32767).astype(np.int16)
            
            voice_frames = 0
            total_frames = 0
            
            # Process overlapping frames within chunk
            hop_size = self.frame_size // 2  # 50% overlap
            
            for start in range(0, len(audio_chunk) - self.frame_size + 1, hop_size):
                end = start + self.frame_size
                frame = audio_chunk[start:end]
                
                if len(frame) != self.frame_size:
                    continue
                
                # Detect voice in frame
                if self.use_webrtc:
                    pcm_frame = pcm_audio[start:end].tobytes()
                    try:
                        is_voice = self.vad.is_speech(pcm_frame, self.sample_rate)
                    except Exception as e:
                        # Fallback to energy-based detection
                        is_voice = self._energy_based_detection(frame)
                else:
                    is_voice = self._energy_based_detection(frame)
                
                if is_voice:
                    voice_frames += 1
                total_frames += 1
            
            if total_frames == 0:
                return False
            
            # Calculate voice ratio for this chunk
            voice_ratio = voice_frames / total_frames
            
            # Apply smoothing with frame history
            return self._apply_temporal_smoothing(voice_ratio)
            
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            return False
    
    def _energy_based_detection(self, frame):
        """Fallback energy-based voice activity detection"""
        try:
            # Calculate frame energy
            energy = np.mean(frame ** 2)
            
            # Update noise floor estimate (exponential moving average)
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * energy
            
            # Adaptive threshold based on noise floor
            adaptive_threshold = max(self.energy_threshold, 
                                   self.noise_floor * 3.0)
            
            # Check if energy exceeds threshold
            is_voice = energy > adaptive_threshold
            
            # Additional spectral analysis for better discrimination
            if is_voice:
                # Check for spectral characteristics of speech
                spectral_centroid = self._calculate_spectral_centroid(frame)
                # Speech typically has centroid between 500-4000 Hz
                if spectral_centroid < 500 or spectral_centroid > 8000:
                    is_voice = False
            
            return is_voice
            
        except Exception as e:
            print(f"❌ Energy detection error: {e}")
            return False
    
    def _calculate_spectral_centroid(self, frame):
        """Calculate spectral centroid for speech discrimination"""
        try:
            # Simple spectral centroid calculation
            fft = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(len(frame), 1/self.sample_rate)
            
            if np.sum(fft) == 0:
                return 0
            
            centroid = np.sum(freqs * fft) / np.sum(fft)
            return centroid
            
        except Exception as e:
            return 2000  # Default speech-like centroid
    
    def _apply_temporal_smoothing(self, voice_ratio):
        """Apply temporal smoothing to reduce false positives/negatives"""
        try:
            # Add current ratio to history
            self.frame_history.append(voice_ratio)
            
            # Calculate smoothed voice probability
            if len(self.frame_history) > 0:
                smooth_ratio = np.mean(list(self.frame_history))
            else:
                smooth_ratio = voice_ratio
            
            # Determine voice activity based on threshold
            is_voice = smooth_ratio > self.speech_ratio_threshold
            
            # Apply hysteresis to prevent rapid switching
            if is_voice:
                self.speech_frames += 1
                self.silence_frames = max(0, self.silence_frames - 1)
            else:
                self.silence_frames += 1
                self.speech_frames = max(0, self.speech_frames - 1)
            
            # Final decision based on consecutive frame counts
            if self.speech_frames >= MIN_SPEECH_FRAMES:
                return True
            elif self.silence_frames >= MIN_SILENCE_FRAMES:
                return False
            else:
                # Use current ratio for unclear cases
                return smooth_ratio > self.speech_ratio_threshold
                
        except Exception as e:
            print(f"❌ Temporal smoothing error: {e}")
            return voice_ratio > self.speech_ratio_threshold
    
    def set_sensitivity(self, sensitivity):
        """Update VAD sensitivity level (0-3)"""
        with self.lock:
            self.sensitivity = max(0, min(3, sensitivity))
            
            if self.use_webrtc and self.vad:
                try:
                    self.vad.set_mode(self.sensitivity)
                    print(f"✅ WebRTC VAD sensitivity updated: {sensitivity}")
                except Exception as e:
                    print(f"⚠️ Failed to update WebRTC VAD sensitivity: {e}")
            
            # Update energy threshold for fallback method
            self.energy_threshold = self._calculate_energy_threshold()
            print(f"📊 Energy threshold updated: {self.energy_threshold:.6f}")
    
    def reset(self):
        """Reset VAD state and statistics"""
        with self.lock:
            self.frame_history.clear()
            self.speech_frames = 0
            self.silence_frames = 0
            self.noise_floor = 0.001
            
            # Reset statistics
            self.stats = {
                'total_chunks': 0,
                'voice_chunks': 0,
                'silence_chunks': 0,
                'processing_time': 0.0
            }
            
            print("🔄 VAD state reset")
    
    def get_statistics(self):
        """Get VAD processing statistics"""
        with self.lock:
            stats = self.stats.copy()
            
            if stats['total_chunks'] > 0:
                stats['voice_ratio'] = stats['voice_chunks'] / stats['total_chunks']
                stats['avg_processing_time'] = stats['processing_time'] / stats['total_chunks']
            else:
                stats['voice_ratio'] = 0.0
                stats['avg_processing_time'] = 0.0
            
            stats['method'] = 'WebRTC' if self.use_webrtc else 'Energy-based'
            stats['sensitivity'] = self.sensitivity
            
            return stats
    
    def is_processing_needed(self, audio_chunk):
        """
        Determine if audio chunk needs processing based on VAD.
        
        Args:
            audio_chunk: numpy array of audio samples
            
        Returns:
            bool: True if processing needed (voice detected), False to skip
        """
        return self.detect_voice_activity(audio_chunk)

class StreamingVAD:
    """
    Streaming wrapper for Voice Activity Detection with chunk buffering.
    """
    
    def __init__(self, sensitivity=2, buffer_size=5):
        """
        Initialize streaming VAD.
        
        Args:
            sensitivity: VAD sensitivity level (0-3)
            buffer_size: Number of recent decisions to buffer for smoothing
        """
        self.vad = VoiceActivityDetector(sensitivity)
        self.buffer_size = buffer_size
        self.decision_buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        
        # Streaming statistics
        self.stream_stats = {
            'chunks_processed': 0,
            'chunks_skipped': 0,
            'cpu_savings': 0.0
        }
        
    def should_process_chunk(self, audio_chunk):
        """
        Determine if streaming chunk should be processed.
        
        Args:
            audio_chunk: numpy array of audio samples
            
        Returns:
            bool: True if chunk should be processed, False to skip
        """
        with self.lock:
            # Detect voice activity
            has_voice = self.vad.detect_voice_activity(audio_chunk)
            
            # Add to decision buffer
            self.decision_buffer.append(has_voice)
            
            # Apply buffer-based smoothing
            if len(self.decision_buffer) >= 2:
                # Require majority vote for processing
                voice_votes = sum(self.decision_buffer)
                should_process = voice_votes >= (len(self.decision_buffer) // 2 + 1)
            else:
                should_process = has_voice
            
            # Update statistics
            if should_process:
                self.stream_stats['chunks_processed'] += 1
            else:
                self.stream_stats['chunks_skipped'] += 1
            
            # Calculate CPU savings
            total_chunks = (self.stream_stats['chunks_processed'] + 
                          self.stream_stats['chunks_skipped'])
            if total_chunks > 0:
                self.stream_stats['cpu_savings'] = (
                    self.stream_stats['chunks_skipped'] / total_chunks * 100
                )
            
            return should_process
    
    def get_comprehensive_stats(self):
        """Get combined VAD and streaming statistics"""
        with self.lock:
            vad_stats = self.vad.get_statistics()
            stream_stats = self.stream_stats.copy()
            
            return {
                'vad': vad_stats,
                'streaming': stream_stats,
                'buffer_size': len(self.decision_buffer),
                'current_decisions': list(self.decision_buffer)
            }
    
    def update_sensitivity(self, sensitivity):
        """Update VAD sensitivity"""
        self.vad.set_sensitivity(sensitivity)
    
    def reset_streaming(self):
        """Reset streaming state"""
        with self.lock:
            self.vad.reset()
            self.decision_buffer.clear()
            self.stream_stats = {
                'chunks_processed': 0,
                'chunks_skipped': 0,
                'cpu_savings': 0.0
            }

def create_voice_detector(sensitivity=2, streaming=True):
    """
    Factory function to create appropriate VAD instance.
    
    Args:
        sensitivity: VAD sensitivity level (0-3)
        streaming: If True, create StreamingVAD, else VoiceActivityDetector
        
    Returns:
        VAD instance
    """
    if streaming:
        return StreamingVAD(sensitivity)
    else:
        return VoiceActivityDetector(sensitivity)

def test_vad_performance():
    """Test function to evaluate VAD performance"""
    print("🧪 Testing VAD performance...")
    
    # Create test audio chunks
    sample_rate = SAMPLE_RATE
    chunk_duration = CHUNK_SIZE / sample_rate
    
    # Generate test signals
    silence = np.zeros(CHUNK_SIZE, dtype=np.float32)
    noise = np.random.normal(0, 0.01, CHUNK_SIZE).astype(np.float32)
    speech = np.random.normal(0, 0.1, CHUNK_SIZE).astype(np.float32)  # Simulated speech
    
    # Test VAD
    vad = create_voice_detector(sensitivity=2, streaming=False)
    
    test_cases = [
        ("Silence", silence, False),
        ("Noise", noise, False),
        ("Speech", speech, True)
    ]
    
    print(f"Chunk size: {CHUNK_SIZE} samples ({chunk_duration*1000:.1f}ms)")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"VAD method: {'WebRTC' if WEBRTCVAD_AVAILABLE else 'Energy-based'}")
    print("-" * 50)
    
    for name, audio, expected in test_cases:
        start_time = time.time()
        detected = vad.detect_voice_activity(audio)
        processing_time = (time.time() - start_time) * 1000  # ms
        
        status = "✅ PASS" if detected == expected else "❌ FAIL"
        print(f"{name:8} | Expected: {expected:5} | Got: {detected:5} | "
              f"{processing_time:.2f}ms | {status}")
    
    # Get statistics
    stats = vad.get_statistics()
    print("\n📊 VAD Statistics:")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Voice ratio: {stats['voice_ratio']:.2%}")
    print(f"Avg processing time: {stats['avg_processing_time']*1000:.2f}ms")
    
    return True

if __name__ == "__main__":
    # Run performance test
    test_vad_performance()
    
    # Test streaming VAD
    print("\n🎵 Testing StreamingVAD...")
    streaming_vad = create_voice_detector(sensitivity=2, streaming=True)
    
    # Simulate streaming chunks
    for i in range(10):
        # Alternate between speech and silence
        if i % 3 == 0:
            chunk = np.random.normal(0, 0.1, CHUNK_SIZE).astype(np.float32)  # Speech
        else:
            chunk = np.random.normal(0, 0.005, CHUNK_SIZE).astype(np.float32)  # Silence/noise
        
        should_process = streaming_vad.should_process_chunk(chunk)
        print(f"Chunk {i:2d}: {'PROCESS' if should_process else 'SKIP   '}")
    
    # Final statistics
    final_stats = streaming_vad.get_comprehensive_stats()
    print(f"\n💾 CPU Savings: {final_stats['streaming']['cpu_savings']:.1f}%")
    print("✅ StreamingVAD test completed")
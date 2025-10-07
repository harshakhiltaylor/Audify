"""
Real-time audio buffer management and overlap-add processing for streaming enhancement.
Handles seamless audio chunk reconstruction with minimal latency.
"""
import numpy as np
import threading
import time
from collections import deque
from scipy.signal import windows
import soundfile as sf
import io

# Configuration constants
CHUNK_SIZE = 512  # samples per chunk
OVERLAP_SIZE = 256  # 50% overlap
SAMPLE_RATE = 16000
BUFFER_SIZE = 10  # Maximum chunks to buffer
FADE_LENGTH = 32  # samples for crossfade

class AudioBuffer:
    """
    Thread-safe circular buffer for audio chunk management.
    Supports overlap-add reconstruction for seamless audio processing.
    """
    
    def __init__(self, chunk_size=CHUNK_SIZE, overlap_size=OVERLAP_SIZE, max_chunks=BUFFER_SIZE):
        """
        Initialize audio buffer.
        
        Args:
            chunk_size: Size of each audio chunk in samples
            overlap_size: Overlap between consecutive chunks
            max_chunks: Maximum number of chunks to buffer
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.hop_size = chunk_size - overlap_size
        self.max_chunks = max_chunks
        
        # Thread-safe storage
        self.input_buffer = deque(maxlen=max_chunks)
        self.output_buffer = deque(maxlen=max_chunks)
        self.lock = threading.RLock()
        
        # Overlap-add reconstruction
        self.overlap_buffer = np.zeros(overlap_size, dtype=np.float32)
        self.reconstruction_buffer = np.zeros(chunk_size * 2, dtype=np.float32)
        self.write_position = 0
        
        # Windows for overlap-add
        self.input_window = windows.hann(chunk_size)
        self.overlap_window = windows.hann(overlap_size * 2)
        
        # Statistics
        self.chunks_added = 0
        self.chunks_processed = 0
        self.underruns = 0
        self.overruns = 0
        
        print(f"AudioBuffer initialized: chunk_size={chunk_size}, overlap={overlap_size}, buffer_size={max_chunks}")
    
    def add_input_chunk(self, audio_data, timestamp=None):
        """
        Add audio chunk to input buffer.
        
        Args:
            audio_data: numpy array of audio samples
            timestamp: optional timestamp for synchronization
            
        Returns:
            bool: True if successfully added, False if buffer full
        """
        with self.lock:
            try:
                # Validate input
                if not isinstance(audio_data, np.ndarray):
                    audio_data = np.array(audio_data, dtype=np.float32)
                
                if len(audio_data) != self.chunk_size:
                    # Pad or trim to correct size
                    if len(audio_data) < self.chunk_size:
                        audio_data = np.pad(audio_data, 
                                          (0, self.chunk_size - len(audio_data)), 
                                          mode='constant', constant_values=0)
                    else:
                        audio_data = audio_data[:self.chunk_size]
                
                # Check for buffer overflow
                if len(self.input_buffer) >= self.max_chunks:
                    self.overruns += 1
                    # Remove oldest chunk to make space
                    self.input_buffer.popleft()
                
                # Add chunk with metadata
                chunk_info = {
                    'id': self.chunks_added,
                    'data': audio_data.astype(np.float32),
                    'timestamp': timestamp or time.time(),
                    'processed': False
                }
                
                self.input_buffer.append(chunk_info)
                self.chunks_added += 1
                
                return True
                
            except Exception as e:
                print(f"Error adding input chunk: {e}")
                return False
    
    def get_input_chunk(self):
        """
        Get next unprocessed input chunk.
        
        Returns:
            dict: Chunk info with data and metadata, or None if empty
        """
        with self.lock:
            for chunk in self.input_buffer:
                if not chunk['processed']:
                    chunk['processed'] = True
                    return chunk
            
            # Buffer underrun
            self.underruns += 1
            return None
    
    def add_output_chunk(self, audio_data, chunk_id=None):
        """
        Add enhanced audio chunk to output buffer with overlap-add reconstruction.
        
        Args:
            audio_data: numpy array of enhanced audio samples
            chunk_id: optional chunk identifier for synchronization
            
        Returns:
            numpy array: Reconstructed audio ready for playback, or None
        """
        with self.lock:
            try:
                # Validate input
                if not isinstance(audio_data, np.ndarray):
                    audio_data = np.array(audio_data, dtype=np.float32)
                
                if len(audio_data) != self.chunk_size:
                    # Resize if necessary
                    if len(audio_data) < self.chunk_size:
                        audio_data = np.pad(audio_data, 
                                          (0, self.chunk_size - len(audio_data)), 
                                          mode='constant')
                    else:
                        audio_data = audio_data[:self.chunk_size]
                
                # Apply windowing to reduce artifacts
                windowed_audio = audio_data * self.input_window
                
                # Perform overlap-add reconstruction
                output_audio = self._overlap_add_reconstruction(windowed_audio)
                
                # Store in output buffer
                if output_audio is not None:
                    output_info = {
                        'id': chunk_id or self.chunks_processed,
                        'data': output_audio,
                        'timestamp': time.time(),
                        'length': len(output_audio)
                    }
                    
                    if len(self.output_buffer) >= self.max_chunks:
                        self.output_buffer.popleft()  # Remove oldest
                    
                    self.output_buffer.append(output_info)
                    self.chunks_processed += 1
                
                return output_audio
                
            except Exception as e:
                print(f"Error adding output chunk: {e}")
                return None
    
    def _overlap_add_reconstruction(self, audio_chunk):
        """
        Perform overlap-add reconstruction for seamless audio playback.
        
        Args:
            audio_chunk: Windowed audio chunk
            
        Returns:
            numpy array: Reconstructed audio segment
        """
        try:
            # Handle first chunk
            if self.chunks_processed == 0:
                self.overlap_buffer = audio_chunk[-self.overlap_size:]
                return audio_chunk[:-self.overlap_size]  # Return non-overlapping part
            
            # Overlap-add reconstruction
            overlap_start = audio_chunk[:self.overlap_size]
            overlap_fade = self.overlap_window[:self.overlap_size]
            
            # Crossfade between previous overlap and current overlap
            faded_previous = self.overlap_buffer * (1 - overlap_fade)
            faded_current = overlap_start * overlap_fade
            overlapped_section = faded_previous + faded_current
            
            # Store new overlap for next chunk
            self.overlap_buffer = audio_chunk[-self.overlap_size:]
            
            # Return reconstructed audio
            if self.overlap_size < self.chunk_size:
                non_overlap = audio_chunk[self.overlap_size:-self.overlap_size]
                return np.concatenate([overlapped_section, non_overlap])
            else:
                return overlapped_section
                
        except Exception as e:
            print(f"Overlap-add reconstruction failed: {e}")
            return audio_chunk  # Return original on failure
    
    def get_output_chunk(self):
        """
        Get next processed audio chunk from output buffer.
        
        Returns:
            dict: Output chunk info, or None if buffer empty
        """
        with self.lock:
            if self.output_buffer:
                return self.output_buffer.popleft()
            return None
    
    def get_continuous_audio(self, duration_seconds=None):
        """
        Get continuous audio stream from output buffer.
        
        Args:
            duration_seconds: Maximum duration to retrieve
            
        Returns:
            numpy array: Continuous audio stream
        """
        with self.lock:
            if not self.output_buffer:
                return np.array([], dtype=np.float32)
            
            audio_segments = []
            total_samples = 0
            max_samples = None
            
            if duration_seconds:
                max_samples = int(duration_seconds * SAMPLE_RATE)
            
            # Concatenate available audio chunks
            while self.output_buffer and (max_samples is None or total_samples < max_samples):
                chunk = self.output_buffer.popleft()
                audio_segments.append(chunk['data'])
                total_samples += len(chunk['data'])
            
            if audio_segments:
                continuous_audio = np.concatenate(audio_segments)
                if max_samples and len(continuous_audio) > max_samples:
                    continuous_audio = continuous_audio[:max_samples]
                return continuous_audio
            
            return np.array([], dtype=np.float32)
    
    def flush_buffers(self):
        """Clear all buffers and reset state"""
        with self.lock:
            self.input_buffer.clear()
            self.output_buffer.clear()
            self.overlap_buffer = np.zeros(self.overlap_size, dtype=np.float32)
            self.reconstruction_buffer = np.zeros(self.chunk_size * 2, dtype=np.float32)
            self.write_position = 0
            
            print("Audio buffers flushed")
    
    def get_buffer_stats(self):
        """Get buffer statistics"""
        with self.lock:
            return {
                'input_chunks': len(self.input_buffer),
                'output_chunks': len(self.output_buffer),
                'chunks_added': self.chunks_added,
                'chunks_processed': self.chunks_processed,
                'underruns': self.underruns,
                'overruns': self.overruns,
                'buffer_health': self._calculate_buffer_health()
            }
    
    def _calculate_buffer_health(self):
        """Calculate buffer health score (0-100)"""
        if self.chunks_added == 0:
            return 100
        
        underrun_ratio = self.underruns / max(1, self.chunks_added)
        overrun_ratio = self.overruns / max(1, self.chunks_added)
        
        # Health decreases with buffer issues
        health = 100 - (underrun_ratio * 50 + overrun_ratio * 30)
        return max(0, min(100, health))

class StreamingAudioProcessor:
    """
    Complete streaming audio processing pipeline with buffer management.
    """
    
    def __init__(self, enhancer=None, chunk_size=CHUNK_SIZE, overlap_size=OVERLAP_SIZE):
        """
        Initialize streaming processor.
        
        Args:
            enhancer: StreamingEnhancer instance
            chunk_size: Audio chunk size
            overlap_size: Overlap between chunks
        """
        self.enhancer = enhancer
        self.buffer = AudioBuffer(chunk_size, overlap_size)
        self.processing_thread = None
        self.running = False
        self.stats_lock = threading.Lock()
        
        # Processing statistics
        self.processing_stats = {
            'chunks_processed': 0,
            'average_processing_time': 0.0,
            'peak_processing_time': 0.0,
            'errors': 0
        }
        
    def start_processing(self):
        """Start background audio processing thread"""
        if not self.enhancer:
            print("No enhancer available for processing")
            return False
        
        if self.running:
            print("Processing already running")
            return False
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        print("Streaming audio processor started")
        return True
    
    def stop_processing(self):
        """Stop background processing and clean up"""
        self.running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        self.buffer.flush_buffers()
        print("Streaming audio processor stopped")
    
    def _processing_loop(self):
        """Main processing loop running in background thread"""
        print("Processing loop started")
        
        while self.running:
            try:
                # Get next input chunk
                chunk_info = self.buffer.get_input_chunk()
                
                if chunk_info is None:
                    # No data available, short sleep
                    time.sleep(0.001)  # 1ms
                    continue
                
                # Process chunk with timing
                start_time = time.time()
                
                enhanced_audio = self.enhancer.enhance_chunk(chunk_info['data'])
                
                processing_time = time.time() - start_time
                
                # Add enhanced audio to output buffer
                self.buffer.add_output_chunk(enhanced_audio, chunk_info['id'])
                
                # Update statistics
                self._update_processing_stats(processing_time)
                
            except Exception as e:
                print(f"Processing loop error: {e}")
                with self.stats_lock:
                    self.processing_stats['errors'] += 1
                
                time.sleep(0.010)  # Brief pause on error
    
    def _update_processing_stats(self, processing_time):
        """Update processing statistics"""
        with self.stats_lock:
            stats = self.processing_stats
            stats['chunks_processed'] += 1
            
            # Update average processing time (exponential moving average)
            alpha = 0.1
            if stats['average_processing_time'] == 0:
                stats['average_processing_time'] = processing_time
            else:
                stats['average_processing_time'] = (
                    alpha * processing_time + 
                    (1 - alpha) * stats['average_processing_time']
                )
            
            # Update peak processing time
            stats['peak_processing_time'] = max(stats['peak_processing_time'], processing_time)
    
    def add_audio_chunk(self, audio_data, timestamp=None):
        """Add audio chunk for processing"""
        return self.buffer.add_input_chunk(audio_data, timestamp)
    
    def get_enhanced_audio(self, duration_seconds=None):
        """Get enhanced audio output"""
        return self.buffer.get_continuous_audio(duration_seconds)
    
    def get_stats(self):
        """Get comprehensive processing statistics"""
        buffer_stats = self.buffer.get_buffer_stats()
        
        with self.stats_lock:
            processing_stats = self.processing_stats.copy()
        
        # Calculate real-time factor
        if processing_stats['chunks_processed'] > 0:
            chunk_duration = CHUNK_SIZE / SAMPLE_RATE
            real_time_factor = processing_stats['average_processing_time'] / chunk_duration
        else:
            real_time_factor = 0.0
        
        return {
            'buffer': buffer_stats,
            'processing': processing_stats,
            'real_time_factor': real_time_factor,
            'is_running': self.running,
            'latency_ms': self._estimate_latency()
        }
    
    def _estimate_latency(self):
        """Estimate total system latency in milliseconds"""
        # Buffer latency + processing latency + overlap latency
        buffer_chunks = len(self.buffer.input_buffer) + len(self.buffer.output_buffer)
        buffer_latency = (buffer_chunks * CHUNK_SIZE / SAMPLE_RATE) * 1000
        
        processing_latency = self.processing_stats['average_processing_time'] * 1000
        overlap_latency = (OVERLAP_SIZE / SAMPLE_RATE) * 1000
        
        total_latency = buffer_latency + processing_latency + overlap_latency
        return total_latency

def create_audio_processor(enhancer=None):
    """
    Factory function to create a streaming audio processor.
    
    Args:
        enhancer: Optional StreamingEnhancer instance
        
    Returns:
        StreamingAudioProcessor instance
    """
    return StreamingAudioProcessor(enhancer)

def test_buffer_performance():
    """Test function to verify buffer performance"""
    print("Testing AudioBuffer performance...")
    
    buffer = AudioBuffer()
    test_chunks = 100
    chunk_size = CHUNK_SIZE
    
    # Generate test audio
    test_audio = np.random.randn(test_chunks * chunk_size).astype(np.float32)
    
    # Test input/output cycle
    start_time = time.time()
    
    for i in range(test_chunks):
        chunk = test_audio[i * chunk_size:(i + 1) * chunk_size]
        buffer.add_input_chunk(chunk)
        
        input_chunk = buffer.get_input_chunk()
        if input_chunk:
            buffer.add_output_chunk(input_chunk['data'])
    
    end_time = time.time()
    duration = end_time - start_time
    
    stats = buffer.get_buffer_stats()
    
    print(f"Processed {test_chunks} chunks in {duration:.3f}s")
    print(f"Average time per chunk: {(duration / test_chunks) * 1000:.2f}ms")
    print(f"Buffer stats: {stats}")
    
    return duration < (test_chunks * 0.032)  # Should be faster than real-time
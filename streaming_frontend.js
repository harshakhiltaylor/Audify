/**
 * Real-time audio streaming frontend for Audify
 * Handles WebRTC audio capture and WebSocket streaming
 */

class AudioStreamer {
    constructor() {
        this.socket = null;
        this.mediaStream = null;
        this.audioContext = null;
        this.mediaRecorder = null;
        this.isRecording = false;
        this.isConnected = false;
        this.audioWorkletNode = null;
        this.gainNode = null;
        
        // Audio configuration
        this.sampleRate = 16000;
        this.chunkSize = 512;
        this.chunkDurationMs = 32; // (512/16000) * 1000
        
        // Buffers
        this.audioBuffer = [];
        this.enhancedBuffer = [];
        
        // UI elements
        this.recordBtn = null;
        this.statusDiv = null;
        this.volumeMeter = null;
        
        // Initialize
        this.initializeUI();
        this.connectSocket();
    }
    
    initializeUI() {
        // Create streaming controls UI
        const streamingSection = document.createElement('div');
        streamingSection.className = 'streaming-section';
        streamingSection.innerHTML = `
            <h2 class="section-title">🎤 Live Enhancement</h2>
            <div class="streaming-controls">
                <button class="btn btn-record" id="recordBtn" onclick="audioStreamer.toggleRecording()">
                    <span id="recordIcon">🎤</span>
                    <span id="recordText">Start Live Enhancement</span>
                </button>
                
                <div class="volume-meter-container">
                    <div class="volume-label">Input Level:</div>
                    <div class="volume-meter" id="volumeMeter">
                        <div class="volume-fill" id="volumeFill"></div>
                    </div>
                </div>
                
                <div class="streaming-status" id="streamingStatus">
                    Ready to stream
                </div>
            </div>
            
            <div class="audio-visualization">
                <canvas id="waveformCanvas" width="400" height="100"></canvas>
            </div>
            
            <div class="enhanced-audio-output" style="display: none;">
                <h3>Enhanced Audio Output</h3>
                <audio id="enhancedOutput" controls autoplay muted></audio>
                <div class="output-controls">
                    <button class="btn btn-secondary" onclick="audioStreamer.toggleOutputMute()">
                        <span id="muteIcon">🔊</span>
                        <span id="muteText">Mute Output</span>
                    </button>
                </div>
            </div>
        `;
        
        // Insert before upload section
        const uploadSection = document.querySelector('.upload-section');
        uploadSection.parentNode.insertBefore(streamingSection, uploadSection);
        
        // Store references
        this.recordBtn = document.getElementById('recordBtn');
        this.statusDiv = document.getElementById('streamingStatus');
        this.volumeMeter = document.getElementById('volumeFill');
        this.canvas = document.getElementById('waveformCanvas');
        this.canvasContext = this.canvas.getContext('2d');
        
        // Initialize visualization
        this.initWaveformVisualization();
    }
    
    connectSocket() {
        // Connect to WebSocket server
        this.socket = io();
        
        this.socket.on('connect', () => {
            this.isConnected = true;
            this.updateStatus('Connected - Ready to stream', 'success');
            console.log('✅ Connected to streaming server');
        });
        
        this.socket.on('disconnect', () => {
            this.isConnected = false;
            this.updateStatus('Disconnected from server', 'error');
            console.log('❌ Disconnected from streaming server');
        });
        
        this.socket.on('streaming_ready', (data) => {
            console.log('Streaming ready:', data);
            this.chunkSize = data.chunk_size;
            this.sampleRate = data.sample_rate;
            this.chunkDurationMs = data.chunk_duration_ms;
        });
        
        this.socket.on('streaming_started', (data) => {
            console.log('Streaming started:', data);
            this.updateStatus('🟢 Live enhancement active', 'processing');
        });
        
        this.socket.on('streaming_stopped', (data) => {
            console.log('Streaming stopped:', data);
            this.updateStatus(`Stopped - Processed ${data.chunks_processed} chunks`, 'success');
        });
        
        this.socket.on('enhanced_chunk', (data) => {
            this.handleEnhancedAudio(data);
        });
        
        this.socket.on('streaming_error', (data) => {
            console.error('Streaming error:', data);
            this.updateStatus(`Error: ${data.error}`, 'error');
            this.stopRecording();
        });
    }
    
    async toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            await this.startRecording();
        }
    }
    
    async startRecording() {
        if (!this.isConnected) {
            this.updateStatus('Not connected to server', 'error');
            return;
        }
        
        try {
            // Request microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: this.sampleRate,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: false, // We'll do this with our model
                    autoGainControl: false
                }
            });
            
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });
            
            // Create audio processing nodes
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            this.gainNode = this.audioContext.createGain();
            this.analyserNode = this.audioContext.createAnalyser();
            
            // Configure analyser for visualization
            this.analyserNode.fftSize = 256;
            this.analyserNode.smoothingTimeConstant = 0.8;
            
            // Connect audio processing chain
            source.connect(this.gainNode);
            this.gainNode.connect(this.analyserNode);
            
            // Create ScriptProcessorNode for audio processing
            this.scriptNode = this.audioContext.createScriptProcessor(this.chunkSize, 1, 1);
            this.scriptNode.onaudioprocess = (event) => {
                this.processAudioChunk(event);
            };
            
            this.gainNode.connect(this.scriptNode);
            this.scriptNode.connect(this.audioContext.destination);
            
            // Start streaming
            this.isRecording = true;
            this.socket.emit('start_streaming');
            
            // Update UI
            this.updateRecordButton(true);
            this.updateStatus('🎤 Recording... Speak now!', 'processing');
            
            // Start visualizations
            this.startVisualization();
            
            console.log('✅ Recording started');
            
        } catch (error) {
            console.error('Error starting recording:', error);
            this.updateStatus(`Microphone error: ${error.message}`, 'error');
        }
    }
    
    stopRecording() {
        if (!this.isRecording) return;
        
        try {
            // Stop media stream
            if (this.mediaStream) {
                this.mediaStream.getTracks().forEach(track => track.stop());
                this.mediaStream = null;
            }
            
            // Close audio context
            if (this.audioContext) {
                if (this.scriptNode) {
                    this.scriptNode.disconnect();
                }
                this.audioContext.close();
                this.audioContext = null;
            }
            
            // Stop streaming
            if (this.isConnected) {
                this.socket.emit('stop_streaming');
            }
            
            this.isRecording = false;
            
            // Update UI
            this.updateRecordButton(false);
            this.updateStatus('Recording stopped', 'success');
            
            // Stop visualizations
            this.stopVisualization();
            
            console.log('✅ Recording stopped');
            
        } catch (error) {
            console.error('Error stopping recording:', error);
        }
    }
    
    processAudioChunk(event) {
        if (!this.isRecording || !this.isConnected) return;
        
        const inputBuffer = event.inputBuffer;
        const audioData = inputBuffer.getChannelData(0);
        
        // Convert to Float32Array and send to server
        const audioArray = new Float32Array(audioData);
        const audioBuffer = audioArray.buffer;
        
        // Send chunk to server for enhancement
        this.socket.emit('audio_chunk', {
            audio: audioBuffer,
            timestamp: Date.now()
        });
        
        // Update volume meter
        this.updateVolumeMeter(audioData);
    }
    
    handleEnhancedAudio(data) {
        // Convert received bytes back to audio
        const audioArray = new Float32Array(data.audio);
        
        // Here you would typically play the enhanced audio
        // For now, we'll just log it
        console.log(`Received enhanced chunk ${data.chunk_id}: ${audioArray.length} samples`);
        
        // TODO: Implement real-time audio playback
        // This requires creating an audio buffer and scheduling playback
    }
    
    updateVolumeMeter(audioData) {
        // Calculate RMS volume
        let sum = 0;
        for (let i = 0; i < audioData.length; i++) {
            sum += audioData[i] * audioData[i];
        }
        const rms = Math.sqrt(sum / audioData.length);
        const volume = Math.min(100, rms * 1000); // Scale to 0-100
        
        // Update volume meter
        if (this.volumeMeter) {
            this.volumeMeter.style.width = `${volume}%`;
        }
    }
    
    initWaveformVisualization() {
        this.waveformData = new Uint8Array(128);
    }
    
    startVisualization() {
        this.visualizationActive = true;
        this.drawWaveform();
    }
    
    stopVisualization() {
        this.visualizationActive = false;
    }
    
    drawWaveform() {
        if (!this.visualizationActive || !this.analyserNode) return;
        
        requestAnimationFrame(() => this.drawWaveform());
        
        this.analyserNode.getByteFrequencyData(this.waveformData);
        
        const ctx = this.canvasContext;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw waveform
        ctx.fillStyle = 'var(--primary-color)';
        const barWidth = width / this.waveformData.length;
        
        for (let i = 0; i < this.waveformData.length; i++) {
            const barHeight = (this.waveformData[i] / 255) * height;
            ctx.fillRect(i * barWidth, height - barHeight, barWidth, barHeight);
        }
    }
    
    updateRecordButton(recording) {
        const icon = document.getElementById('recordIcon');
        const text = document.getElementById('recordText');
        
        if (recording) {
            icon.textContent = '⏹️';
            text.textContent = 'Stop Enhancement';
            this.recordBtn.classList.add('recording');
        } else {
            icon.textContent = '🎤';
            text.textContent = 'Start Live Enhancement';
            this.recordBtn.classList.remove('recording');
        }
    }
    
    updateStatus(message, type) {
        if (this.statusDiv) {
            this.statusDiv.textContent = message;
            this.statusDiv.className = `streaming-status status-${type}`;
        }
    }
    
    toggleOutputMute() {
        const audio = document.getElementById('enhancedOutput');
        const icon = document.getElementById('muteIcon');
        const text = document.getElementById('muteText');
        
        if (audio.muted) {
            audio.muted = false;
            icon.textContent = '🔊';
            text.textContent = 'Mute Output';
        } else {
            audio.muted = true;
            icon.textContent = '🔇';
            text.textContent = 'Unmute Output';
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Check for browser support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error('WebRTC not supported in this browser');
        return;
    }
    
    // Initialize audio streamer
    window.audioStreamer = new AudioStreamer();
});

// Export for global access
window.AudioStreamer = AudioStreamer;
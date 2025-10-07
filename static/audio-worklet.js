/**
 * Web Audio API AudioWorkletProcessor for low-latency real-time audio processing.
 * Handles 128-sample frames and manages buffering for 512-sample chunks.
 */

class AudioStreamProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        
        // Configuration constants
        this.CHUNK_SIZE = 512;  // Target chunk size for processing
        this.QUANTUM_SIZE = 128; // Web Audio quantum size
        this.SAMPLE_RATE = 16000; // Expected sample rate
        this.BUFFER_SIZE = 2048; // Maximum buffer size
        
        // Audio buffers
        this.inputBuffer = new Float32Array(this.BUFFER_SIZE);
        this.outputBuffer = new Float32Array(this.BUFFER_SIZE);
        this.bufferIndex = 0;
        this.outputIndex = 0;
        this.availableOutput = 0;
        
        // Processing state
        this.isProcessing = false;
        this.chunkCounter = 0;
        this.droppedFrames = 0;
        this.lastProcessTime = 0;
        
        // Performance monitoring
        this.stats = {
            framesProcessed: 0,
            chunksGenerated: 0,
            bufferUnderruns: 0,
            bufferOverruns: 0,
            avgLatency: 0,
            maxLatency: 0
        };
        
        // Initialize worklet
        this.initializeProcessor();
        
        console.log('Audio Worklet Processor initialized:', {
            chunkSize: this.CHUNK_SIZE,
            quantumSize: this.QUANTUM_SIZE,
            sampleRate: this.SAMPLE_RATE
        });
    }
    
    initializeProcessor() {
        // Set up message handlers
        this.port.onmessage = (event) => {
            this.handleMessage(event.data);
        };
        
        // Initialize buffers with zeros
        this.inputBuffer.fill(0);
        this.outputBuffer.fill(0);
        
        // Send ready signal
        this.postMessage({
            type: 'worklet_ready',
            timestamp: currentTime
        });
    }
    
    handleMessage(data) {
        try {
            switch (data.type) {
                case 'start_processing':
                    this.startProcessing();
                    break;
                    
                case 'stop_processing':
                    this.stopProcessing();
                    break;
                    
                case 'enhanced_chunk':
                    this.receiveEnhancedChunk(data.audioData);
                    break;
                    
                case 'reset_buffers':
                    this.resetBuffers();
                    break;
                    
                case 'get_stats':
                    this.sendStatistics();
                    break;
                    
                default:
                    console.warn('Unknown message type:', data.type);
            }
        } catch (error) {
            this.postMessage({
                type: 'error',
                error: error.message,
                timestamp: currentTime
            });
        }
    }
    
    startProcessing() {
        this.isProcessing = true;
        this.resetBuffers();
        
        this.postMessage({
            type: 'processing_started',
            timestamp: currentTime
        });
        
        console.log('Audio worklet processing started');
    }
    
    stopProcessing() {
        this.isProcessing = false;
        
        this.postMessage({
            type: 'processing_stopped',
            stats: this.getProcessingStats(),
            timestamp: currentTime
        });
        
        console.log('Audio worklet processing stopped');
    }
    
    resetBuffers() {
        this.inputBuffer.fill(0);
        this.outputBuffer.fill(0);
        this.bufferIndex = 0;
        this.outputIndex = 0;
        this.availableOutput = 0;
        this.chunkCounter = 0;
        
        // Reset stats
        this.stats = {
            framesProcessed: 0,
            chunksGenerated: 0,
            bufferUnderruns: 0,
            bufferOverruns: 0,
            avgLatency: 0,
            maxLatency: 0
        };
    }
    
    process(inputs, outputs, parameters) {
        const input = inputs[0];
        const output = outputs[0];
        
        if (!input || !output || !this.isProcessing) {
            return true; // Keep processor alive
        }
        
        try {
            // Process each channel (assuming mono)
            const inputChannel = input[0];
            const outputChannel = output[0];
            
            if (inputChannel && outputChannel) {
                this.processAudioFrame(inputChannel, outputChannel);
            }
            
            this.stats.framesProcessed++;
            return true; // Continue processing
            
        } catch (error) {
            console.error('Audio worklet process error:', error);
            this.postMessage({
                type: 'error',
                error: error.message,
                timestamp: currentTime
            });
            return true; // Continue despite error
        }
    }
    
    processAudioFrame(inputFrame, outputFrame) {
        const frameSize = inputFrame.length;
        
        // Add input frame to buffer
        this.addToInputBuffer(inputFrame);
        
        // Generate output from enhanced buffer
        this.generateOutputFrame(outputFrame, frameSize);
        
        // Check if we have enough input for a chunk
        if (this.bufferIndex >= this.CHUNK_SIZE) {
            this.processInputChunk();
        }
    }
    
    addToInputBuffer(inputFrame) {
        const frameSize = inputFrame.length;
        
        // Check for buffer overflow
        if (this.bufferIndex + frameSize > this.BUFFER_SIZE) {
            this.stats.bufferOverruns++;
            // Shift buffer to make room
            const shift = frameSize;
            this.inputBuffer.copyWithin(0, shift, this.bufferIndex);
            this.bufferIndex -= shift;
        }
        
        // Copy input frame to buffer
        for (let i = 0; i < frameSize; i++) {
            this.inputBuffer[this.bufferIndex + i] = inputFrame[i];
        }
        this.bufferIndex += frameSize;
    }
    
    processInputChunk() {
        try {
            // Extract chunk from buffer
            const chunk = new Float32Array(this.CHUNK_SIZE);
            for (let i = 0; i < this.CHUNK_SIZE; i++) {
                chunk[i] = this.inputBuffer[i];
            }
            
            // Shift remaining buffer content
            const remainingSize = this.bufferIndex - this.CHUNK_SIZE;
            if (remainingSize > 0) {
                this.inputBuffer.copyWithin(0, this.CHUNK_SIZE, this.bufferIndex);
            }
            this.bufferIndex = remainingSize;
            
            // Send chunk for processing
            this.postMessage({
                type: 'audio_chunk',
                audioData: chunk,
                chunkId: this.chunkCounter,
                timestamp: currentTime
            });
            
            this.chunkCounter++;
            this.stats.chunksGenerated++;
            
        } catch (error) {
            console.error('Chunk processing error:', error);
        }
    }
    
    generateOutputFrame(outputFrame, frameSize) {
        // Fill output frame from enhanced buffer
        for (let i = 0; i < frameSize; i++) {
            if (this.availableOutput > 0) {
                // Use enhanced audio if available
                outputFrame[i] = this.outputBuffer[this.outputIndex];
                this.outputIndex = (this.outputIndex + 1) % this.BUFFER_SIZE;
                this.availableOutput--;
            } else {
                // No enhanced audio available - buffer underrun
                outputFrame[i] = 0;
                if (i === 0) { // Count once per frame
                    this.stats.bufferUnderruns++;
                }
            }
        }
    }
    
    receiveEnhancedChunk(audioData) {
        try {
            // Ensure audioData is Float32Array
            const enhancedChunk = audioData instanceof Float32Array ? 
                audioData : new Float32Array(audioData);
            
            const chunkSize = enhancedChunk.length;
            
            // Check if there's room in output buffer
            if (this.availableOutput + chunkSize > this.BUFFER_SIZE) {
                console.warn('Output buffer full, dropping enhanced chunk');
                return;
            }
            
            // Add enhanced chunk to output buffer
            let writeIndex = (this.outputIndex + this.availableOutput) % this.BUFFER_SIZE;
            
            for (let i = 0; i < chunkSize; i++) {
                this.outputBuffer[writeIndex] = enhancedChunk[i];
                writeIndex = (writeIndex + 1) % this.BUFFER_SIZE;
            }
            
            this.availableOutput += chunkSize;
            
            // Calculate latency
            const latency = currentTime - this.lastProcessTime;
            if (latency > 0) {
                this.updateLatencyStats(latency);
            }
            this.lastProcessTime = currentTime;
            
        } catch (error) {
            console.error('Enhanced chunk processing error:', error);
        }
    }
    
    updateLatencyStats(latency) {
        const latencyMs = latency * 1000; // Convert to milliseconds
        
        // Update maximum latency
        if (latencyMs > this.stats.maxLatency) {
            this.stats.maxLatency = latencyMs;
        }
        
        // Update average latency (exponential moving average)
        if (this.stats.avgLatency === 0) {
            this.stats.avgLatency = latencyMs;
        } else {
            this.stats.avgLatency = 0.9 * this.stats.avgLatency + 0.1 * latencyMs;
        }
    }
    
    getProcessingStats() {
        const totalFrames = this.stats.framesProcessed;
        
        return {
            ...this.stats,
            bufferHealth: this.calculateBufferHealth(),
            outputBufferUsage: (this.availableOutput / this.BUFFER_SIZE) * 100,
            inputBufferUsage: (this.bufferIndex / this.BUFFER_SIZE) * 100,
            underrunRate: totalFrames > 0 ? (this.stats.bufferUnderruns / totalFrames) * 100 : 0,
            overrunRate: totalFrames > 0 ? (this.stats.bufferOverruns / totalFrames) * 100 : 0,
            chunksPerSecond: this.stats.chunksGenerated / (currentTime > 0 ? currentTime : 1)
        };
    }
    
    calculateBufferHealth() {
        // Buffer health score (0-100)
        let health = 100;
        
        // Penalize buffer issues
        const totalFrames = this.stats.framesProcessed;
        if (totalFrames > 0) {
            const underrunRate = this.stats.bufferUnderruns / totalFrames;
            const overrunRate = this.stats.bufferOverruns / totalFrames;
            
            health -= underrunRate * 50; // Underruns are critical
            health -= overrunRate * 30;  // Overruns are problematic
        }
        
        // Factor in buffer usage
        const outputUsage = (this.availableOutput / this.BUFFER_SIZE);
        if (outputUsage < 0.1) { // Too empty
            health -= 20;
        } else if (outputUsage > 0.9) { // Too full
            health -= 15;
        }
        
        return Math.max(0, Math.min(100, health));
    }
    
    sendStatistics() {
        this.postMessage({
            type: 'statistics',
            stats: this.getProcessingStats(),
            timestamp: currentTime
        });
    }
    
    postMessage(message) {
        try {
            this.port.postMessage(message);
        } catch (error) {
            console.error('Failed to post message:', error);
        }
    }
    
    // Utility method to detect if processing is active
    static get parameterDescriptors() {
        return [];
    }
}

// Register the audio worklet processor
registerProcessor('audio-stream-processor', AudioStreamProcessor);

// Export for testing (if needed)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AudioStreamProcessor;
}
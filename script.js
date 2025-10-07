// // Global variables
// let selectedFile = null;
// let processingId = null;
// let enhancedFilename = null;
// let displayedProgress = 0;
// const API_BASE = ''; // Empty string to use same origin as Flask server

// // Theme management
// function toggleTheme() {
//     const body = document.body;
//     const themeIcon = document.getElementById('theme-icon');
//     const themeText = document.getElementById('theme-text');

//     if (body.getAttribute('data-theme') === 'dark') {
//         body.removeAttribute('data-theme');
//         themeIcon.textContent = '🌑';
//         themeText.textContent = 'Dark Mode';
//         localStorage.setItem('theme', 'light');
//     } else {
//         body.setAttribute('data-theme', 'dark');
//         themeIcon.textContent = '☀️';
//         themeText.textContent = 'Light Mode';
//         localStorage.setItem('theme', 'dark');
//     }
// }

// // Initialize theme
// function initTheme() {
//     const savedTheme = localStorage.getItem('theme');
//     if (savedTheme === 'dark') {
//         document.body.setAttribute('data-theme', 'dark');
//         document.getElementById('theme-icon').textContent = '☀️';
//         document.getElementById('theme-text').textContent = 'Light Mode';
//     }
// }

// // File upload handling
// function initFileUpload() {
//     const uploadArea = document.getElementById('uploadArea');
//     const fileInput = document.getElementById('fileInput');

//     fileInput.addEventListener('click', e => e.stopPropagation());
//     // Click to upload
//     uploadArea.addEventListener('click', () => fileInput.click());

//     // File selection
//     fileInput.addEventListener('change', handleFileSelect);

//     // Drag and drop
//     uploadArea.addEventListener('dragover', (e) => {
//         e.preventDefault();
//         uploadArea.classList.add('dragover');
//     });

//     uploadArea.addEventListener('dragleave', () => {
//         uploadArea.classList.remove('dragover');
//     });

//     uploadArea.addEventListener('drop', (e) => {
//         e.preventDefault();
//         uploadArea.classList.remove('dragover');
//         const files = e.dataTransfer.files;
//         if (files.length > 0) {
//             handleFileSelect({ target: { files } });
//         }
//     });
// }

// function handleFileSelect(event) {
//     const file = event.target.files[0];
//     // if (!file) return;

//     selectedFile = file;

//     // Show file info
//     document.getElementById('fileName').textContent = file.name;
//     document.getElementById('fileSize').textContent = formatFileSize(file.size);
//     document.getElementById('fileInfo').style.display = 'flex';

//     // Enable enhance button
//     document.getElementById('enhanceBtn').disabled = false;

//     // Show original audio
//     const originalAudio = document.getElementById('originalAudio');
//     originalAudio.src = URL.createObjectURL(file);
//     document.getElementById('originalAudioSection').style.display = 'block';

//     // Update duration when loaded
//     originalAudio.addEventListener('loadedmetadata', () => {
//         document.getElementById('originalDuration').textContent = formatDuration(originalAudio.duration);
//     });
// }

// function formatFileSize(bytes) {
//     if (bytes === 0) return '0 Bytes';
//     const k = 1024;
//     const sizes = ['Bytes', 'KB', 'MB', 'GB'];
//     const i = Math.floor(Math.log(bytes) / Math.log(k));
//     return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
// }

// function formatDuration(seconds) {
//     const minutes = Math.floor(seconds / 60);
//     const remainingSeconds = Math.floor(seconds % 60);
//     return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
// }

// // Audio enhancement
// async function enhanceAudio() {
//     displayedProgress = 0;
//     updateProgress(0);
//     if (!selectedFile) return;

//     const enhanceBtn = document.getElementById('enhanceBtn');
//     const progressContainer = document.getElementById('progressContainer');
//     const statusMessage = document.getElementById('statusMessage');

//     // Disable button and show progress
//     enhanceBtn.disabled = true;
//     enhanceBtn.innerHTML = '<span class="spinning">⚙️</span> Processing...';
//     progressContainer.style.display = 'block';

//     try {
//         // Upload file
//         const formData = new FormData();
//         formData.append('audio', selectedFile);

//         const response = await fetch(`${API_BASE}/enhance`, {
//             method: 'POST',
//             body: formData
//         });

//         console.log('Response:', response);

//         if (!response.ok) {
//             throw new Error(`HTTP error! status: ${response.status}`);
//         }

//         const result = await response.json();

//         if (result.success) {
//             console.log('Enhancement result:', result);
//             processingId = result.processing_id;
//             showStatus('Processing audio...', 'processing');

//             // Start polling for status
//             pollProcessingStatus();
//         } else {
//             throw new Error(result.error || 'Upload failed');
//         }

//     } catch (error) {
//         console.error('Error:', error);
//         showStatus(`Error: ${error.message}`, 'error');
//         resetUI();
//     }
// }

// async function pollProcessingStatus() {
//     if (!processingId) return;

//     try {
//         const response = await fetch(`${API_BASE}/status/${processingId}`);
//         const status = await response.json();

//         if (status.status === 'completed') {
//             console.log('Processing completed:', status);
//             handleProcessingComplete(status.result);
//         } else if (status.status === 'error') {
//             throw new Error(status.error || 'Processing failed');
//         } else {
//             // Update progress
//             updateProgress(status.progress);

//             // Continue polling
//             setTimeout(pollProcessingStatus, 1000);
//         }

//     } catch (error) {
//         console.error('Error polling status:', error);
//         showStatus(`Error: ${error.message}`, 'error');
//         resetUI();
//     }
// }

// function handleProcessingComplete(result) {
//     console.log('Processing result:', result);
//     if (result.success) {
//         enhancedFilename = result.output_filename;

//         // Show enhanced audio
//         const enhancedAudio = document.getElementById('enhancedAudio');
//         enhancedAudio.src = `${API_BASE}/outputs/${enhancedFilename}`;
//         console.log('Enhanced audio URL:', enhancedAudio.src);
//         document.getElementById('enhancedAudioSection').style.display = 'block';

//         // Update duration when loaded
//         enhancedAudio.addEventListener('loadedmetadata', () => {
//             document.getElementById('enhancedDuration').textContent = formatDuration(enhancedAudio.duration);
//         });

//         // Show metrics
//         // if (result.metrics) {
//         //     showMetrics(result.metrics);
//         // }

//         document.getElementById('resultsSection').style.display = 'block';

//         showStatus('Enhancement completed successfully!', 'success');
//         updateProgress(100);

//     } else {
//         throw new Error(result.error || 'Enhancement failed');
//     }

//     resetUI();
// }

// function showMetrics(metrics) {
//     const metricsGrid = document.getElementById('metricsGrid');
//     //     if (!metricsGrid) {
//     //     // nothing to do
//     //     return;
//     //   }
//     metricsGrid.innerHTML = '';

//     // Define metric display configurations
//     const metricConfigs = {
//         'segmental_snr': { label: 'Segmental SNR', unit: 'dB', decimals: 2 },
//         'pesq': { label: 'PESQ Score', unit: '', decimals: 2 },
//         'stoi': { label: 'STOI Score', unit: '', decimals: 3 },
//         'snr': { label: 'SNR', unit: 'dB', decimals: 2 },
//         'signal_length': { label: 'Duration', unit: 's', decimals: 2 },
//         'signal_rms': { label: 'RMS Level', unit: '', decimals: 4 }
//     };

//     Object.entries(metrics).forEach(([key, value]) => {
//         if (metricConfigs[key]) {
//             const config = metricConfigs[key];
//             const metricCard = document.createElement('div');
//             metricCard.className = 'metric-card';
//             metricCard.innerHTML = `
//                         <div class="metric-value">${value.toFixed(config.decimals)}${config.unit}</div>
//                         <div class="metric-label">${config.label}</div>
//                     `;
//             metricsGrid.appendChild(metricCard);
//         }
//     });

//     document.getElementById('metricsSection').style.display = 'block';
// }

// function updateProgress(newProgress) {
//     const progressFill = document.getElementById('progressFill');
//     const progressText = document.getElementById('progressText');

//     // If the new target is lower, just jump
//     if (newProgress <= displayedProgress) {
//         displayedProgress = newProgress;
//         progressFill.style.width = `${displayedProgress}%`;
//         progressText.textContent = `Processing... ${displayedProgress}%`;
//         return;
//     }

//     // Otherwise, animate a little at a time
//     const start = displayedProgress;
//     const end = newProgress;
//     const duration = 500;              // total ms to animate between steps
//     const stepTime = 20;               // update every 20ms
//     const steps = Math.ceil(duration / stepTime);
//     const delta = (end - start) / steps;
//     let current = start;
//     let count = 0;

//     const iv = setInterval(() => {
//         count++;
//         current += delta;
//         if (count >= steps) {
//             current = end;
//             clearInterval(iv);
//         }
//         displayedProgress = Math.round(current);
//         progressFill.style.width = `${displayedProgress}%`;
//         progressText.textContent = `Processing... ${displayedProgress}%`;
//     }, stepTime);
// }

// function showStatus(message, type) {
//     const statusMessage = document.getElementById('statusMessage');
//     statusMessage.textContent = message;
//     statusMessage.className = `status-message status-${type}`;
//     statusMessage.style.display = 'block';
// }

// // const player = new Plyr('#player', {
// //       controls: ['play', 'progress', 'current-time', 'mute', 'volume']
// //     });

// function resetUI() {
//     const enhanceBtn = document.getElementById('enhanceBtn');
//     enhanceBtn.disabled = false;
//     enhanceBtn.innerHTML = '<span>🚀</span> Enhance Audio';

//     setTimeout(() => {
//         document.getElementById('progressContainer').style.display = 'none';
//     }, 2000);
// }

// async function downloadEnhanced() {
//     if (!enhancedFilename) return;

//     try {
//         const response = await fetch(`${API_BASE}/outputs/${enhancedFilename}`);
//         console.log('Download response:', response);
//         if (!response.ok) throw new Error('Download failed');

//         const blob = await response.blob();
//         const url = window.URL.createObjectURL(blob);
//         const a = document.createElement('a');
//         a.href = url;
//         a.download = enhancedFilename;
//         document.body.appendChild(a);
//         a.click();
//         window.URL.revokeObjectURL(url);
//         document.body.removeChild(a);

//     } catch (error) {
//         console.error('Download error:', error);
//         showStatus('Download failed. Please try again.', 'error');
//     }
// }

// // Initialize app
// document.addEventListener('DOMContentLoaded', () => {
//     initTheme();
//     initFileUpload();

//     // Test API connection
//     fetch(`${API_BASE}/health`)
//         .then(response => response.json())
//         .then(data => {
//             console.log('API connected:', data);
//         })
//         .catch(error => {
//             console.error('API connection error:', error);
//             showStatus('Unable to connect to the enhancement server. Please ensure the backend is running.', 'error');
//         });
// });





// Global variables
let selectedFile = null;
let processingId = null;
let enhancedFilename = null;
let displayedProgress = 0;
const API_BASE = ''; // Empty string to use same origin as Flask server

// Streaming status tracking
let streamingStatus = {
    isStreaming: false,
    isConnected: false,
    connectionAttempts: 0,
    maxRetries: 3
};

// Theme management
function toggleTheme() {
    const body = document.body;
    const themeIcon = document.getElementById('theme-icon');
    const themeText = document.getElementById('theme-text');

    if (body.getAttribute('data-theme') === 'dark') {
        body.removeAttribute('data-theme');
        themeIcon.textContent = '🌑';
        themeText.textContent = 'Dark Mode';
        localStorage.setItem('theme', 'light');
    } else {
        body.setAttribute('data-theme', 'dark');
        themeIcon.textContent = '☀️';
        themeText.textContent = 'Light Mode';
        localStorage.setItem('theme', 'dark');
    }
}

// Initialize theme
function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.setAttribute('data-theme', 'dark');
        document.getElementById('theme-icon').textContent = '☀️';
        document.getElementById('theme-text').textContent = 'Light Mode';
    }
}

// Streaming status management
function updateStreamingStatus(message, type = 'info') {
    const statusElement = document.getElementById('streamingStatus');
    if (statusElement) {
        statusElement.textContent = message;
        statusElement.className = `streaming-status status-${type}`;
        console.log(`🎤 Streaming: ${message}`);
    }
}

function updateStreamingUI(isStreaming) {
    const recordBtn = document.getElementById('recordBtn');
    const recordIcon = document.getElementById('recordIcon');
    const recordText = document.getElementById('recordText');
    
    if (!recordBtn || !recordIcon || !recordText) return;
    
    if (isStreaming) {
        recordBtn.classList.add('recording');
        recordIcon.textContent = '⏹️';
        recordText.textContent = 'Stop Enhancement';
        recordBtn.style.background = 'linear-gradient(135deg, #EF4444 0%, #DC2626 100%)';
    } else {
        recordBtn.classList.remove('recording');
        recordIcon.textContent = '🎤';
        recordText.textContent = 'Start Live Enhancement';
        recordBtn.style.background = '';
    }
}

function showStreamingConnection() {
    updateStreamingStatus('Connecting to server...', 'processing');
}

function showStreamingReady() {
    streamingStatus.isConnected = true;
    streamingStatus.connectionAttempts = 0;
    updateStreamingStatus('Ready for live enhancement', 'success');
}

function showStreamingError(error) {
    console.error('❌ Streaming error:', error);
    streamingStatus.isConnected = false;
    
    if (streamingStatus.connectionAttempts < streamingStatus.maxRetries) {
        streamingStatus.connectionAttempts++;
        updateStreamingStatus(`Connection error, retrying... (${streamingStatus.connectionAttempts}/${streamingStatus.maxRetries})`, 'error');
        
        // Retry connection after delay
        setTimeout(() => {
            if (window.audioStreamer) {
                window.audioStreamer.reconnect();
            }
        }, 2000 * streamingStatus.connectionAttempts);
    } else {
        updateStreamingStatus('Streaming unavailable - please refresh to retry', 'error');
    }
}

function showStreamingStarted() {
    streamingStatus.isStreaming = true;
    updateStreamingStatus('Live enhancement active', 'processing');
    updateStreamingUI(true);
}

function showStreamingStopped() {
    streamingStatus.isStreaming = false;
    updateStreamingStatus('Live enhancement stopped', 'info');
    updateStreamingUI(false);
}

// Integration with window.audioStreamer
function initStreamingIntegration() {
    // Wait for audioStreamer to be available
    const checkAudioStreamer = () => {
        if (window.audioStreamer) {
            console.log('✅ AudioStreamer integration ready');
            
            // Set up event callbacks
            window.audioStreamer.onStatusChange = (status, data) => {
                switch (status) {
                    case 'connecting':
                        showStreamingConnection();
                        break;
                    case 'connected':
                        showStreamingReady();
                        break;
                    case 'error':
                        showStreamingError(data?.error || 'Connection error');
                        break;
                    case 'streaming_started':
                        showStreamingStarted();
                        break;
                    case 'streaming_stopped':
                        showStreamingStopped();
                        break;
                    case 'disconnected':
                        streamingStatus.isConnected = false;
                        updateStreamingStatus('Disconnected from server', 'error');
                        updateStreamingUI(false);
                        break;
                }
            };
            
            // Set up volume level callback
            window.audioStreamer.onVolumeLevel = (level) => {
                updateVolumeLevel(level);
            };
            
            // Set up enhanced audio callback
            window.audioStreamer.onEnhancedAudio = (audioBuffer) => {
                playEnhancedAudio(audioBuffer);
            };
            
        } else {
            // Retry after short delay
            setTimeout(checkAudioStreamer, 100);
        }
    };
    
    checkAudioStreamer();
}

function updateVolumeLevel(level) {
    const volumeFill = document.getElementById('volumeFill');
    if (volumeFill) {
        // Convert level (0-1) to percentage
        const percentage = Math.min(100, Math.max(0, level * 100));
        volumeFill.style.width = `${percentage}%`;
        
        // Color coding based on level
        if (percentage > 80) {
            volumeFill.style.background = 'linear-gradient(90deg, #EF4444, #DC2626)';
        } else if (percentage > 50) {
            volumeFill.style.background = 'linear-gradient(90deg, #F59E0B, #D97706)';
        } else {
            volumeFill.style.background = 'linear-gradient(90deg, #10B981, #059669)';
        }
    }
}

function playEnhancedAudio(audioBuffer) {
    try {
        const enhancedPlayer = document.getElementById('enhancedOutputPlayer');
        const enhancedOutput = document.getElementById('enhancedOutput');
        
        if (enhancedPlayer && enhancedOutput) {
            // Convert Float32Array to blob for playback
            const blob = new Blob([audioBuffer], { type: 'audio/wav' });
            const url = URL.createObjectURL(blob);
            
            enhancedPlayer.src = url;
            enhancedOutput.style.display = 'block';
            
            // Clean up URL after playing
            enhancedPlayer.addEventListener('ended', () => {
                URL.revokeObjectURL(url);
            }, { once: true });
        }
    } catch (error) {
        console.error('❌ Error playing enhanced audio:', error);
    }
}

// File upload handling
function initFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    if (!uploadArea || !fileInput) return;

    fileInput.addEventListener('click', e => e.stopPropagation());
    
    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());

    // File selection
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect({ target: { files } });
        }
    });
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    selectedFile = file;

    // Show file info
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const fileInfo = document.getElementById('fileInfo');
    
    if (fileName) fileName.textContent = file.name;
    if (fileSize) fileSize.textContent = formatFileSize(file.size);
    if (fileInfo) fileInfo.style.display = 'flex';

    // Enable enhance button
    const enhanceBtn = document.getElementById('enhanceBtn');
    if (enhanceBtn) enhanceBtn.disabled = false;

    // Show original audio
    const originalAudio = document.getElementById('originalAudio');
    const originalAudioSection = document.getElementById('originalAudioSection');
    
    if (originalAudio && originalAudioSection) {
        originalAudio.src = URL.createObjectURL(file);
        originalAudioSection.style.display = 'block';

        // Update duration when loaded
        originalAudio.addEventListener('loadedmetadata', () => {
            const originalDuration = document.getElementById('originalDuration');
            if (originalDuration) {
                originalDuration.textContent = formatDuration(originalAudio.duration);
            }
        });
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// Audio enhancement (batch processing)
async function enhanceAudio() {
    displayedProgress = 0;
    updateProgress(0);
    
    if (!selectedFile) {
        showStatus('Please select an audio file first', 'error');
        return;
    }

    const enhanceBtn = document.getElementById('enhanceBtn');
    const progressContainer = document.getElementById('progressContainer');

    // Disable button and show progress
    if (enhanceBtn) {
        enhanceBtn.disabled = true;
        enhanceBtn.innerHTML = '<span class="spinning">⚙️</span> Processing...';
    }
    
    if (progressContainer) {
        progressContainer.style.display = 'block';
    }

    try {
        // Upload file
        const formData = new FormData();
        formData.append('audio', selectedFile);

        const response = await fetch(`${API_BASE}/enhance`, {
            method: 'POST',
            body: formData
        });

        console.log('✅ Enhancement response:', response);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        if (result.success) {
            console.log('✅ Enhancement started:', result);
            processingId = result.processing_id;
            showStatus('Processing audio...', 'processing');

            // Start polling for status
            pollProcessingStatus();
        } else {
            throw new Error(result.error || 'Upload failed');
        }

    } catch (error) {
        console.error('❌ Enhancement error:', error);
        showStatus(`Error: ${error.message}`, 'error');
        resetUI();
    }
}

async function pollProcessingStatus() {
    if (!processingId) return;

    try {
        const response = await fetch(`${API_BASE}/status/${processingId}`);
        const status = await response.json();

        if (status.status === 'completed') {
            console.log('✅ Processing completed:', status);
            handleProcessingComplete(status.result);
        } else if (status.status === 'error') {
            throw new Error(status.error || 'Processing failed');
        } else {
            // Update progress
            updateProgress(status.progress || 0);

            // Continue polling
            setTimeout(pollProcessingStatus, 1000);
        }

    } catch (error) {
        console.error('❌ Status polling error:', error);
        showStatus(`Error: ${error.message}`, 'error');
        resetUI();
    }
}

function handleProcessingComplete(result) {
    console.log('✅ Processing result:', result);
    
    if (result && result.success) {
        enhancedFilename = result.output_filename;

        // Show enhanced audio
        const enhancedAudio = document.getElementById('enhancedAudio');
        const enhancedAudioSection = document.getElementById('enhancedAudioSection');
        const resultsSection = document.getElementById('resultsSection');
        
        if (enhancedAudio && enhancedFilename) {
            enhancedAudio.src = `${API_BASE}/outputs/${enhancedFilename}`;
            console.log('✅ Enhanced audio URL:', enhancedAudio.src);
            
            if (enhancedAudioSection) {
                enhancedAudioSection.style.display = 'block';
            }

            // Update duration when loaded
            enhancedAudio.addEventListener('loadedmetadata', () => {
                const enhancedDuration = document.getElementById('enhancedDuration');
                if (enhancedDuration) {
                    enhancedDuration.textContent = formatDuration(enhancedAudio.duration);
                }
            });
        }

        // Show metrics if available
        if (result.metrics) {
            showMetrics(result.metrics);
        }

        if (resultsSection) {
            resultsSection.style.display = 'block';
        }

        showStatus('Enhancement completed successfully!', 'success');
        updateProgress(100);

    } else {
        throw new Error(result?.error || 'Enhancement failed');
    }

    resetUI();
}

function showMetrics(metrics) {
    const metricsGrid = document.getElementById('metricsGrid');
    const metricsSection = document.getElementById('metricsSection');
    
    if (!metricsGrid) return;

    metricsGrid.innerHTML = '';

    // Define metric display configurations
    const metricConfigs = {
        'segmental_snr': { label: 'Segmental SNR', unit: 'dB', decimals: 2 },
        'pesq': { label: 'PESQ Score', unit: '', decimals: 2 },
        'stoi': { label: 'STOI Score', unit: '', decimals: 3 },
        'snr': { label: 'SNR', unit: 'dB', decimals: 2 },
        'signal_length': { label: 'Duration', unit: 's', decimals: 2 },
        'signal_rms': { label: 'RMS Level', unit: '', decimals: 4 }
    };

    Object.entries(metrics).forEach(([key, value]) => {
        if (metricConfigs[key] && typeof value === 'number') {
            const config = metricConfigs[key];
            const metricCard = document.createElement('div');
            metricCard.className = 'metric-card';
            metricCard.innerHTML = `
                <div class="metric-value">${value.toFixed(config.decimals)}${config.unit}</div>
                <div class="metric-label">${config.label}</div>
            `;
            metricsGrid.appendChild(metricCard);
        }
    });

    if (metricsSection) {
        metricsSection.style.display = 'block';
    }
}

function updateProgress(newProgress) {
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');

    // Ensure progress is within bounds
    newProgress = Math.min(100, Math.max(0, newProgress));

    // If the new target is lower or equal, just set it
    if (newProgress <= displayedProgress) {
        displayedProgress = newProgress;
        if (progressFill) progressFill.style.width = `${displayedProgress}%`;
        if (progressText) progressText.textContent = `Processing... ${displayedProgress}%`;
        return;
    }

    // Otherwise, animate smoothly
    const start = displayedProgress;
    const end = newProgress;
    const duration = 500;
    const stepTime = 20;
    const steps = Math.ceil(duration / stepTime);
    const delta = (end - start) / steps;
    let current = start;
    let count = 0;

    const interval = setInterval(() => {
        count++;
        current += delta;
        if (count >= steps) {
            current = end;
            clearInterval(interval);
        }
        
        displayedProgress = Math.round(current);
        if (progressFill) progressFill.style.width = `${displayedProgress}%`;
        if (progressText) progressText.textContent = `Processing... ${displayedProgress}%`;
    }, stepTime);
}

function showStatus(message, type) {
    const statusMessage = document.getElementById('statusMessage');
    if (statusMessage) {
        statusMessage.textContent = message;
        statusMessage.className = `status-message status-${type}`;
        statusMessage.style.display = 'block';
    }
    console.log(`📱 Status (${type}): ${message}`);
}

function resetUI() {
    const enhanceBtn = document.getElementById('enhanceBtn');
    if (enhanceBtn) {
        enhanceBtn.disabled = false;
        enhanceBtn.innerHTML = '<span>🚀</span> Enhance Audio';
    }

    // Hide progress after delay
    setTimeout(() => {
        const progressContainer = document.getElementById('progressContainer');
        if (progressContainer) {
            progressContainer.style.display = 'none';
        }
    }, 2000);
}

async function downloadEnhanced() {
    if (!enhancedFilename) {
        showStatus('No enhanced audio available for download', 'error');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/outputs/${enhancedFilename}`);
        console.log('✅ Download response:', response);
        
        if (!response.ok) throw new Error('Download failed');

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = enhancedFilename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        showStatus('Download started successfully', 'success');

    } catch (error) {
        console.error('❌ Download error:', error);
        showStatus('Download failed. Please try again.', 'error');
    }
}

// Connection health monitoring
function monitorConnection() {
    setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/health`);
            if (!response.ok) {
                throw new Error('Health check failed');
            }
        } catch (error) {
            console.warn('⚠️ Connection health check failed:', error);
            // Could add UI indicator here
        }
    }, 30000); // Check every 30 seconds
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    console.log('🎵 Initializing Audify application...');
    
    initTheme();
    initFileUpload();
    initStreamingIntegration();
    
    // Monitor connection health
    monitorConnection();

    // Test API connection
    fetch(`${API_BASE}/health`)
        .then(response => response.json())
        .then(data => {
            console.log('✅ API connected:', data);
            showStatus('System ready', 'success');
            
            // Hide status after delay
            setTimeout(() => {
                const statusMessage = document.getElementById('statusMessage');
                if (statusMessage) {
                    statusMessage.style.display = 'none';
                }
            }, 3000);
        })
        .catch(error => {
            console.error('❌ API connection error:', error);
            showStatus('Unable to connect to the enhancement server. Please ensure the backend is running.', 'error');
        });
});
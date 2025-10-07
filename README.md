# Audify - AI Speech Enhancement Tool

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A powerful web-based AI speech enhancement tool that transforms noisy audio recordings into crystal-clear speech using deep learning. Built with TensorFlow and Flask, Audify provides real-time audio processing with comprehensive quality metrics.

![Audify Demo](demo.gif) <!-- Add your demo gif here -->

## Features

- **Deep Learning AI**: Advanced neural network trained for superior noise reduction
- **Lightning Fast**: Optimized processing pipeline delivers results in seconds
- **Precision Enhancement**: Preserves natural speech while removing noise
- **Quality Metrics**: Comprehensive evaluation with PESQ, STOI, and SNR metrics
- **Web Interface**: User-friendly drag-and-drop interface
- **Multiple Formats**: Support for WAV, MP3, FLAC, OGG, M4A
- **Responsive Design**: Works on desktop and mobile devices
- **Dark/Light Mode**: Theme switching for better user experience

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/[your-username]/audify.git
   cd audify
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare training data**
   
   Create the following directory structure:
   ```
   dataset/
   ├── clean/     # Clean speech audio files
   └── noisy/     # Corresponding noisy versions
   ```
   
   > **Note**: Ensure clean and noisy files have matching filenames (e.g., `speech1.wav` in both folders)

4. **Train the model**
   ```bash
   python backend/train.py
   ```

5. **Start the application**
   ```bash
   python backend/app.py
   ```

6. **Access the application**
   
   Open your browser and navigate to `http://localhost:5000`

## Architecture

### System Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌───────────────────┐
│   Frontend      │    │   Flask Backend  │    │   ML Pipeline     │
│   (HTML/JS/CSS) │◄──►│   API Server     │◄──►│   TensorFlow      │
│                 │    │                  │    │   Audio Processing│
└─────────────────┘    └──────────────────┘    └───────────────────┘
```

### Model Architecture

- **Type**: Frame-wise Deep Neural Network (DNN)
- **Input**: Magnitude spectrogram (dB scale)
- **Architecture**: 
  - Input Layer: Spectrogram frames
  - Hidden Layer: 1024 neurons with ReLU activation
  - Output Layer: Enhanced spectrogram frames
- **Training**: Mean Squared Error (MSE) loss with Adam optimizer

### Audio Processing Pipeline

1. **Feature Extraction**: STFT analysis with librosa
2. **Preprocessing**: Magnitude spectrogram conversion to dB scale
3. **Normalization**: Z-score normalization using training statistics
4. **Enhancement**: Frame-wise neural network processing
5. **Reconstruction**: ISTFT synthesis with original phase information
6. **Post-processing**: Butterworth low-pass filtering

## Project Structure

```
audify/
├── backend/
│   ├── api.py              # Flask API endpoints
│   ├── app.py              # Main Flask application
│   ├── train.py            # Model training script
│   ├── data/
│   │   └── features.py     # Audio feature extraction
│   ├── metrics/
│   │   └── quality.py      # Quality evaluation metrics
│   └── models/
│       └── frame_model.py  # Neural network model
├── dataset/
│   ├── clean/              # Clean audio files
│   └── noisy/              # Noisy audio files
├── temp/                   # Temporary upload storage
├── outputs/                # Enhanced audio outputs
├── index.html              # Frontend HTML
├── script.js               # Frontend JavaScript
├── style.css               # Frontend styling
└── requirements.txt        # Python dependencies
```

## API Reference

### Enhancement Endpoint

**POST** `/enhance`

Upload and enhance an audio file.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `audio` (file)

**Response:**
```json
{
  "success": true,
  "processing_id": "uuid-string"
}
```

### Status Endpoint

**GET** `/status/<processing_id>`

Check processing status and retrieve results.

**Response:**
```json
{
  "status": "completed",
  "progress": 100,
  "result": {
    "success": true,
    "output_filename": "enhanced_uuid.wav",
    "metrics": {
      "segmental_snr": 15.2,
      "pesq": 2.8,
      "stoi": 0.92
    }
  }
}
```

## Quality Metrics

- **PESQ (Perceptual Evaluation of Speech Quality)**: Range 1.0-4.5, higher is better
- **STOI (Short-Time Objective Intelligibility)**: Range 0.0-1.0, higher is better  
- **Segmental SNR**: Signal-to-Noise Ratio in dB, higher is better

## Training Your Own Model

### Dataset Requirements

- **Format**: WAV, MP3, or FLAC files
- **Sample Rate**: 16 kHz (automatically resampled)
- **Duration**: Recommended 10-30 seconds per file
- **Pairs**: Each clean file must have a corresponding noisy version

### Training Process

```bash
# Place your data in the correct directories
dataset/clean/speech001.wav
dataset/noisy/speech001.wav
# ... more pairs

# Train the model
python backend/train.py
```

### Training Parameters

- **Epochs**: 10 (configurable)
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error

## Development

### Running in Development Mode

```bash
# Enable debug mode
export FLASK_ENV=development
python backend/app.py
```

### Adding New Features

1. **Backend**: Extend `api.py` for new endpoints
2. **Frontend**: Modify `script.js` for new UI features
3. **Models**: Update `frame_model.py` for model improvements

## Requirements

### Python Dependencies

```txt
flask>=2.3.0
flask-cors>=4.0.0
tensorflow>=2.13.0
librosa>=0.10.0
soundfile>=0.12.0
scipy>=1.11.0
numpy>=1.24.0
pesq>=0.0.4
pystoi>=0.3.3
```

### System Requirements

- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 2GB free space for models and temporary files
- **CPU**: Multi-core processor recommended for faster processing

## Deployment

### Local Deployment

```bash
python backend/app.py
```

### Docker Deployment

```dockerfile
# Dockerfile example
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "backend/app.py"]
```

### Cloud Deployment

Compatible with:
- Heroku
- AWS EC2
- Google Cloud Platform
- DigitalOcean

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [librosa](https://librosa.org/) for audio processing capabilities
- [TensorFlow](https://tensorflow.org/) for deep learning framework
- [Flask](https://flask.palletsprojects.com/) for web framework
- [PESQ](https://github.com/ludlows/python-pesq) for speech quality evaluation
- [pySTOI](https://github.com/mpariente/pystoi) for intelligibility metrics

## Contact

**[Your Name]** - [your.email@example.com]

Project Link: [https://github.com/[your-username]/audify](https://github.com/[your-username]/audify)

---

If you found this project helpful, please give it a star on GitHub!
# """
# Main Flask application server for Audify.
# Serves both API endpoints and static HTML frontend.
# """
# import os
# import sys
# from flask import Flask, render_template, jsonify



# # Import the API blueprint from api.py
# from api import app as api_app

# # Create main Flask app
# app = Flask(__name__, 
#             static_folder='../', 
#             static_url_path='/',
#             template_folder='../')

# # Check if model exists on startup
# MODEL_PATH = "backend/models/frame_model.keras"
# STATS_PATH = "backend/models/norm_stats.json"

# def check_model_availability():
#     """Check if trained model exists"""
#     return os.path.exists(MODEL_PATH) and os.path.exists(STATS_PATH)

# @app.route('/')
# def index():
#     """Serve the main HTML page"""
#     return render_template('index.html')

# @app.route('/health')
# def health():
#     """Health check endpoint"""
#     model_available = check_model_availability()
#     return jsonify({
#         "status": "healthy",
#         "model_loaded": model_available,
#         "model_path": MODEL_PATH,
#         "stats_path": STATS_PATH
#     })

# # Register API routes from api.py
# @app.errorhandler(404)
# def not_found(error):
#     """Handle 404 errors"""
#     return jsonify({'error': 'Endpoint not found'}), 404

# @app.errorhandler(500)
# def internal_error(error):
#     """Handle 500 errors"""
#     return jsonify({'error': 'Internal server error'}), 500

# # Import API routes
# try:
#     from api import enhance, get_status, download_file
#     app.add_url_rule('/enhance', 'enhance', enhance, methods=['POST'])
#     app.add_url_rule('/status/<processing_id>', 'get_status', get_status, methods=['GET'])
#     app.add_url_rule('/outputs/<filename>', 'download_file', download_file, methods=['GET'])
# except ImportError as e:
#     print(f"❌ Failed to import API routes: {e}")
#     sys.exit(1)

# if __name__ == '__main__':
#     # Check model availability on startup
#     if not check_model_availability():
#         print("❌ ERROR: Trained model not found!")
#         print("Please run 'python backend/train.py' first to train the model.")
#         print(f"Expected files:")
#         print(f"  - {MODEL_PATH}")
#         print(f"  - {STATS_PATH}")
#         sys.exit(1)
    
#     print("🚀 Starting Audify server...")
#     print("📱 Open http://localhost:5000 in your browser")
    
#     # Create necessary directories
#     os.makedirs('temp', exist_ok=True)
#     os.makedirs('/outputs', exist_ok=True)
    
#     app.run(host='0.0.0.0', port=5000, debug=False)




"""
Main Flask application server for Audify.
Serves both API endpoints and static HTML frontend with SocketIO streaming support.
"""
import os
import sys
from flask import Flask, render_template, jsonify

# Import the streaming module
try:
    from streaming_handler import init_streaming
    STREAMING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Streaming module not available: {e}")
    STREAMING_AVAILABLE = False

# Create main Flask app
app = Flask(__name__, 
            static_folder='../', 
            static_url_path='/',
            template_folder='../')

# Initialize SocketIO for streaming if available
socketio = None
if STREAMING_AVAILABLE:
    try:
        socketio = init_streaming(app)
        print("✅ SocketIO streaming initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize streaming: {e}")
        STREAMING_AVAILABLE = False

# Check if model exists on startup
MODEL_PATH = "backend/models/frame_model.keras"
STATS_PATH = "backend/models/norm_stats.json"

def check_model_availability():
    """Check if trained model exists"""
    return os.path.exists(MODEL_PATH) and os.path.exists(STATS_PATH)

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    model_available = check_model_availability()
    return jsonify({
        "status": "healthy",
        "model_loaded": model_available,
        "streaming_available": STREAMING_AVAILABLE,
        "socketio_enabled": socketio is not None,
        "model_path": MODEL_PATH,
        "stats_path": STATS_PATH
    })

@app.route('/streaming/health')
def streaming_health():
    """Streaming-specific health check"""
    if not STREAMING_AVAILABLE:
        return jsonify({
            "status": "unavailable",
            "error": "Streaming module not loaded"
        }), 503
    
    return jsonify({
        "status": "available",
        "socketio_enabled": socketio is not None,
        "model_loaded": check_model_availability()
    })

# Register API routes from api.py
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

# Import API routes
try:
    from api import enhance, get_status, download_file
    app.add_url_rule('/enhance', 'enhance', enhance, methods=['POST'])
    app.add_url_rule('/status/<processing_id>', 'get_status', get_status, methods=['GET'])
    app.add_url_rule('/outputs/<filename>', 'download_file', download_file, methods=['GET'])
    print("✅ API routes registered successfully")
except ImportError as e:
    print(f"❌ Failed to import API routes: {e}")
    sys.exit(1)

def get_app():
    """Return the Flask app instance for external use"""
    return app

def get_socketio():
    """Return the SocketIO instance for external use"""
    return socketio

if __name__ == '__main__':
    # Check model availability on startup
    if not check_model_availability():
        print("❌ ERROR: Trained model not found!")
        print("Please run 'python backend/train.py' first to train the model.")
        print(f"Expected files:")
        print(f"  - {MODEL_PATH}")
        print(f"  - {STATS_PATH}")
        sys.exit(1)
    
    print("🚀 Starting Audify server...")
    print("📱 Open http://localhost:5000 in your browser")
    
    if STREAMING_AVAILABLE:
        print("🎤 Real-time streaming enabled")
    else:
        print("⚠️  Real-time streaming disabled (missing dependencies)")
    
    # Create necessary directories
    os.makedirs('temp', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Run with SocketIO if available, otherwise regular Flask
    if socketio:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    else:
        app.run(host='0.0.0.0', port=5000, debug=False)
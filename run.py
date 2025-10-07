# """
# Main launcher script for Audify AI Speech Enhancement.
# Checks model availability and starts the Flask server.
# """
# import os
# import sys
# import webbrowser
# import time
# import socket
# from contextlib import closing

# # MAKE SURE this import pulls in your Flask `app` object:
# sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))
# from app import app   # <- now `app` is in this module’s globals

# def check_model_availability():
#     """Check if trained model and stats files exist"""
#     model_path = "backend/models/frame_model.keras"
#     stats_path = "backend/models/norm_stats.json"
    
#     model_exists = os.path.exists(model_path)
#     stats_exists = os.path.exists(stats_path)
    
#     return model_exists and stats_exists, model_path, stats_path

# def check_port_available(port):
#     """Check if port is available"""
#     with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
#         return sock.connect_ex(('localhost', port)) != 0

# def print_training_instructions():
#     """Print clear instructions for training the model"""
#     print("\n" + "="*60)
#     print("🤖 AUDIFY AI SPEECH ENHANCEMENT")
#     print("="*60)
#     print("\n❌ ERROR: Trained model not found!")
#     print("\n📋 SETUP REQUIRED:")
#     print("1. Prepare your dataset:")
#     print("   - Add clean audio files to: dataset/clean/")
#     print("   - Add noisy audio files to: dataset/noisy/")
#     print("   - Files should have matching names (e.g., audio1.wav in both folders)")
#     print("   - Supported formats: WAV, MP3, FLAC")
#     print("\n2. Train the model (one-time setup):")
#     print("   python backend/train.py")
#     print("\n3. Then run this script again:")
#     print("   python run.py")
#     print("\n💡 Training typically takes 5-15 minutes depending on dataset size.")
#     print("="*60)

# def print_startup_banner():
#     """Print startup banner with server info"""
#     print("\n" + "="*60)
#     print("🚀 AUDIFY AI SPEECH ENHANCEMENT")
#     print("="*60)
#     print("✅ Model loaded successfully!")
#     print("🌐 Starting Flask server...")
#     print("📱 Server URL: http://localhost:5000")
#     print("💡 Press Ctrl+C to stop the server")
#     print("="*60)

# def main():
#     """Main function to start Audify"""
#     try:
#         # Check if model exists
#         model_available, model_path, stats_path = check_model_availability()
        
#         if not model_available:
#             print_training_instructions()
#             print(f"\n📁 Expected files:")
#             print(f"   - {model_path}")
#             print(f"   - {stats_path}")
#             sys.exit(1)
        
#         # Check if port 5000 is available
#         if not check_port_available(5000):
#             print("❌ ERROR: Port 5000 is already in use!")
#             print("💡 Please stop any other applications using port 5000 or kill the process.")
#             print("   On Windows: netstat -ano | findstr :5000")
#             print("   On Mac/Linux: lsof -i :5000")
#             sys.exit(1)
        
#         # Print startup banner
#         print_startup_banner()
        
#         # Create necessary directories
#         os.makedirs('temp', exist_ok=True)
#         os.makedirs('outputs', exist_ok=True)
        
#         # Import and start Flask app
#         try:
#     # Add backend to Python path
#             sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))
#             from waitress import serve
            
            
#             print("\n🚀 Starting server...")
#             serve(app, host='0.0.0.0', port=5000)
#             # Start Flask server
            
#         except ImportError as e:
#             print(f"❌ ERROR: Failed to import Flask app: {e}")
#             print("💡 Make sure all dependencies are installed:")
#             print("   pip install -r requirements.txt")
#             sys.exit(1)
            
#         except Exception as e:
#             print(f"❌ ERROR: Failed to start server: {e}")
#             sys.exit(1)
    
#     except KeyboardInterrupt:
#         print("\n\n👋 Server stopped by user. Goodbye!")
#         sys.exit(0)
    
#     except Exception as e:
#         print(f"\n❌ UNEXPECTED ERROR: {e}")
#         print("💡 Please check the error message above and try again.")
#         sys.exit(1)

# if __name__ == '__main__':
#     main()



"""
Main launcher script for Audify AI Speech Enhancement with SocketIO streaming support.
Checks model availability and starts the Flask server with production deployment configuration.
"""
import os
import sys
import webbrowser
import time
import socket
from contextlib import closing

# Add backend to Python path
sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))

def check_model_availability():
    """Check if trained model and stats files exist"""
    model_path = "backend/models/frame_model.keras"
    stats_path = "backend/models/norm_stats.json"
    
    model_exists = os.path.exists(model_path)
    stats_exists = os.path.exists(stats_path)
    
    return model_exists and stats_exists, model_path, stats_path

def check_streaming_availability():
    """Check if streaming dependencies are available"""
    try:
        import flask_socketio
        import eventlet
        return True, "eventlet"
    except ImportError:
        try:
            from waitress import serve
            return True, "waitress"
        except ImportError:
            return False, None

def check_port_available(port):
    """Check if port is available"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex(('localhost', port)) != 0

def print_training_instructions():
    """Print clear instructions for training the model"""
    print("\n" + "="*60)
    print("🤖 AUDIFY AI SPEECH ENHANCEMENT")
    print("="*60)
    print("\n❌ ERROR: Trained model not found!")
    print("\n📋 SETUP REQUIRED:")
    print("1. Prepare your dataset:")
    print("   - Add clean audio files to: dataset/clean/")
    print("   - Add noisy audio files to: dataset/noisy/")
    print("   - Files should have matching names (e.g., audio1.wav in both folders)")
    print("   - Supported formats: WAV, MP3, FLAC")
    print("\n2. Train the model (one-time setup):")
    print("   python backend/train.py")
    print("\n3. Then run this script again:")
    print("   python run.py")
    print("\n💡 Training typically takes 5-15 minutes depending on dataset size.")
    print("="*60)

def print_startup_banner(streaming_available, server_type):
    """Print startup banner with server info"""
    print("\n" + "="*60)
    print("🚀 AUDIFY AI SPEECH ENHANCEMENT")
    print("="*60)
    print("✅ Model loaded successfully!")
    print("🌐 Starting Flask server...")
    print("📱 Server URL: http://localhost:5000")
    
    if streaming_available:
        print(f"🎤 Real-time streaming enabled ({server_type})")
        print("🎵 Live enhancement ready for use")
    else:
        print("⚠️  Real-time streaming disabled (missing dependencies)")
        print("   Install flask-socketio and eventlet for streaming support")
    
    print("💡 Press Ctrl+C to stop the server")
    print("="*60)

def run_with_socketio_eventlet():
    """Run server with SocketIO using eventlet"""
    try:
        from app import get_app, get_socketio
        
        app = get_app()
        socketio = get_socketio()
        
        if socketio is None:
            raise ImportError("SocketIO not initialized in app")
        
        print("🔌 Starting SocketIO server with eventlet...")
        
        # Configure eventlet for production
        import eventlet
        eventlet.monkey_patch()
        
        # Start SocketIO server with eventlet
        socketio.run(app, 
                    host='0.0.0.0', 
                    port=5000, 
                    debug=False,
                    use_reloader=False,
                    allow_unsafe_werkzeug=True)
        
    except Exception as e:
        print(f"❌ ERROR: Failed to start SocketIO server with eventlet: {e}")
        print("💡 Falling back to waitress server...")
        return run_with_waitress()

def run_with_waitress():
    """Run server with waitress (no SocketIO streaming)"""
    try:
        from app import get_app
        from waitress import serve
        
        app = get_app()
        
        print("🔧 Starting server with waitress (streaming disabled)...")
        print("⚠️  Note: Real-time streaming features will not be available")
        
        # Configure waitress for production
        serve(app, 
              host='0.0.0.0', 
              port=5000,
              threads=4,
              connection_limit=100,
              cleanup_interval=30,
              channel_timeout=120)
              
    except Exception as e:
        print(f"❌ ERROR: Failed to start waitress server: {e}")
        return run_development_server()

def run_development_server():
    """Fallback to Flask development server"""
    try:
        from app import get_app
        
        app = get_app()
        
        print("🔧 Starting Flask development server...")
        print("⚠️  Warning: This is not suitable for production use")
        print("⚠️  Note: Real-time streaming features will not be available")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except Exception as e:
        print(f"❌ ERROR: Failed to start development server: {e}")
        sys.exit(1)

def verify_streaming_health():
    """Verify streaming components are working"""
    try:
        from app import get_app, get_socketio
        
        app = get_app()
        socketio = get_socketio()
        
        if socketio is None:
            return False, "SocketIO not initialized"
        
        # Test basic SocketIO functionality
        with app.app_context():
            # Check if streaming handler is available
            try:
                from streaming_handler import client_sessions
                return True, f"Streaming ready (clients: {len(client_sessions)})"
            except ImportError as e:
                return False, f"Streaming handler not available: {e}"
                
    except Exception as e:
        return False, f"Health check failed: {e}"

def main():
    """Main function to start Audify with streaming support"""
    try:
        # Check if model exists
        model_available, model_path, stats_path = check_model_availability()
        
        if not model_available:
            print_training_instructions()
            print(f"\n📁 Expected files:")
            print(f"   - {model_path}")
            print(f"   - {stats_path}")
            sys.exit(1)
        
        # Check if port 5000 is available
        if not check_port_available(5000):
            print("❌ ERROR: Port 5000 is already in use!")
            print("💡 Please stop any other applications using port 5000 or kill the process.")
            print("   On Windows: netstat -ano | findstr :5000")
            print("   On Mac/Linux: lsof -i :5000")
            sys.exit(1)
        
        # Check streaming availability
        streaming_available, server_type = check_streaming_availability()
        
        # Print startup banner
        print_startup_banner(streaming_available, server_type)
        
        # Create necessary directories
        os.makedirs('temp', exist_ok=True)
        os.makedirs('outputs', exist_ok=True)
        
        # Verify streaming health if available
        if streaming_available:
            health_ok, health_msg = verify_streaming_health()
            if health_ok:
                print(f"✅ Streaming health check: {health_msg}")
            else:
                print(f"⚠️  Streaming health warning: {health_msg}")
                print("   Batch processing will still be available")
        
        # Start appropriate server based on capabilities
        print("\n🚀 Starting server...")
        
        if streaming_available and server_type == "eventlet":
            # Full SocketIO support with eventlet
            run_with_socketio_eventlet()
            
        elif streaming_available and server_type == "waitress":
            # Production server without SocketIO
            run_with_waitress()
            
        else:
            # Development fallback
            run_development_server()
            
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user. Goodbye!")
        sys.exit(0)
    
    except ImportError as e:
        print(f"❌ ERROR: Missing dependencies: {e}")
        print("💡 Please install required packages:")
        print("   pip install -r requirements.txt")
        
        # Try to start basic server without streaming
        print("\n🔧 Attempting to start basic server without streaming...")
        try:
            run_development_server()
        except:
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        print("💡 Please check the error message above and try again.")
        
        # Print debug information
        print(f"\n🔍 Debug information:")
        print(f"   Python version: {sys.version}")
        print(f"   Working directory: {os.getcwd()}")
        print(f"   Python path: {sys.path[:3]}...")  # First 3 entries
        
        sys.exit(1)

if __name__ == '__main__':
    main()
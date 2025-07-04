# Optimized Flask Face Recognition App for i.MX8M Plus (aarch64)
# Key optimizations: eventlet, lazy loading, reduced video streaming, lightweight TTS

import eventlet
eventlet.monkey_patch()  # Must be first for async Socket.IO

from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import time
import uuid
import tempfile
import threading
import subprocess
import os
import cv2
import numpy as np
import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor

# Lazy imports for heavy models
whisper = None
YoloFace = None
Facenet = None
FaceDatabase = None

# --- CONFIGURATIONS ---
WHISPER_MODEL = "tiny.en"  # Faster English-only model
NAME_PROMPT = "Adarsh, Aarav, Nayan, Riya, Lakshmi, Arjun, Priya, Deepa, Neha, Rohan, Anjali, Vijay, Vinay, Vikram, Sanjay, Suraj"
COMMAND_PROMPT = "new, add, remove, delete, quit"
DEFAULT_AUDIO_DEVICE = "default"
DEFAULT_TTS_DEVICE = "default"
COMMAND_DURATION = 10
NAME_DURATION = 4
CONFIRM_DURATION = 3
TTS_DEBOUNCE = 3.0
PADDING = 10
DEFAULT_VIDEO_DEVICE_INDEX = 0

# Optimized video settings for ARM
VIDEO_WIDTH = 320  # Reduced from 640
VIDEO_HEIGHT = 240  # Reduced from 480
VIDEO_FPS = 5  # Reduced from ~33 FPS
FRAME_SKIP = 0.2  # 200ms between frames instead of 30ms

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Thread pool for heavy operations
executor = ThreadPoolExecutor(max_workers=2)

# Global state
app_state = {
    'operation': None,
    'add_state': None,
    'add_name': None,
    'remove_state': None,
    'remove_name': None,
    'is_recording': False,
    'last_embedding': None,
    'running': False,
    'status_message': 'System Ready',
    'status_type': 'info',
    'camera_status': 'Disconnected',
    'total_faces': 0,
    'faces_detected': 0,
    'audio_device': DEFAULT_AUDIO_DEVICE,
    'tts_device': DEFAULT_TTS_DEVICE,
    'video_device_index': DEFAULT_VIDEO_DEVICE_INDEX,
    'models_loaded': False,
}

# Lazy-loaded global variables
detector = None
recognizer = None
face_db = None
whisper_model = None
camera_capture = None
camera_thread = None

# --- LAZY MODEL LOADING ---
def load_models():
    """Load heavy models in background thread"""
    global detector, recognizer, face_db, whisper_model, whisper, YoloFace, Facenet, FaceDatabase
    
    if app_state['models_loaded']:
        return
    
    try:
        logger.info("Loading models...")
        update_status("Loading AI models...", "warning")
        
        # Import modules only when needed
        if not whisper:
            import whisper
        if not YoloFace:
            from face_detection import YoloFace
        if not Facenet:
            from face_recognition import Facenet
        if not FaceDatabase:
            from face_database import FaceDatabase
        
        # Load models
        detector = YoloFace("yoloface_int8.tflite", "")
        recognizer = Facenet("facenet_512_int_quantized.tflite", "")
        face_db = FaceDatabase()
        
        app_state['models_loaded'] = True
        app_state['total_faces'] = len(get_all_names()) if face_db else 0
        
        logger.info("Models loaded successfully")
        update_status("AI models loaded", "success")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        update_status(f"Model loading failed: {str(e)}", "error")

def load_whisper_model():
    """Load Whisper model only when needed"""
    global whisper_model, whisper
    
    if whisper_model is not None:
        return whisper_model
    
    try:
        if not whisper:
            import whisper
        
        logger.info("Loading Whisper model...")
        update_status("Loading speech recognition...", "warning")
        whisper_model = whisper.load_model(WHISPER_MODEL)
        update_status("Speech recognition ready", "success")
        return whisper_model
    except Exception as e:
        logger.error(f"Error loading Whisper: {e}")
        update_status(f"Speech recognition failed: {str(e)}", "error")
        return None

# --- DEVICE ENUMERATION HELPERS ---
def get_audio_input_devices():
    devices = []
    try:
        arecord_output = subprocess.check_output(['arecord', '-l'], stderr=subprocess.DEVNULL, timeout=5).decode()
        for line in arecord_output.splitlines():
            if 'card' in line and 'device' in line:
                import re
                m = re.search(r'card (\d+): ([^\[]+)\[([^\]]+)\], device (\d+): ([^\[]+)\[([^\]]+)\]', line)
                if m:
                    card_num = m.group(1)
                    card_desc = m.group(3).strip()
                    device_num = m.group(4)
                    device_desc = m.group(6).strip()
                    hw_id = f"plughw:{card_num},{device_num}"
                    disp = f"{card_desc} - {device_desc} ({hw_id})"
                    devices.append({"display": disp, "id": hw_id})
    except Exception as e:
        logger.warning(f"Audio input enumeration failed: {e}")
    
    if not devices:
        devices = [{"display": "Default", "id": DEFAULT_AUDIO_DEVICE}]
    return devices

def get_audio_output_devices():
    devices = []
    try:
        aplay_output = subprocess.check_output(['aplay', '-l'], stderr=subprocess.DEVNULL, timeout=5).decode()
        for line in aplay_output.splitlines():
            if 'card' in line and 'device' in line:
                import re
                m = re.search(r'card (\d+): ([^\[]+)\[([^\]]+)\], device (\d+): ([^\[]+)\[([^\]]+)\]', line)
                if m:
                    card_num = m.group(1)
                    card_desc = m.group(3).strip()
                    device_num = m.group(4)
                    device_desc = m.group(6).strip()
                    hw_id = f"plughw:{card_num},{device_num}"
                    disp = f"{card_desc} - {device_desc} ({hw_id})"
                    devices.append({"display": disp, "id": hw_id})
    except Exception as e:
        logger.warning(f"Audio output enumeration failed: {e}")
    
    if not devices:
        devices = [{"display": "Default", "id": DEFAULT_TTS_DEVICE}]
    return devices

def get_video_devices(max_devices=5):  # Reduced from 10
    available = []
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            available.append({"display": f"Camera {idx}", "id": idx})
            cap.release()
    if not available:
        available = [{"display": "Default (0)", "id": 0}]
    return available

# --- STATUS MANAGEMENT ---
def update_status(message, status_type="info"):
    app_state['status_message'] = message
    app_state['status_type'] = status_type
    app_state['status_time'] = time.time()
    socketio.emit('status_update', {
        'message': message,
        'type': status_type
    })
    logger.info(f"Status: {message} ({status_type})")

# --- LIGHTWEIGHT TTS ENGINE ---
class LightweightTTS:
    """Lightweight TTS using espeak-ng instead of pyttsx3"""
    
    def __init__(self, tts_device=None):
        self.tts_thread = None
        self.last_tts_time = 0
        self.tts_device = tts_device or app_state.get("tts_device", DEFAULT_TTS_DEVICE)
        self.use_espeak = self._check_espeak()
    
    def _check_espeak(self):
        """Check if espeak-ng is available"""
        try:
            subprocess.run(['espeak-ng', '--version'], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
            return True
        except:
            logger.warning("espeak-ng not found, TTS disabled")
            return False
    
    def say(self, text):
        if not self.use_espeak:
            return
            
        if self.tts_thread and self.tts_thread.is_alive():
            return
        if time.time() - self.last_tts_time < TTS_DEBOUNCE:
            return
        
        def run_tts():
            try:
                # Use espeak-ng for faster, lighter TTS
                subprocess.run(['espeak-ng', text], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            except Exception as e:
                logger.warning(f"TTS failed: {e}")
        
        self.tts_thread = threading.Thread(target=run_tts, daemon=True)
        self.tts_thread.start()
        self.last_tts_time = time.time()

tts = LightweightTTS()

# --- AUDIO FUNCTIONS ---
def record_audio(filename, duration=4):
    try:
        app_state['is_recording'] = True
        update_status(f"🎤 Recording audio for {duration}s...", "warning")
        socketio.emit('recording_status', {'recording': True})
        
        audio_dev = app_state.get("audio_device", DEFAULT_AUDIO_DEVICE)
        cmd = ['arecord', '-D', audio_dev, '-f', 'S16_LE', '-r', '16000', '-c', '1', '-d', str(duration), filename]
        result = subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=duration+2)
        time.sleep(0.1)  # Reduced from 0.5
        
        app_state['is_recording'] = False
        if result.returncode == 0:
            update_status("Recording completed", "success")
            socketio.emit('recording_status', {'recording': False})
            return os.path.exists(filename)
        else:
            update_status("Recording failed", "error")
            socketio.emit('recording_status', {'recording': False})
            return False
            
    except Exception as e:
        logger.error(f"Recording error: {e}")
        app_state['is_recording'] = False
        update_status("Recording failed", "error")
        socketio.emit('recording_status', {'recording': False})
        return False

def whisper_transcribe(audio_file, prompt_context=""):
    try:
        model = load_whisper_model()
        if not model:
            return ""
            
        # Optimized Whisper inference
        audio_input = whisper.load_audio(audio_file)
        audio_input = whisper.pad_or_trim(audio_input)
        mel = whisper.log_mel_spectrogram(audio_input).to(model.device)
        
        # Faster decoding options
        options = whisper.DecodingOptions(
            language="en", 
            fp16=False, 
            prompt=prompt_context, 
            temperature=0.3,  # Lower temperature for faster inference
            #no_speech_threshold=0.6,
            #logprob_threshold=-1.0
        )
        
        result = whisper.decode(model, mel, options)
        return result.text.strip().lower()
        
    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        return ""
    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)

def recognize_command():
    tmp_file = f"cmd_{uuid.uuid4()}.wav"
    if not record_audio(tmp_file, COMMAND_DURATION):
        return ""
    cmd_text = whisper_transcribe(tmp_file, f"Commands: {COMMAND_PROMPT}")
    
    if "new" in cmd_text or "add" in cmd_text:
        return "add"
    elif "remove" in cmd_text or "delete" in cmd_text:
        return "delete"
    elif "quit" in cmd_text or "exit" in cmd_text:
        return "quit"
    return ""

def recognize_name():
    tmp_file = f"name_{uuid.uuid4()}.wav"
    if not record_audio(tmp_file, NAME_DURATION):
        return None
    name_text = whisper_transcribe(tmp_file, f"Example names: {NAME_PROMPT}")
    
    if not name_text:
        return None
        
    import string
    name_text = name_text.translate(str.maketrans('', '', string.punctuation))
    words = [w.strip().capitalize() for w in name_text.split()
             if w.strip().isalpha() and w.lower() not in ['names', 'name', 'example']]
    
    if not words or len(words) > 2:
        return None
    return ' '.join(words)

# --- DATABASE HELPERS ---
def add_name_to_db(name, embedding):
    if face_db:
        face_db.add_name(name, embedding)
        app_state['total_faces'] = len(get_all_names())

def delete_name_from_db(name):
    if face_db:
        face_db.del_name(name)
        app_state['total_faces'] = len(get_all_names())

def get_all_names():
    if face_db:
        return face_db.get_names()
    return []

def find_name_from_embedding(embedding):
    if face_db:
        return face_db.find_name(embedding)
    return "Unknown"

# --- OPTIMIZED CAMERA PROCESSING ---
def camera_process():
    global camera_capture
    video_index = app_state.get("video_device_index", 0)
    
    try:
        camera_capture = cv2.VideoCapture(video_index)
        camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
        camera_capture.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
        camera_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer
        
        if not camera_capture.isOpened():
            app_state['camera_status'] = 'Error'
            update_status("Camera access failed", "error")
            return
        
        app_state['camera_status'] = 'Connected'
        frame_count = 0
        
        while app_state['running']:
            ret, frame = camera_capture.read()
            if not ret:
                continue
            
            # Skip frames to reduce CPU load
            frame_count += 1
            if frame_count % 2 != 0:  # Process every 2nd frame
                continue
                
            # Ensure frame is correct size
            frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
            
            current_embedding = None
            detected_count = 0
            
            # Only process faces if models are loaded
            if detector and recognizer:
                try:
                    boxes = detector.detect(frame)
                    detected_count = len(boxes)
                    
                    for box in boxes:
                        box[[0, 2]] *= frame.shape[1]
                        box[[1, 3]] *= frame.shape[0]
                        x1, y1, x2, y2 = box.astype(np.int32)
                        x1, y1 = max(x1 - PADDING, 0), max(y1 - PADDING, 0)
                        x2, y2 = min(x2 + PADDING, frame.shape[1]), min(y2 + PADDING, frame.shape[0])
                        
                        face = frame[y1:y2, x1:x2]
                        if face.size == 0:
                            continue
                            
                        embeddings = recognizer.get_embeddings(face)
                        name = "Unknown"
                        if embeddings is not None:
                            current_embedding = embeddings
                            try:
                                name = find_name_from_embedding(embeddings)
                            except Exception:
                                name = "Error"
                        
                        color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, name, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)  # Smaller font
                        
                except Exception as e:
                    logger.error(f"Face processing error: {e}")
            
            app_state['last_embedding'] = current_embedding
            app_state['faces_detected'] = detected_count
            
            # Optimized frame encoding
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # Lower quality for speed
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_data = base64.b64encode(buffer).decode('utf-8')
            
            socketio.emit('video_frame', {
                'frame': frame_data,
                'faces_detected': detected_count
            })
            
            time.sleep(FRAME_SKIP)  # Reduced FPS
            
    except Exception as e:
        logger.error(f"Camera process error: {e}")
        app_state['camera_status'] = 'Error'
        update_status(f"Camera error: {str(e)}", "error")
    finally:
        if camera_capture:
            camera_capture.release()

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/devices')
def get_devices():
    return jsonify({
        'video_devices': get_video_devices(),
        'audio_input_devices': get_audio_input_devices(),
        'audio_output_devices': get_audio_output_devices()
    })

@app.route('/api/names')
def get_names():
    names = get_all_names()
    return jsonify({
        'names': names,
        'total_faces': len(names)
    })

@app.route('/api/status')
def get_status():
    return jsonify({
        'status_message': app_state['status_message'],
        'status_type': app_state['status_type'],
        'camera_status': app_state['camera_status'],
        'faces_detected': app_state['faces_detected'],
        'is_recording': app_state['is_recording'],
        'running': app_state['running'],
        'models_loaded': app_state['models_loaded']
    })

# --- SOCKET EVENTS ---
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    # Start loading models in background when first client connects
    if not app_state['models_loaded']:
        executor.submit(load_models)

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('start_camera')
def handle_start_camera():
    global camera_thread
    if not app_state['running']:
        if not app_state['models_loaded']:
            update_status("Loading models first...", "warning")
            load_models()
        
        app_state['running'] = True
        app_state['camera_status'] = 'Starting'
        update_status("Camera starting...", "info")
        camera_thread = threading.Thread(target=camera_process, daemon=True)
        camera_thread.start()

@socketio.on('stop_camera')
def handle_stop_camera():
    app_state['running'] = False
    app_state['camera_status'] = 'Disconnected'
    update_status("Camera stopped", "warning")

@socketio.on('voice_command')
def handle_voice_command():
    def process_command():
        if not app_state['models_loaded']:
            update_status("Models not loaded", "error")
            return
            
        command = recognize_command()
        if command == "add":
            app_state["operation"] = "add"
            app_state["add_state"] = "ask_name"
            update_status("Ready to add new face", "info")
            tts.say("Say the name to add")
            socketio.emit('operation_update', {
                'operation': 'add',
                'state': 'ask_name'
            })
        elif command == "delete":
            app_state["operation"] = "delete"
            app_state["remove_state"] = "ask_name"
            update_status("Ready to delete face", "info")
            tts.say("Say the name to delete")
            socketio.emit('operation_update', {
                'operation': 'delete',
                'state': 'ask_name'
            })
        elif command == "quit":
            app_state["running"] = False
            app_state["operation"] = None
            update_status("Camera stopped by voice command", "warning")
            tts.say("Camera stopped")
            socketio.emit('operation_update', {
                'operation': None,
                'state': None
            })
        else:
            update_status("Unknown voice command", "error")
            tts.say("Unknown command")
    
    executor.submit(process_command)

@socketio.on('record_name')
def handle_record_name():
    def process_name():
        name = recognize_name()
        if name:
            app_state["add_name"] = name
            app_state["add_state"] = "confirm"
            update_status(f"Ready to add: {name}", "info")
            tts.say(f"Confirm to add {name}")
            socketio.emit('name_recorded', {
                'name': name,
                'state': 'confirm'
            })
        else:
            update_status("Could not recognize name", "error")
            tts.say("Could not recognize name")
    
    executor.submit(process_name)

@socketio.on('record_delete_name')
def handle_record_delete_name():
    def process_name():
        name = recognize_name()
        if name:
            app_state["remove_name"] = name
            app_state["remove_state"] = "confirm"
            update_status(f"Ready to delete: {name}", "warning")
            tts.say(f"Confirm to delete {name}")
            socketio.emit('delete_name_recorded', {
                'name': name,
                'state': 'confirm'
            })
        else:
            update_status("Could not recognize name", "error")
            tts.say("Could not recognize name")
    
    executor.submit(process_name)

@socketio.on('confirm_add')
def handle_confirm_add(data):
    confirmed = data.get('confirmed', False)
    if confirmed:
        if app_state.get('last_embedding') is not None:
            add_name_to_db(app_state['add_name'], app_state['last_embedding'])
            update_status(f"Added {app_state['add_name']}", "success")
            tts.say(f"Added {app_state['add_name']}")
        else:
            update_status("No face detected", "error")
            tts.say("No face detected")
    else:
        update_status("Add operation cancelled", "warning")
        tts.say("Cancelled")
    
    app_state["operation"] = None
    app_state["add_state"] = None
    app_state["add_name"] = None
    socketio.emit('operation_complete')

@socketio.on('confirm_delete')
def handle_confirm_delete(data):
    confirmed = data.get('confirmed', False)
    names = get_all_names()
    
    if confirmed:
        if app_state['remove_name'] in names:
            delete_name_from_db(app_state['remove_name'])
            update_status(f"Deleted {app_state['remove_name']}", "success")
            tts.say(f"Deleted {app_state['remove_name']}")
        else:
            update_status("Name not found", "error")
            tts.say("Name not found")
    else:
        update_status("Delete operation cancelled", "info")
        tts.say("Cancelled")
    
    app_state["operation"] = None
    app_state["remove_state"] = None
    app_state["remove_name"] = None
    socketio.emit('operation_complete')

@socketio.on('quick_delete')
def handle_quick_delete(data):
    name = data.get('name')
    if name:
        delete_name_from_db(name)
        update_status(f"Deleted {name}", "success")
        tts.say(f"Deleted {name}")
        socketio.emit('operation_complete')

@socketio.on('update_device')
def handle_update_device(data):
    device_type = data.get('type')
    device_id = data.get('id')
    
    if device_type == 'video':
        app_state['video_device_index'] = device_id
    elif device_type == 'audio_input':
        app_state['audio_device'] = device_id
    elif device_type == 'audio_output':
        app_state['tts_device'] = device_id
        global tts
        tts = LightweightTTS(device_id)

@socketio.on('reset_system')
def handle_reset_system():
    global app_state
    app_state.update({
        'operation': None,
        'add_state': None,
        'add_name': None,
        'remove_state': None,
        'remove_name': None,
        'is_recording': False,
        'last_embedding': None,
        'running': False,
        'status_message': 'System Ready',
        'status_type': 'info',
        'camera_status': 'Disconnected',
        'faces_detected': 0,
    })
    update_status("System reset", "info")
    socketio.emit('system_reset')

@socketio.on('load_models')
def handle_load_models():
    """Manual model loading trigger"""
    if not app_state['models_loaded']:
        executor.submit(load_models)
    else:
        update_status("Models already loaded", "info")

if __name__ == '__main__':
    logger.info("Starting Face Recognition Server for i.MX8M Plus")
    logger.info("Optimizations: eventlet, lazy loading, reduced video streaming")
    
    # Load models in background
    executor.submit(load_models)
    
    # Run with eventlet for better performance
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=5000, 
        debug=False,  # Disabled for production
        use_reloader=False,  # Disabled for production
        log_output=False  # Reduce logging overhead
    )

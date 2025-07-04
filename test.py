from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import time
import uuid
import tempfile
import threading
import whisper
import pyttsx3
import subprocess
import os
import cv2
import numpy as np
import base64
import json
from face_detection import YoloFace
from face_recognition import Facenet
from face_database import FaceDatabase

# --- CONFIGURATIONS ---
WHISPER_MODEL = "tiny"
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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

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
}

# Initialize models
detector = YoloFace("yoloface_int8.tflite", "")
recognizer = Facenet("facenet_512_int_quantized.tflite", "")
face_db = FaceDatabase()
whisper_model = whisper.load_model(WHISPER_MODEL)

# Camera capture object
camera_capture = None
camera_thread = None

# --- DEVICE ENUMERATION HELPERS ---
def get_audio_input_devices():
    devices = []
    try:
        arecord_output = subprocess.check_output(['arecord', '-l'], stderr=subprocess.DEVNULL).decode()
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
    except Exception:
        pass
    if not devices:
        devices = [{"display": "Default", "id": DEFAULT_AUDIO_DEVICE}]
    return devices

def get_audio_output_devices():
    devices = []
    try:
        aplay_output = subprocess.check_output(['aplay', '-l'], stderr=subprocess.DEVNULL).decode()
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
    except Exception:
        pass
    if not devices:
        devices = [{"display": "Default", "id": DEFAULT_TTS_DEVICE}]
    return devices

def get_video_devices(max_devices=10):
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

# --- TTS ENGINE ---
class TTSEngine:
    def __init__(self, tts_device=None):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('volume', 1.0)
            self.engine.setProperty('rate', 150)
        except Exception:
            self.engine = None
        self.tts_thread = None
        self.last_tts_time = 0
        self.tts_device = tts_device or app_state.get("tts_device", DEFAULT_TTS_DEVICE)

    def say(self, text):
        if self.tts_thread and self.tts_thread.is_alive():
            return
        if time.time() - self.last_tts_time < TTS_DEBOUNCE:
            return
        
        def run_tts():
            if not self.engine:
                return
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmp_file = tmpfile.name
            try:
                self.engine.save_to_file(text, tmp_file)
                self.engine.runAndWait()
                tts_dev = app_state.get("tts_device", DEFAULT_TTS_DEVICE)
                subprocess.run(f"aplay -D {tts_dev} {tmp_file}", shell=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
            finally:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
        
        self.tts_thread = threading.Thread(target=run_tts, daemon=True)
        self.tts_thread.start()
        self.last_tts_time = time.time()

tts = TTSEngine()

# --- AUDIO FUNCTIONS ---
def record_audio(filename, duration=4):
    try:
        app_state['is_recording'] = True
        update_status(f"🎤 Recording audio for {duration}s...", "warning")
        socketio.emit('recording_status', {'recording': True})
        
        audio_dev = app_state.get("audio_device", DEFAULT_AUDIO_DEVICE)
        cmd = ['arecord', '-D', audio_dev, '-f', 'S16_LE', '-r', '16000', '-c', '1', '-d', str(duration), filename]
        subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        time.sleep(0.5)
        
        app_state['is_recording'] = False
        update_status("Recording completed", "success")
        socketio.emit('recording_status', {'recording': False})
        return os.path.exists(filename)
    except Exception:
        app_state['is_recording'] = False
        update_status("Recording failed", "error")
        socketio.emit('recording_status', {'recording': False})
        return False

def whisper_transcribe(audio_file, prompt_context=""):
    try:
        audio_input = whisper.load_audio(audio_file)
        audio_input = whisper.pad_or_trim(audio_input)
        mel = whisper.log_mel_spectrogram(audio_input).to(whisper_model.device)
        options = whisper.DecodingOptions(language="en", fp16=False, prompt=prompt_context, temperature=0.5)
        result = whisper.decode(whisper_model, mel, options)
        return result.text.strip().lower()
    except Exception:
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
    face_db.add_name(name, embedding)
    app_state['total_faces'] = len(get_all_names())

def delete_name_from_db(name):
    face_db.del_name(name)
    app_state['total_faces'] = len(get_all_names())

def get_all_names():
    return face_db.get_names()

def find_name_from_embedding(embedding):
    return face_db.find_name(embedding)

# --- CAMERA PROCESSING ---
def camera_process():
    global camera_capture
    video_index = app_state.get("video_device_index", 0)
    camera_capture = cv2.VideoCapture(video_index)
    
    if not camera_capture.isOpened():
        app_state['camera_status'] = 'Error'
        update_status("Camera access failed", "error")
        return
    
    app_state['camera_status'] = 'Connected'
    
    while app_state['running']:
        ret, frame = camera_capture.read()
        if not ret:
            continue
            
        frame = cv2.resize(frame, (640, 480))
        boxes = detector.detect(frame)
        current_embedding = None
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
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        app_state['last_embedding'] = current_embedding
        app_state['faces_detected'] = detected_count
        
        # Convert frame to base64 for web transmission
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        
        socketio.emit('video_frame', {
            'frame': frame_data,
            'faces_detected': detected_count
        })
        
        time.sleep(0.03)
    
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
    return jsonify({
        'names': get_all_names(),
        'total_faces': len(get_all_names())
    })

@app.route('/api/status')
def get_status():
    return jsonify({
        'status_message': app_state['status_message'],
        'status_type': app_state['status_type'],
        'camera_status': app_state['camera_status'],
        'faces_detected': app_state['faces_detected'],
        'is_recording': app_state['is_recording'],
        'running': app_state['running']
    })

# --- SOCKET EVENTS ---
@socketio.on('start_camera')
def handle_start_camera():
    global camera_thread
    if not app_state['running']:
        app_state['running'] = True
        app_state['camera_status'] = 'Connected'
        update_status("Camera started", "success")
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
    
    threading.Thread(target=process_command, daemon=True).start()

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
    
    threading.Thread(target=process_name, daemon=True).start()

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
    
    threading.Thread(target=process_name, daemon=True).start()

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
        tts = TTSEngine(device_id)

@socketio.on('reset_system')
def handle_reset_system():
    global app_state
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
    }
    update_status("System reset", "info")
    socketio.emit('system_reset')

if __name__ == '__main__':
    update_status("Face detector loaded", "success")
    update_status("Face recognizer loaded", "success")
    update_status("Database initialized", "success")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

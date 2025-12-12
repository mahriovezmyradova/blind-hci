import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
import tempfile
import os
import time
import threading
import queue
import pygame
from pygame import mixer
import warnings
warnings.filterwarnings('ignore')


try:
    pygame.init()
    mixer.init()
except:
    pass


st.set_page_config(
    page_title="Blinders Off Assistant",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton button {
        width: 100%;
        font-size: 1.1rem;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
    }
    .start-btn {
        background-color: #10B981 !important;
        color: white !important;
    }
    .start-btn:hover {
        background-color: #0D966E !important;
    }
    .stop-btn {
        background-color: #EF4444 !important;
        color: white !important;
    }
    .stop-btn:hover {
        background-color: #DC2626 !important;
    }
    .status-indicator {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
    .status-active {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .status-inactive {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    .simple-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .detection-item {
        padding: 8px;
        margin: 5px 0;
        background-color: white;
        border-radius: 5px;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="main-header">👁️ Blind Navigation Assistant</h1>', unsafe_allow_html=True)


if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False
if 'environment' not in st.session_state:
    st.session_state.environment = "room"
if 'last_audio_time' not in st.session_state:
    st.session_state.last_audio_time = 0
if 'last_audio_text' not in st.session_state:
    st.session_state.last_audio_text = ""
if 'detected_objects' not in st.session_state:
    st.session_state.detected_objects = []
if 'video_capture' not in st.session_state:
    st.session_state.video_capture = None
if 'audio_cooldown' not in st.session_state:
    st.session_state.audio_cooldown = 0
if 'last_guidance_hash' not in st.session_state:
    st.session_state.last_guidance_hash = ""
if 'frame_counter' not in st.session_state:
    st.session_state.frame_counter = 0
if 'object_detector' not in st.session_state:
    st.session_state.object_detector = None


class AudioManager:
    def __init__(self):
        self.queue = queue.Queue()
        self.is_playing = False
        self.thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.thread.start()
        self.min_interval = 5  # Minimum 5 seconds between audio messages
    
    def _audio_worker(self):
        while True:
            text = self.queue.get()
            if text is None:
                break
            if not self.is_playing:
                self.is_playing = True
                self.speak(text)
                self.is_playing = False
    
    def speak(self, text):
        try:
          
            current_time = time.time()
            if hasattr(self, 'last_speak_time') and current_time - self.last_speak_time < 2:
                return
            
            tts = gTTS(text=text, lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
                tts.save(tmp.name)
                temp_file = tmp.name
            
            mixer.music.load(temp_file)
            mixer.music.play()
            
         
            while mixer.music.get_busy():
                time.sleep(0.1)
            
            time.sleep(0.5)
            os.unlink(temp_file)
            self.last_speak_time = current_time
            
        except Exception as e:
            print(f"Audio error: {str(e)}")
    
    def add_to_queue(self, text, force=False):
        current_time = time.time()
        
     
        if text == st.session_state.last_audio_text and current_time - st.session_state.last_audio_time < 10:
            return
        
     
        if current_time < st.session_state.audio_cooldown and not force:
            return
        
    
        self.queue.put(text)
        st.session_state.last_audio_time = current_time
        st.session_state.last_audio_text = text
        st.session_state.audio_cooldown = current_time + self.min_interval

# Initialize audio manager
audio_manager = AudioManager()

class RealObjectDetector:
    def __init__(self):
        try:

            import torch
            from ultralytics import YOLO
            

            st.info("Loading YOLO object detection model...")
            self.model = YOLO('yolov8n.pt')  # Nano model for speed
            
  
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
            
            self.model_loaded = True
            st.success("YOLO model loaded successfully!")
            
        except ImportError:
            st.warning("YOLO not available. Using simulated detection.")
            self.model = None
            self.model_loaded = False
            self.class_names = []
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            self.model = None
            self.model_loaded = False
            self.class_names = []
    
    def detect_objects(self, frame, environment="room"):
        """Perform real object detection on a frame"""
        detected_objects = []
        
        if self.model_loaded and self.model is not None:
            try:
                # Run YOLO inference
                results = self.model(frame, conf=0.25, verbose=False)
                
                # Process results
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            
                            if cls < len(self.class_names):
                                obj_name = self.class_names[cls]
                                
                                # Filter based on environment
                                if self.is_relevant_object(obj_name, environment):
                                    # Calculate object position in frame
                                    frame_height, frame_width = frame.shape[:2]
                                    x_center = (x1 + x2) / 2
                                    y_center = (y1 + y2) / 2
                                    
                                    # Determine direction based on horizontal position
                                    if x_center < frame_width * 0.33:
                                        direction = "left"
                                    elif x_center > frame_width * 0.66:
                                        direction = "right"
                                    else:
                                        direction = "ahead"
                                    
                                    # Calculate distance estimate based on bounding box size
                                    bbox_area = (x2 - x1) * (y2 - y1)
                                    frame_area = frame_width * frame_height
                                    size_ratio = bbox_area / frame_area
                                    
                                    detected_objects.append({
                                        'object': obj_name,
                                        'direction': direction,
                                        'confidence': float(conf),
                                        'size_ratio': float(size_ratio),
                                        'x_center': float(x_center),
                                        'y_center': float(y_center),
                                        'timestamp': time.time()
                                    })
                
                return detected_objects
                
            except Exception as e:
                st.error(f"Detection error: {str(e)}")
                return self.detect_objects_simulated(environment)
        else:
            # Fallback to simulated detection
            return self.detect_objects_simulated(environment)
    
    def is_relevant_object(self, obj_name, environment):
        """Check if object is relevant for the current environment"""
        # Objects relevant for both environments
        common_objects = {'person', 'chair', 'door', 'stairs', 'wall'}
        
        if environment == "room":
            room_objects = {'bed', 'table', 'window', 'tv', 'couch', 'laptop', 'book', 'clock', 'vase'}
            return obj_name in common_objects or obj_name in room_objects
        else:  # shopping mall
            mall_objects = {'bench', 'escalator', 'elevator', 'sign', 'display', 'plant', 'fountain', 'kiosk'}
            return obj_name in common_objects or obj_name in mall_objects
    
    def detect_objects_simulated(self, environment):
        """Fallback simulated detection when YOLO is not available"""
        detected = []
        
        if environment == "room":
            possible_objects = ['person', 'chair', 'table', 'door', 'bed', 'window', 'tv', 'laptop']
        else:  # shopping mall
            possible_objects = ['person', 'chair', 'door', 'bench', 'escalator', 'display', 'plant']
        
        num_objects = min(3, len(possible_objects))
        selected = np.random.choice(possible_objects, num_objects, replace=False)
        
        for obj in selected:
            direction = np.random.choice(['left', 'right', 'ahead'])
            size_ratio = np.random.uniform(0.05, 0.3)
            
            detected.append({
                'object': obj,
                'direction': direction,
                'confidence': np.random.uniform(0.7, 0.95),
                'size_ratio': size_ratio,
                'x_center': np.random.uniform(0.1, 0.9),
                'y_center': np.random.uniform(0.1, 0.9),
                'timestamp': time.time()
            })
        
        return detected
    
    def estimate_distance(self, size_ratio):
        """Estimate distance based on object size in frame"""
        if size_ratio > 0.2:
            return 'very close', '1-2 meters', 'Immediate attention needed'
        elif size_ratio > 0.1:
            return 'close', '2-4 meters', 'Approach with caution'
        elif size_ratio > 0.05:
            return 'medium', '4-6 meters', 'Be aware of it'
        else:
            return 'far', '6+ meters', 'No immediate concern'
    
    def generate_guidance(self, detected_objects, environment):
        """Generate specific guidance based on detected objects"""
        if not detected_objects:
            return "No objects detected. Continue forward slowly."
        
        detected_objects.sort(key=lambda x: x['size_ratio'], reverse=True)
        
        guidance_parts = []
        objects_mentioned = set()
        
        for obj in detected_objects[:2]:  # Only mention top 2 closest objects
            obj_name = obj['object']
            if obj_name in objects_mentioned:
                continue
                
            objects_mentioned.add(obj_name)
            
            distance_category, distance_meters, distance_action = self.estimate_distance(obj['size_ratio'])
            direction = obj['direction']
            
            # Generate specific guidance
            if obj_name == 'person':
                guidance = f"{distance_meters} to your {direction}, person. Move slowly and give space."
            elif obj_name == 'chair':
                guidance = f"{distance_meters} to your {direction}, chair. Step carefully or you can sit here."
            elif obj_name == 'table':
                guidance = f"{distance_meters} to your {direction}, table. Move around it carefully."
            elif obj_name == 'door':
                if environment == "room":
                    guidance = f"{distance_meters} to your {direction}, door. This could be an exit or entrance."
                else:
                    guidance = f"{distance_meters} to your {direction}, door. Store entrance or exit."
            elif obj_name == 'stairs':
                guidance = f"{distance_meters} to your {direction}, STAIRS. Be very careful!"
            elif obj_name == 'bed':
                guidance = f"{distance_meters} to your {direction}, bed. Sleeping area ahead."
            elif obj_name == 'bench':
                guidance = f"{distance_meters} to your {direction}, bench. You can rest here."
            elif obj_name == 'escalator':
                guidance = f"{distance_meters} to your {direction}, escalator. Use handrails carefully."
            else:
                guidance = f"{distance_meters} to your {direction}, {obj_name}. Be aware of it."
            
            guidance_parts.append(guidance)
        
        # Add environment-specific advice
        if environment == "shopping mall" and len(guidance_parts) < 2:
            guidance_parts.append("You're in a shopping mall. Move slowly and watch for crowds.")
        elif environment == "room" and len(guidance_parts) < 2:
            guidance_parts.append("You're in a room. Navigate furniture carefully.")
        
        return " ".join(guidance_parts)


if st.session_state.object_detector is None:
    st.session_state.object_detector = RealObjectDetector()

detector = st.session_state.object_detector


def initialize_camera():
    try:
        video_capture = cv2.VideoCapture(0)
        if video_capture.isOpened():
            # Set camera properties for stability
            video_capture.set(cv2.CAP_PROP_FPS, 15)
            video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Test if camera actually works
            for _ in range(5):
                success, test_frame = video_capture.read()
                if success and test_frame is not None:
                    st.session_state.video_capture = video_capture
                    return True
                time.sleep(0.1)
            
            video_capture.release()
            return False
            
    except Exception as e:
        st.error(f"Camera error: {str(e)}")
        return False

def start_navigation():
    if not st.session_state.video_capture:
        if not initialize_camera():
            st.warning("Camera not available - Running in simulation mode")
    
    st.session_state.camera_active = True
    st.session_state.detection_active = True
    st.session_state.last_audio_time = 0
    st.session_state.audio_cooldown = 0
    
    audio_manager.add_to_queue(f"Navigation started. {st.session_state.environment} environment activated. I will guide you every 5 seconds.", force=True)
    return True

def stop_navigation():
    if st.session_state.video_capture:
        st.session_state.video_capture.release()
        st.session_state.video_capture = None
    
    st.session_state.camera_active = False
    st.session_state.detection_active = False
    audio_manager.add_to_queue("Navigation stopped.", force=True)

# Main layout
col1, col2 = st.columns([3, 2])

with col1:
    # Control buttons section
    st.markdown("### Navigation Controls")
    button_col1, button_col2 = st.columns(2)
    
    with button_col1:
        if st.button("▶️ START NAVIGATION", key="start_btn", use_container_width=True):
            if start_navigation():
                st.success("Navigation started")
                st.rerun()
    
    with button_col2:
        if st.button("⏹️ STOP NAVIGATION", key="stop_btn", use_container_width=True):
            stop_navigation()
            st.info("Navigation stopped")
            st.rerun()
    

    st.markdown("### Select Environment")
    environment = st.radio(
        "Choose environment:",
        ["🏠 Room", "🛍️ Shopping Mall"],
        horizontal=True,
        label_visibility="collapsed"
    )
    

    if "Room" in environment:
        st.session_state.environment = "room"
    else:
        st.session_state.environment = "shopping mall"
    

    st.markdown("### Live View")
    camera_display = st.empty()
    
    if st.session_state.camera_active:
        try:
            if st.session_state.video_capture:
                success, frame = st.session_state.video_capture.read()
                if success and frame is not None:
                    # Create a copy for display
                    display_frame = frame.copy()
                    
                    # Add environment indicator
                    env_text = "ROOM" if st.session_state.environment == "room" else "SHOPPING MALL"
                    env_color = (0, 255, 255) if st.session_state.environment == "room" else (255, 200, 0)
                    
                    # Add semi-transparent overlay
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 60), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)
                    
                    # Add text
                    cv2.putText(display_frame, env_text, 
                               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, env_color, 2)
                    
                    # Run real object detection
                    detected = detector.detect_objects(frame, st.session_state.environment)
                    
                    # Draw bounding boxes for visualization
                    for obj in detected:
                        # Simulate bounding box for visualization
                        frame_height, frame_width = frame.shape[:2]
                        x_center = int(obj['x_center'] * frame_width)
                        y_center = int(obj['y_center'] * frame_height)
                        
                        # Draw a simple circle at object center
                        cv2.circle(display_frame, (x_center, y_center), 10, (0, 0, 255), 2)
                        cv2.putText(display_frame, obj['object'][:10], 
                                   (x_center - 30, y_center - 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # Convert to RGB for Streamlit
                    display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    camera_display.image(display_frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Store detected objects
                    st.session_state.detected_objects = detected[-5:]  # Keep last 5
                    
                    # Generate and provide audio guidance every 5 seconds
                    current_time = time.time()
                    if (st.session_state.detection_active and 
                        current_time - st.session_state.last_audio_time >= 5):
                        
                        if detected:
                            guidance = detector.generate_guidance(detected, st.session_state.environment)
                            guidance_hash = hash(guidance)
                            
                            if guidance_hash != st.session_state.last_guidance_hash:
                                audio_manager.add_to_queue(guidance)
                                st.session_state.last_guidance_hash = guidance_hash
                        
                        st.session_state.last_audio_time = current_time
                    
                    # Update status
                    if st.session_state.detection_active:
                        next_update = max(0, 5 - (time.time() - st.session_state.last_audio_time))
                        status_text = f'<div class="status-indicator status-active">🟢 LIVE - Next guidance in {next_update:.0f}s</div>'
                    else:
                        status_text = '<div class="status-indicator status-paused">🟡 PAUSED</div>'
                    
                    st.markdown(status_text, unsafe_allow_html=True)
                    
                else:
                    camera_display.info("Camera feed unavailable - Simulation mode")
                    show_simulation_frame(camera_display)
            else:
                camera_display.info("Camera not available - Simulation mode")
                show_simulation_frame(camera_display)
                
        except Exception as e:
            camera_display.info("Camera error - Simulation mode")
            show_simulation_frame(camera_display)
    else:
        camera_display.info("### System Ready\n\nPress START to begin navigation assistance")
        st.markdown('<div class="status-indicator status-inactive">🔴 SYSTEM READY</div>', unsafe_allow_html=True)

with col2:

    st.markdown("### Navigation Status")
    

    env_display = "🏠 Room" if st.session_state.environment == "room" else "🛍️ Shopping Mall"
    env_description = "Indoor home environment" if st.session_state.environment == "room" else "Large commercial space with stores"
    
    st.markdown(f"""
    <div class="simple-box">
        <h4>{env_display}</h4>
        <p>{env_description}</p>
    </div>
    """, unsafe_allow_html=True)
    

    if st.session_state.camera_active:
        last_update = time.time() - st.session_state.last_audio_time
        next_update = max(0, 5 - last_update)
        
        status_box = f"""
        <div class="simple-box">
            <h4>✅ System Active</h4>
            <p>Real-time object detection active</p>
            <p><small>Next audio guidance in {next_update:.0f} seconds</small></p>
        </div>
        """
    else:
        status_box = """
        <div class="simple-box">
            <h4>⏸System Ready</h4>
            <p>Press START to begin navigation</p>
        </div>
        """
    
    st.markdown(status_box, unsafe_allow_html=True)
    

    if st.session_state.detected_objects:
        st.markdown("### Recent Detections")
        
        for obj in st.session_state.detected_objects[-3:]:  # Show last 3
            icon_map = {
                'person': '👤', 'chair': '🪑', 'table': '🪵',
                'door': '🚪', 'stairs': '🪜', 'wall': '🧱',
                'bed': '🛏️', 'window': '🪟', 'tv': '📺',
                'bench': '🪑', 'escalator': '🔼', 'display': '🖥️',
                'plant': '🌿'
            }
            
            icon = icon_map.get(obj['object'], '●')
            distance = detector.estimate_distance(obj.get('size_ratio', 0.15))[1]
            
            st.markdown(f"""
            <div class="detection-item">
                <strong>{icon} {obj['object'].title()}</strong><br>
                <small>{distance} • {obj['direction']} • {obj.get('confidence', 0.8):.0%} confidence</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No objects detected yet. The system will scan every 5 seconds.")

    st.markdown("### Quick Controls")
    
    control_col1, control_col2 = st.columns(2)
    
    with control_col1:
        if st.button("🗣️ Repeat", use_container_width=True, key="repeat_btn"):
            if st.session_state.last_audio_text:
                audio_manager.add_to_queue("Repeating: " + st.session_state.last_audio_text, force=True)
    
    with control_col2:
        if st.button("📍 Scan Now", use_container_width=True, key="scan_btn"):
            if st.session_state.camera_active:
                audio_manager.add_to_queue("Immediate scan initiated.", force=True)
                st.session_state.last_audio_time = 0

def show_simulation_frame(placeholder):
    """Show a simulation frame when camera is not available"""
    height, width = 480, 640
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        color_value = int(100 + (i / height) * 100)
        if st.session_state.environment == "room":
            frame[i, :] = [color_value, color_value, 150]  # Blueish for room
        else:
            frame[i, :] = [150, color_value, color_value]  # Greenish for mall
    
    env_text = "ROOM SIMULATION" if st.session_state.environment == "room" else "MALL SIMULATION"
    cv2.putText(frame, env_text, 
               (width//2 - 180, height//2 - 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.putText(frame, "Real camera not available", 
               (width//2 - 150, height//2), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    placeholder.image(frame_rgb, channels="RGB", use_column_width=True)


st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Blinders Off Assistant | by Mahri Ovezmyradova"
    "</div>",
    unsafe_allow_html=True
)

if st.session_state.camera_active and st.session_state.detection_active:
    time.sleep(0.5)
    st.rerun()


def cleanup():
    if st.session_state.video_capture:
        st.session_state.video_capture.release()


import atexit
atexit.register(cleanup)
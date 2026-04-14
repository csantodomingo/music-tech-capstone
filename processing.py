import statistics
import sys, os
import numpy as np
from scipy.spatial import ConvexHull
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import time
import threading
from pythonosc import udp_client, dispatcher, osc_server

note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
major_scale = [0, 2, 4, 5, 7, 9, 11]
minor_scale = [0, 2, 3, 5, 7, 8, 10]

class LandmarkPoint:
    """Mimics MediaPipe landmark objects so extract_features works unchanged."""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

LANDMARK_ORDER = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_finger_mcp", "pinky_finger_pip", "pinky_finger_dip", "pinky_finger_tip",
]

def osc_dict_to_landmarks(osc_data: dict):
    """Convert parsed OSC gesture dict to ordered list of LandmarkPoint objects."""
    landmarks = []
    for name in LANDMARK_ORDER:
        if name not in osc_data:
            print(f"Warning: missing landmark '{name}'")
            return None
        pt = osc_data[name]
        landmarks.append(LandmarkPoint(pt["x"], pt["y"], pt["z"]))
    return landmarks

def parse_osc_args(args):
    tokens = list(args)
    result = {"Right": {}, "Left": {}}
    current_hand = None
    i = 0

    while i < len(tokens):
        if isinstance(tokens[i], str) and i + 2 < len(tokens) and tokens[i+1] == ':' and tokens[i+2] == '{':
            block_name = tokens[i]
            i += 3

            if block_name in ("Right", "Left"):
                current_hand = block_name
                continue

            # only skip Gestures if we don't have a current hand context
            if block_name == "Gestures" and current_hand is not None:
                gesture_data = {}
                while i < len(tokens) and tokens[i] != '}':
                    if isinstance(tokens[i], str) and i + 2 < len(tokens) and tokens[i+1] == ':':
                        gesture_data[tokens[i]] = float(tokens[i+2])
                        i += 3
                    else:
                        i += 1
                i += 1  # skip '}'
                result[current_hand]["Gestures"] = gesture_data
                continue

            coords = {}
            while i < len(tokens) and tokens[i] != '}':
                if tokens[i] in ('x', 'y', 'z') and i + 2 < len(tokens) and tokens[i+1] == ':':
                    coords[tokens[i]] = float(tokens[i+2])
                    i += 3
                else:
                    i += 1
            i += 1

            if len(coords) == 3 and current_hand:
                result[current_hand][block_name] = coords
        else:
            i += 1

    return result if (result["Right"] or result["Left"]) else None

# normalize hand landmarks to the bounding box
def normalize_landmarks(landmarks, bbox):
    x_min, y_min, z_min = bbox[0]
    x_max, y_max, z_max = bbox[1]
    norm_landmarks = []
    for landmark in landmarks:
        norm_l = [(landmark.x - x_min) / (x_max - x_min),
                  (landmark.y - y_min) / (y_max - y_min),
                  (landmark.z - z_min) / (z_max - z_min)]
        norm_landmarks.append(norm_l)
    return norm_landmarks


# center hand landmarks around the wrist
def center_landmarks(landmarks):
    wrist = landmarks[0]
    return [[x - wrist[0], y - wrist[1], z - wrist[2]] for x, y, z in landmarks]


# calculate distances between hand landmarks
def calculate_distances(landmarks):
    distances = []
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            distances.append(np.linalg.norm(np.array(landmarks[i]) - np.array(landmarks[j])))
    return distances


# calculate angles between hand landmarks
def calculate_angles(landmarks):
    angles = []
    for i in range(1, 5):
        base = np.array(landmarks[0])
        vec1 = np.array(landmarks[i * 4 - 3]) - base
        vec2 = np.array(landmarks[i * 4]) - np.array(landmarks[i * 4 - 3])
        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angles.append(np.arccos(cos_theta) / np.pi)
    return angles


# calculate convex hull area of hand landmarks
def calculate_convex_hull_area(landmarks):
    points = np.array(landmarks)[:, :2]
    hull = ConvexHull(points)
    return hull.area


# extract features from hand landmarks
def extract_features(landmarks, bbox):
    normalized = normalize_landmarks(landmarks, bbox)
    centered = center_landmarks(normalized)
    distances = calculate_distances(centered)
    angles = calculate_angles(centered)
    convex_area = calculate_convex_hull_area(centered)
    return np.array(centered).flatten().tolist()[3:] + distances + angles + [convex_area]

def get_left_gesture(osc_data):
    left = osc_data.get("Left", {})
    gestures = left.get("Gestures", {})
    if not gestures:
        return None
    # return the gesture name with highest confidence
    return max(gestures, key=gestures.get)

# get working directory
def get_working_dir():
    if hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


# measures network model
class MeasuresNetworkKodalyC1C2_slim(nn.Module):
    def __init__(self):
        super(MeasuresNetworkKodalyC1C2_slim, self).__init__()
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(275, 8192)
        self.fc2 = nn.Linear(8192, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 7)
        for fc in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            init.xavier_uniform_(fc.weight)
            init.constant_(fc.bias, 0.0)

    def forward(self, x):
        x = self.dropout(torch.nn.functional.relu(self.fc1(x)))
        x = self.dropout(torch.nn.functional.relu(self.fc2(x)))
        x = self.dropout(torch.nn.functional.relu(self.fc3(x)))
        x = self.dropout(torch.nn.functional.relu(self.fc4(x)))
        return self.fc5(x)


# predict single input
def predict_single_input(model, input_vector):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
        predictions = model(input_tensor)
        predicted_class = torch.argmax(predictions, dim=1).item()
        probabilities = torch.nn.functional.softmax(predictions, dim=1).squeeze()
        return predicted_class, probabilities.numpy()

class OSCInputHandler:
    def __init__(self, app):
        self.app = app
        self.last_midi_note = None
        self.last_velocity = 0
        self.note_vector = []
        self.note_vector_size = 11
        self.octave_shift = 0 
        self.last_gesture_time = 0  
        self.no_hand_timeout = 0.2  # seconds needed for no hand timeout
        
        # start a watchdog thread that checks for hand disappearance
        watchdog = threading.Thread(target=self.watchdog_loop, daemon=True)
        watchdog.start()

        checkpoint_path = os.path.join(get_working_dir(), "checkpoints", "checkpoint_latest.pt")
        self.model = MeasuresNetworkKodalyC1C2_slim()
        self.model.eval()
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Running!")

    def watchdog_loop(self):
        """sends note_off if no gesture received within timeout period"""
        while True:
            time.sleep(0.05)  # check every 50ms
            if self.last_midi_note is not None:
                elapsed = time.time() - self.last_gesture_time
                if elapsed > self.no_hand_timeout:
                    self.app.osc_client.send_message("/note_off", [])
                    self.last_midi_note = None
                    self.note_vector = []  # clear median buffer too
                    print("[DEBUG] hand lost — note off")

    def compute_median_midi_note(self, midi_note):
        self.note_vector.append(midi_note)
        if len(self.note_vector) > self.note_vector_size:
            self.note_vector = self.note_vector[1:]
        median = statistics.median(self.note_vector)
        if self.note_vector.count(median) >= int(self.note_vector_size / 2):
            return median
        return None

    def handle_gesture(self, address, *args):
        print(f"[DEBUG] message received")
        self.last_gesture_time = time.time()
        """OSC handler for /gesture messages coming from Max."""
        osc_data = parse_osc_args(args)
        if osc_data is None:
            return

        right_landmarks = osc_dict_to_landmarks(osc_data["Right"])  # goes to model
        if right_landmarks is None:
            return

        w, h = 960, 540
        landmarks_scaled = [LandmarkPoint((1 - l.x) * w, l.y * h, l.z * h) for l in right_landmarks]

        min_x = min(l.x for l in landmarks_scaled)
        max_x = max(l.x for l in landmarks_scaled)
        min_y = min(l.y for l in landmarks_scaled)
        max_y = max(l.y for l in landmarks_scaled)
        min_z = min(l.z for l in landmarks_scaled)
        max_z = max(l.z for l in landmarks_scaled)
        bbox = (
            (int(min_x), int(min_y), int(min_z)),
            (int(max_x), int(max_y), int(max_z))
        )

        try:
            features = extract_features(landmarks_scaled, bbox)
            true_val, output = predict_single_input(self.model, features)
            print(f"[DEBUG] probabilities: {[f'{p:.3f}' for p in output]}")

            # compute velocity from hand size (distance proxy)
            hand_width = max_x - min_x
            hand_height = max_y - min_y
            hand_size = (hand_width + hand_height) / 2

            # map to velocity 0-127 — tune min_size/max_size to your range
            min_size = 50   # hand far away
            max_size = 200  # hand close to camera
            velocity = int(np.clip((hand_size - min_size) / (max_size - min_size) * 127, 0, 127))
            print(f"[DEBUG] hand_size: {hand_size:.1f}, velocity: {velocity}")

            # left hand octave control
            left_gesture = get_left_gesture(osc_data)
            if left_gesture == "Thumb_Up":
                self.octave_shift = 1
            elif left_gesture == "Thumb_Down":
                self.octave_shift = -1
            else:
                self.octave_shift = 0
            print(f"[DEBUG] left gesture: {left_gesture}, octave shift: {self.octave_shift}")

            # compute midi note
            raw_class = np.argmax(output)
            raw_midi = self.app.current_scale[raw_class] + self.app.base_note + (self.octave_shift * 12)
            midi_note = self.compute_median_midi_note(raw_midi)
            print(f"[DEBUG] raw class: {raw_class}, raw_midi: {raw_midi}, smoothed: {midi_note}")

            if midi_note is not None:
                if midi_note != self.last_midi_note:
                    self.app.osc_client.send_message("/note", [int(midi_note), int(velocity), 0])
                    self.last_midi_note = midi_note

        except Exception as e:
            print("Processing error:", e)

class CameraApp:
    def __init__(self):
        # State
        self.base_note = 60
        self.current_scale = major_scale
        self.min_velocity_to_trigger_note_on = 40
        self.gesture_handler = OSCInputHandler(self)

        # OSC out (to Max)
        self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 7400)

        # OSC in (from Max)
        disp = dispatcher.Dispatcher()
        disp.map("/base_note", self.handle_base_note)
        disp.map("/scale", self.handle_scale)
        disp.map("/min_velocity", self.handle_min_velocity)
        disp.map("/gesture", self.gesture_handler.handle_gesture) 
        self.osc_server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 7401), disp)
        osc_thread = threading.Thread(target=self.osc_server.serve_forever, daemon=True)
        osc_thread.start()

    def handle_base_note(self, address, *args):
        if args:
            self.base_note = int(args[0])
            print(f"Base note: {self.base_note}")

    def handle_scale(self, address, *args):
        if args:
            self.current_scale = major_scale if int(args[0]) == 0 else minor_scale
            print(f"Scale: {'major' if int(args[0]) == 0 else 'minor'}")

    def handle_min_velocity(self, address, *args):
        if args:
            self.min_velocity_to_trigger_note_on = int(args[0])
            print(f"Min velocity: {self.min_velocity_to_trigger_note_on}")

    def shutdown(self, *args):
        print("\nShutting down...")
        self.osc_server.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    app = CameraApp()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        app.shutdown()
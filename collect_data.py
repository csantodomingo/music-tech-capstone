import sys, os
import numpy as np
import csv
import threading
import time
from scipy.spatial import ConvexHull
from pythonosc import dispatcher, osc_server

# same as PyChiro
LANDMARK_ORDER = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_finger_mcp", "pinky_finger_pip", "pinky_finger_dip", "pinky_finger_tip",
]

GESTURES = ["Do", "Re", "Mi", "Fa", "Sol", "La", "Ti"]  # classes 0-6
CURRENT_LABEL = None
COLLECTING = False
DATA = []

class LandmarkPoint:
    def __init__(self, x, y, z):
        self.x = x; self.y = y; self.z = z

def parse_osc_args(args):
    tokens = list(args)
    result = {}
    i = 0
    while i < len(tokens):
        if isinstance(tokens[i], str) and i + 2 < len(tokens) and tokens[i+1] == ':' and tokens[i+2] == '{':
            block_name = tokens[i]
            i += 3
            if block_name in ("Right", "Left", "Gestures"):
                while i < len(tokens) and tokens[i] != '}':
                    i += 1
                i += 1
                continue
            coords = {}
            while i < len(tokens) and tokens[i] != '}':
                if tokens[i] in ('x', 'y', 'z') and i + 2 < len(tokens) and tokens[i+1] == ':':
                    coords[tokens[i]] = float(tokens[i+2])
                    i += 3
                else:
                    i += 1
            i += 1
            if len(coords) == 3:
                result[block_name] = coords
        else:
            i += 1
    return result if result else None

def osc_dict_to_landmarks(osc_data):
    landmarks = []
    for name in LANDMARK_ORDER:
        if name not in osc_data:
            return None
        pt = osc_data[name]
        landmarks.append(LandmarkPoint(pt["x"], pt["y"], pt["z"]))
    return landmarks

def normalize_landmarks(landmarks, bbox):
    x_min, y_min, z_min = bbox[0]
    x_max, y_max, z_max = bbox[1]
    return [[(l.x-x_min)/(x_max-x_min), (l.y-y_min)/(y_max-y_min), (l.z-z_min)/(z_max-z_min)] for l in landmarks]

def center_landmarks(landmarks):
    wrist = landmarks[0]
    return [[x-wrist[0], y-wrist[1], z-wrist[2]] for x,y,z in landmarks]

def calculate_distances(landmarks):
    distances = []
    for i in range(len(landmarks)):
        for j in range(i+1, len(landmarks)):
            distances.append(np.linalg.norm(np.array(landmarks[i]) - np.array(landmarks[j])))
    return distances

def calculate_angles(landmarks):
    angles = []
    for i in range(1, 5):
        base = np.array(landmarks[0])
        vec1 = np.array(landmarks[i*4-3]) - base
        vec2 = np.array(landmarks[i*4]) - np.array(landmarks[i*4-3])
        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angles.append(np.arccos(np.clip(cos_theta, -1, 1)) / np.pi)
    return angles

def calculate_convex_hull_area(landmarks):
    points = np.array(landmarks)[:, :2]
    hull = ConvexHull(points)
    return hull.area

def extract_features(landmarks, bbox):
    normalized = normalize_landmarks(landmarks, bbox)
    centered = center_landmarks(normalized)
    distances = calculate_distances(centered)
    angles = calculate_angles(centered)
    convex_area = calculate_convex_hull_area(centered)
    return np.array(centered).flatten().tolist()[3:] + distances + angles + [convex_area]

def handle_gesture(address, *args):
    global COLLECTING, CURRENT_LABEL, DATA
    if not COLLECTING or CURRENT_LABEL is None:
        return

    osc_data = parse_osc_args(args)
    if osc_data is None:
        return
    landmarks = osc_dict_to_landmarks(osc_data)
    if landmarks is None:
        return

    w, h = 960, 540
    landmarks_scaled = [LandmarkPoint(l.x*w, l.y*h, l.z*h) for l in landmarks]
    min_x = min(l.x for l in landmarks_scaled)
    max_x = max(l.x for l in landmarks_scaled)
    min_y = min(l.y for l in landmarks_scaled)
    max_y = max(l.y for l in landmarks_scaled)
    min_z = min(l.z for l in landmarks_scaled)
    max_z = max(l.z for l in landmarks_scaled)
    bbox = ((int(min_x), int(min_y), int(min_z)), (int(max_x), int(max_y), int(max_z)))

    try:
        features = extract_features(landmarks_scaled, bbox)
        DATA.append([CURRENT_LABEL] + features)
        print(f"  Recorded sample for {GESTURES[CURRENT_LABEL]} (total: {sum(1 for d in DATA if d[0] == CURRENT_LABEL)})")
    except Exception as e:
        print(f"Error: {e}")

def save_data():
    filename = "training_data.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(DATA)
    print(f"\nSaved {len(DATA)} samples to {filename}")

def print_counts():
    print("\nCurrent sample counts:")
    for i, g in enumerate(GESTURES):
        count = sum(1 for d in DATA if d[0] == i)
        bar = "█" * (count // 5)  # one block per 5 samples
        print(f"  {i} {g:>4}: {count:>4} {bar}")
    print()

def input_loop():

    global COLLECTING, CURRENT_LABEL, DATA, FILENAME

    # ask for name at startup
    name = input("Enter your name: ").strip().lower().replace(" ", "_")
    FILENAME = f"training_data_{name}.csv"
    print(f"Data will be saved to: {FILENAME}")

    print("\n=== Gesture Data Collector ===")
    print("Press 0-6 to start recording a gesture:")
    for i, g in enumerate(GESTURES):
        print(f"  {i} = {g}")
    print("Press SPACE (then Enter) to stop recording")
    print("Press S to save and quit\n")

    while True:
        print_counts()
        key = input("> ").strip()

        if key in [str(i) for i in range(7)]:
            CURRENT_LABEL = int(key)
            COLLECTING = True
            print(f"Recording {GESTURES[CURRENT_LABEL]}... (press Enter to stop)")

        elif key == "" or key == " ":
            COLLECTING = False
            CURRENT_LABEL = None
            print("Stopped.")

        elif key.lower() == "s":
            COLLECTING = False
            save_data()
            sys.exit(0)

if __name__ == "__main__":
    disp = dispatcher.Dispatcher()
    disp.map("/gesture", handle_gesture)
    server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 7401), disp)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print("Listening on port 7401...")
    input_loop()
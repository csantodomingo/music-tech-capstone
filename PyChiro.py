import sys, os
import cv2
import math
import statistics
import mediapipe as mp
import numpy as np
from scipy.spatial import ConvexHull
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import signal
import threading
from pythonosc import udp_client, dispatcher, osc_server

mp_drawing = mp.solutions.drawing_utils
notes_onehot_header = ["Do (C)", "Re (D)", "Mi (E)", "Fa (F)", "Sol (G)", "La (A)", "Si (B)"]
major_scale = [0, 2, 4, 5, 7, 9, 11]
minor_scale = [0, 2, 3, 5, 7, 8, 10]


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


def center_landmarks(landmarks):
    wrist = landmarks[0]
    return [[x - wrist[0], y - wrist[1], z - wrist[2]] for x, y, z in landmarks]


def calculate_distances(landmarks):
    distances = []
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            distances.append(np.linalg.norm(np.array(landmarks[i]) - np.array(landmarks[j])))
    return distances


def calculate_angles(landmarks):
    angles = []
    for i in range(1, 5):
        base = np.array(landmarks[0])
        vec1 = np.array(landmarks[i * 4 - 3]) - base
        vec2 = np.array(landmarks[i * 4]) - np.array(landmarks[i * 4 - 3])
        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angles.append(np.arccos(cos_theta) / np.pi)
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


def get_working_dir():
    if hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


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


def predict_single_input(model, input_vector):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
        predictions = model(input_tensor)
        predicted_class = torch.argmax(predictions, dim=1).item()
        probabilities = torch.nn.functional.softmax(predictions, dim=1).squeeze()
        return predicted_class, probabilities.numpy()


def add_text_to_frame(frame, text="---"):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    h, w, d = frame.shape
    color = (20, 20, 20)
    thickness = 1
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    org = (0, h - text_h)
    cv2.rectangle(frame, (org[0], org[1] - 45), (org[0] + text_w, org[1] + text_h), (255, 255, 255), -1)
    cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def draw_filled_ellipse(frame, center, radius, start_angle, end_angle, color, transparency=0.5):
    overlay = frame.copy()
    cv2.ellipse(overlay, center, radius, 0, start_angle, end_angle, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0, frame)


def draw_mirrored_text(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, color=(255, 255, 255), thickness=1):
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_img = np.zeros((text_size[1], text_size[0], 3), dtype=np.uint8)
    cv2.putText(text_img, text, (0, text_size[1]), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    mirrored_text = cv2.flip(text_img, 1)
    text_x = position[0] - mirrored_text.shape[1]
    text_y = position[1]
    frame[text_y:text_y + mirrored_text.shape[0], text_x:text_x + mirrored_text.shape[1]] = mirrored_text


def draw_filled_ellipse_with_text(frame, center, radius, start_angle, end_angle, color, transparency=0.5, text=""):
    overlay = frame.copy()
    cv2.ellipse(overlay, center, radius, 0, start_angle, end_angle, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0, frame)
    angle_mid = (start_angle + end_angle) / 2
    text_x = int(center[0] + radius[0] * np.cos(np.radians(angle_mid)))
    text_y = int(center[1] + radius[1] * np.sin(np.radians(angle_mid)))
    draw_mirrored_text(frame, text, (text_x, text_y))


class CameraThread(threading.Thread):
    def __init__(self, app):
        super().__init__(daemon=True)
        self.app = app
        self.cap = None
        self.running = False

        checkpoint_path = os.path.join(get_working_dir(), "checkpoints", "checkpoint_epoch_10.pt")
        print(f"Loading model: {checkpoint_path} ...")
        self.model = MeasuresNetworkKodalyC1C2_slim()
        self.model.eval()
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Checkpoint loaded!")

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
            enable_segmentation=True, smooth_segmentation=True, model_complexity=1)

    def compute_velocity(self, results):
        mp_holistic = mp.solutions.holistic
        velocity = 0
        if results.pose_landmarks and results.right_hand_landmarks:
            right_elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
            right_wrist = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
            current_distance_x = abs(right_wrist.x - right_elbow.x)

            self.app.max_elbow_wrist_x_distance = max(self.app.max_elbow_wrist_x_distance, current_distance_x)
            self.app.min_elbow_wrist_x_distance = min(self.app.min_elbow_wrist_x_distance, current_distance_x)

            if self.app.max_elbow_wrist_x_distance > self.app.min_elbow_wrist_x_distance:
                norm = (current_distance_x - self.app.min_elbow_wrist_x_distance) / (
                    self.app.max_elbow_wrist_x_distance - self.app.min_elbow_wrist_x_distance)
                norm = max(0.0, min(1.0, norm))
                velocity = max(0, min(127, int(((1 - norm) ** 2) * 127)))
        return velocity

    def compute_octave(self, results):
        mp_holistic = mp.solutions.holistic
        octave = 0
        if results.pose_landmarks and results.right_hand_landmarks:
            right_elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
            right_wrist = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
            elbow_wrist_vector = [right_wrist.x - right_elbow.x, right_wrist.y - right_elbow.y]
            norm_ew = math.sqrt(elbow_wrist_vector[0] ** 2 + elbow_wrist_vector[1] ** 2)
            dot = elbow_wrist_vector[0]  # dot with [1,0]
            angle = math.acos(dot / norm_ew)
            cross = -elbow_wrist_vector[1]  # cross with [1,0]
            angle_deg = math.degrees(angle)
            if cross < 0 and angle_deg > 25:
                octave = -1
            elif angle_deg < 45:
                octave = 0
            else:
                octave = 1
        return octave

    def compute_median_midi_note(self, midi_note):
        if midi_note is not None:
            self.app.note_vector.append(midi_note)
            if len(self.app.note_vector) > self.app.note_vector_size:
                self.app.note_vector = self.app.note_vector[1:]
            median = statistics.median(self.app.note_vector)
            if self.app.note_vector.count(median) >= int(self.app.note_vector_size / 2):
                return median  
            else:
                return None  
        return None

    def run(self):
        self.running = True
        last_midi_note = None
        last_velocity = 0
        self.cap = cv2.VideoCapture(0)

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            midi_note = None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (960, 540))
            results = self.holistic.process(frame)
            frame = cv2.resize(frame, (640, 360))
            text = "---"

            if results.right_hand_landmarks:
                right_hand_landmarks = results.right_hand_landmarks.landmark
                min_x = min(l.x for l in right_hand_landmarks)
                max_x = max(l.x for l in right_hand_landmarks)
                min_y = min(l.y for l in right_hand_landmarks)
                max_y = max(l.y for l in right_hand_landmarks)
                min_z = min(l.z for l in right_hand_landmarks)
                max_z = max(l.z for l in right_hand_landmarks)
                h, w, _ = frame.shape
                bbox = (int(min_x*w), int(min_y*h), int(min_z*h)), (int(max_x*w), int(max_y*h), int(max_z*h))

                try:
                    features = extract_features(right_hand_landmarks, bbox)
                    true_val, output = predict_single_input(self.model, features)
                    self.app.osc_client.send_message("/probabilities", output.tolist())

                    octave = self.compute_octave(results)
                    midi_note = self.app.current_scale[np.argmax(output)] + self.app.base_note + (octave * 12)
                    midi_note = self.compute_median_midi_note(midi_note)
                    velocity = self.compute_velocity(results)

                    if midi_note is not None:
                        text = notes_onehot_header[np.argmax(output)] + f"{octave}; v:{velocity}"
                        self.app.osc_client.send_message("/note", [int(midi_note), int(velocity), int(octave)])

                   # Draw hand skeleton
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 250, 0), thickness=2, circle_radius=4))

                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 250, 0), thickness=2, circle_radius=4))

                    # Draw elbow/wrist arc visualization
                    right_elbow = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_ELBOW]
                    right_wrist = results.right_hand_landmarks.landmark[self.mp_holistic.HandLandmark.WRIST]
                    h, w, _ = frame.shape
                    elbow_x, elbow_y = int(right_elbow.x * w), int(right_elbow.y * h)
                    wrist_x, wrist_y = int(right_wrist.x * w), int(right_wrist.y * h)

                    dx = wrist_x - elbow_x
                    dy = elbow_y - wrist_y
                    angle_degrees = np.degrees(np.arctan2(dy, dx))
                    if angle_degrees < 0:
                        angle_degrees += 360
                    if angle_degrees > 180:
                        start_angle, end_angle = 0, angle_degrees - 360
                    else:
                        start_angle, end_angle = 0, angle_degrees
                    end_angle = -end_angle

                    oct_0_color = (0, 255, 0)
                    oct_1_color = (255, 255, 0)
                    oct__1_color = (0, 255, 255)
                    arc_color = oct_0_color if -22.5 <= end_angle <= 22.5 else (oct_1_color if end_angle > 22.5 else oct__1_color)

                    cv2.circle(frame, (elbow_x, elbow_y), 4, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (elbow_x, elbow_y), 3, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.line(frame, (elbow_x, elbow_y), (wrist_x, wrist_y), arc_color, 1, cv2.LINE_AA)
                    cv2.line(frame, (elbow_x, elbow_y), (wrist_x, elbow_y), arc_color, 1, cv2.LINE_AA)

                    r = (abs(wrist_x - elbow_x), abs(wrist_x - elbow_x))
                    if -22.5 <= end_angle <= 22.5:
                        draw_filled_ellipse(frame, (elbow_x, elbow_y), r, start_angle, end_angle, oct_0_color)
                    elif end_angle > 22.5:
                        draw_filled_ellipse(frame, (elbow_x, elbow_y), r, start_angle, 22.5, oct_1_color)
                        draw_filled_ellipse(frame, (elbow_x, elbow_y), r, 22.5, end_angle, oct_1_color)
                    else:
                        draw_filled_ellipse(frame, (elbow_x, elbow_y), r, start_angle, -22.5, oct__1_color)
                        draw_filled_ellipse(frame, (elbow_x, elbow_y), r, -22.5, end_angle, oct__1_color)

                    cv2.ellipse(frame, (elbow_x, elbow_y), r, 0, start_angle, end_angle, arc_color, 1, cv2.LINE_AA)
                    draw_filled_ellipse_with_text(frame, (elbow_x, elbow_y), (50, 50), -22.5, 22.5, oct_0_color, text="oct=0")
                    draw_filled_ellipse_with_text(frame, (elbow_x, elbow_y), (50, 50), -22.5, -67.5, oct__1_color, text="oct=1")
                    draw_filled_ellipse_with_text(frame, (elbow_x, elbow_y), (50, 50), 22.5, 67.5, oct_1_color, text="oct=-1")

                except Exception as e:
                    if "broadcast input array" not in str(e):
                        print("Error on tracking:", e)

            else:
                velocity = 0

            if midi_note is not None:
                if last_midi_note != midi_note:
                    if last_midi_note is not None:
                        self.app.osc_client.send_message("/note_off", [])
                    if velocity >= self.app.min_velocity_to_trigger_note_on:
                        self.app.osc_client.send_message("/note", [int(midi_note), int(velocity), int(octave)])
                    else:
                        self.app.osc_client.send_message("/note_off", [])
                elif last_midi_note == midi_note and last_velocity < self.app.min_velocity_to_trigger_note_on <= velocity:
                    self.app.osc_client.send_message("/note", [int(midi_note), int(velocity), int(octave)])
                elif last_midi_note == midi_note and velocity < self.app.min_velocity_to_trigger_note_on <= last_velocity:
                    self.app.osc_client.send_message("/note_off", [])
            else:
                self.app.osc_client.send_message("/note_off", [])

            last_midi_note = midi_note
            last_velocity = velocity

            frame = cv2.flip(frame, 1)
            add_text_to_frame(frame, text)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("PyChiro", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False


class CameraApp:
    def __init__(self):
        # State
        self.base_note = 60
        self.current_scale = major_scale
        self.min_velocity_to_trigger_note_on = 40
        self.max_elbow_wrist_x_distance = 0.0
        self.min_elbow_wrist_x_distance = 1.0
        self.note_vector = []
        self.note_vector_size = 11

        # OSC out (to Max)
        self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 7400)

        # OSC in (from Max)
        disp = dispatcher.Dispatcher()
        disp.map("/base_note", self.handle_base_note)
        disp.map("/scale", self.handle_scale)
        disp.map("/min_velocity", self.handle_min_velocity)
        self.osc_server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 7401), disp)
        osc_thread = threading.Thread(target=self.osc_server.serve_forever, daemon=True)
        osc_thread.start()

        signal.signal(signal.SIGINT, self.shutdown)

        # Start camera
        self.camera_thread = CameraThread(self)
        self.camera_thread.start()

    def handle_base_note(self, addr, value):
        self.base_note = int(value)
        print(f"Base note: {self.base_note}")

    def handle_scale(self, addr, value):
        self.current_scale = major_scale if int(value) == 0 else minor_scale
        print(f"Scale: {'major' if int(value) == 0 else 'minor'}")

    def handle_min_velocity(self, addr, value):
        self.min_velocity_to_trigger_note_on = int(value)
        print(f"Min velocity: {self.min_velocity_to_trigger_note_on}")

    def shutdown(self, sig, frame):
        print("\nShutting down...")
        self.camera_thread.stop()
        self.osc_server.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    app = CameraApp()
    try:
        app.camera_thread.join()
    except KeyboardInterrupt:
        app.shutdown(None, None)
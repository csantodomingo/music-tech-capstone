# ©2024, Francesco Roberto Dani
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
import torch.optim as optim
import torch.nn.functional as F
import rtmidi
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QLabel,
    QPushButton, QComboBox, QMenu, QSpinBox, QLineEdit, QCheckBox
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from pythonosc import udp_client


mp_drawing = mp.solutions.drawing_utils
notes_onehot_header = ["Do (C)", "Re (D)", "Mi (E)", "Fa (F)", "Sol (G)", "La (A)", "Si (B)"]
major_scale = [0, 2, 4, 5, 7, 9, 11]
minor_scale = [0, 2, 3, 5, 7, 8, 10]

def normalize_landmarks(landmarks, bbox):
    x_min, y_min, z_min = bbox[0]
    x_max, y_max, z_max = bbox[1]
    norm_landmarks = []
    for landmark in landmarks:
        norm_l = [(landmark.x - x_min) / (x_max - x_min), (landmark.y - y_min) / (y_max - y_min), (landmark.z - z_min) / (z_max - z_min)]
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
    for i in range(1, 5):  # Per ogni dito
        base = np.array(landmarks[0])  # Polso
        vec1 = np.array(landmarks[i * 4 - 3]) - base
        vec2 = np.array(landmarks[i * 4]) - np.array(landmarks[i * 4 - 3])
        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angles.append(np.arccos(cos_theta) / np.pi)
    return angles


def calculate_convex_hull_area(landmarks):
    points = np.array(landmarks)[:, :2]  # Usa solo x, y
    hull = ConvexHull(points)
    return hull.area


def extract_features(landmarks, bbox):
    # Normalizza e centra
    normalized = normalize_landmarks(landmarks, bbox)
    centered = center_landmarks(normalized)
    # Estrai feature
    distances = calculate_distances(centered)
    angles = calculate_angles(centered)
    convex_area = calculate_convex_hull_area(centered)
    # Combina
    return np.array(centered).flatten().tolist()[3:] + distances + angles + [convex_area]


def get_working_dir():
    if hasattr(sys, '_MEIPASS'):
        # Percorso durante l'esecuzione da un bundle PyInstaller
        base_path = sys._MEIPASS
    else:
        # Percorso durante l'esecuzione normale
        base_path = os.path.dirname(os.path.abspath(__file__))
    return base_path


"""class MeasuresNetworkKodalyC1C2_slim(nn.Module):
    def __init__(self):
        super(MeasuresNetworkKodalyC1C2_slim, self).__init__()
        self.dropout = nn.Dropout(0.25)
        # Define the architecture of the neural network
        self.fc1 = nn.Linear(275, 1024)  # 275
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 7)  # Output size: 7 (NO C2)

        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        init.xavier_uniform_(self.fc4.weight)
        init.xavier_uniform_(self.fc5.weight)

        init.constant_(self.fc1.bias, 0.0)
        init.constant_(self.fc2.bias, 0.0)
        init.constant_(self.fc3.bias, 0.0)
        init.constant_(self.fc4.bias, 0.0)
        init.constant_(self.fc5.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x"""


class MeasuresNetworkKodalyC1C2_slim(nn.Module):
    def __init__(self):
        super(MeasuresNetworkKodalyC1C2_slim, self).__init__()
        self.dropout = nn.Dropout(0.25)
        # Define the architecture of the neural network
        self.fc1 = nn.Linear(275, 8192)  # 275
        self.fc2 = nn.Linear(8192, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 7)  # Output size: 7 (NO C2)

        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        init.xavier_uniform_(self.fc4.weight)
        init.xavier_uniform_(self.fc5.weight)

        init.constant_(self.fc1.bias, 0.0)
        init.constant_(self.fc2.bias, 0.0)
        init.constant_(self.fc3.bias, 0.0)
        init.constant_(self.fc4.bias, 0.0)
        init.constant_(self.fc5.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def predict_single_input(model, input_vector):
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation during prediction
        # Convert the input vector to a PyTorch tensor
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)

        # Reshape the input tensor to match the expected shape (batch_size=1, input_size)
        input_tensor = input_tensor.unsqueeze(0)  # Shape: (1, N)

        # Forward pass through the model to get the predictions
        predictions = model(input_tensor)

        # Option 1: Get the predicted class (using argmax)
        predicted_class = torch.argmax(predictions, dim=1).item()  # Return the index of the highest logit

        # Option 2: Get the probabilities (using softmax)
        probabilities = F.softmax(predictions, dim=1).squeeze()  # Apply softmax to get probabilities

        # Return both predicted class and probabilities
        return predicted_class, probabilities.numpy()


def add_text_to_frame(frame, text="---"):
    # Add text to the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    h, w, d = frame.shape
    color = (20, 20, 20)
    text_color_bg = (255, 255, 255, 127)
    thickness = 1
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    org = (0, h - text_h)
    cv2.rectangle(frame, (org[0], org[1] - 45), (org[0] + text_w, org[1] + text_h), text_color_bg, -1)
    cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def draw_filled_ellipse(frame, center, radius, start_angle, end_angle, color, transparency=0.5):
    """Funzione per disegnare un arco pieno con trasparenza in un frame di opencv """
    overlay = frame.copy()
    cv2.ellipse(
        overlay,
        center,
        radius,
        0,
        start_angle,
        end_angle,
        color,
        thickness=-1,
        lineType=cv2.LINE_AA
    )
    # Applica la trasparenza
    cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0, frame)


def draw_filled_ellipse_cartesian(frame, start_point, end_point, center, color, transparency=0.5):
    """Funzione per disegnare un arco pieno tra due punti specifici con trasparenza in un frame di OpenCV."""

    # Calcolare i raggi lungo gli assi X e Y
    # Raggio lungo l'asse maggiore (X)
    radius_x = np.linalg.norm(np.array(start_point) - np.array(center))
    # Raggio lungo l'asse minore (Y)
    radius_y = np.linalg.norm(np.array(end_point) - np.array(center))

    # Calcola gli angoli di inizio e fine basati sui punti dati
    start_angle = np.degrees(np.arctan2(start_point[1] - center[1], start_point[0] - center[0]))
    end_angle = np.degrees(np.arctan2(end_point[1] - center[1], end_point[0] - center[0]))

    # Se l'angolo di fine è più piccolo dell'angolo di inizio, lo correggiamo per farlo circolare correttamente
    if end_angle < start_angle:
        end_angle += 360

    # Crea una copia del frame per disegnare sopra con trasparenza
    overlay = frame.copy()

    # Disegna l'ellisse (arco pieno)
    cv2.ellipse(
        overlay,
        center,
        (int(radius_x), int(radius_y)),  # I raggi calcolati
        0,
        start_angle,
        end_angle,
        color,
        thickness=-1,
        lineType=cv2.LINE_AA
    )

    # Applica la trasparenza
    cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0, frame)


def draw_filled_ellipse_with_text(frame, center, radius, start_angle, end_angle, color, transparency=0.5, text=""):
    overlay = frame.copy()
    cv2.ellipse(
        overlay,
        center,
        radius,
        0,
        start_angle,
        end_angle,
        color,
        thickness=-1,  # Riempie l'arco
        lineType=cv2.LINE_AA
    )
    # Applica la trasparenza
    cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0, frame)

    # Calcola la posizione del testo
    angle_mid = (start_angle + end_angle) / 2  # Angolo medio per posizionare il testo
    text_x = int(center[0] + radius[0] * np.cos(np.radians(angle_mid)))
    text_y = int(center[1] + radius[1] * np.sin(np.radians(angle_mid)))

    # Aggiungi il testo "oct=%d"
    # cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)
    draw_mirrored_text(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)


def draw_mirrored_text(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, color=(255, 255, 255), thickness=1):
    # Crea un'immagine vuota dove disegnare il testo
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_img = np.zeros((text_size[1], text_size[0], 3), dtype=np.uint8)  # Immagine vuota per il testo

    # Disegna il testo sull'immagine vuota
    cv2.putText(text_img, text, (0, text_size[1]), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    # Capovolgi l'immagine orizzontalmente
    mirrored_text = cv2.flip(text_img, 1)

    # Posiziona l'immagine specchiata sul frame
    text_x = position[0] - mirrored_text.shape[1]
    text_y = position[1]

    # Aggiungi il testo rovesciato al frame
    frame[text_y:text_y + mirrored_text.shape[0], text_x:text_x + mirrored_text.shape[1]] = mirrored_text


class ProbabilityHistogram(QLabel):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Probabilità per le Note Musicali")
        # self.setGeometry(100, 100, 400, 400)
        self.setFixedHeight(200)
        self.note_labels = notes_onehot_header
        self.probabilities = [0] * len(self.note_labels)  # Expected to be a list of probabilities (size 12 for the notes Do-Si)
        self.gesture_images = {  # Associare immagini ai gesti per le note
            'Do (C)': QImage(os.path.join(os.path.join(get_working_dir(), "img"), "Do.png")),
            'Re (D)': QImage(os.path.join(os.path.join(get_working_dir(), "img"), "Re.png")),
            'Mi (E)': QImage(os.path.join(os.path.join(get_working_dir(), "img"), "Mi.png")),
            'Fa (F)': QImage(os.path.join(os.path.join(get_working_dir(), "img"), "Fa.png")),
            'Sol (G)': QImage(os.path.join(os.path.join(get_working_dir(), "img"), "Sol.png")),
            'La (A)': QImage(os.path.join(os.path.join(get_working_dir(), "img"), "La.png")),
            'Si (B)': QImage(os.path.join(os.path.join(get_working_dir(), "img"), "Si.png"))
        }

    def set_probabilities(self, probabilities):
        self.probabilities = probabilities
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(255, 255, 255), 2))

        # Background
        painter.fillRect(self.rect(), QColor(0, 0, 0))  # black background

        # Colori per le note
        note_colors = {
            'Do (C)': QColor(255, 0, 0, 200),  # Red with transparency
            'Re (D)': QColor(139, 69, 19, 200),  # Brown with transparency
            'Mi (E)': QColor(255, 182, 193, 200),  # Salmon pink with transparency
            'Fa (F)': QColor(144, 238, 144, 200),  # Light green with transparency
            'Sol (G)': QColor(135, 206, 235, 200),  # Light blue with transparency
            'La (A)': QColor(138, 43, 226, 200),  # Violet with transparency
            'Si (B)': QColor(255, 105, 180, 200)  # Hot pink with transparency
        }

        # Draw histogram
        num_notes = len(self.probabilities)
        width = self.width()
        bar_width = width // num_notes

        for i, prob in enumerate(self.probabilities):
            bar_height = int(prob * (self.height() - 30))  # Scale bar height (leaving space for labels)
            x = i * bar_width
            y = self.height() - bar_height - 20  # Make space for labels at the bottom

            # Set color for the current note (with transparency)
            note_name = self.note_labels[i]
            painter.setBrush(note_colors.get(note_name, note_colors[self.note_labels[i]]))  # Default to cyan if not found

            # Draw bar with rounded corners
            painter.setPen(QPen(QColor(255, 255, 255), 2))  # White border
            painter.drawRoundedRect(x, y, bar_width - 4, bar_height, 5, 5)  # Rounded corners

            # Draw note labels above the bars
            painter.setPen(QPen(QColor(255, 255, 255)))  # White text
            note_text = self.note_labels[i]
            text_width = painter.fontMetrics().horizontalAdvance(note_text)
            text_x = x + (bar_width - text_width) // 2  # Center text
            text_y = self.height() - 8  # Place text just above the bars
            painter.drawText(text_x, text_y, note_text)

            # Draw gesture image under each bar
            if note_name in self.gesture_images:
                gesture_image = self.gesture_images[note_name]
                image_width = 50  # Fixed width for all images
                image_height = 50  # Fixed height for all images
                image_x = x + (bar_width - image_width) // 2  # Center the image
                # image_y = y + bar_height  # Position the image just below the bar
                image_y = int(self.height() / 2) - int(image_height / 2)
                painter.drawImage(image_x, image_y, gesture_image.scaled(image_width, image_height))

        painter.end()


class CameraThread(QThread):
    frame_ready = pyqtSignal(object)  # Signal to emit frames
    probabilities = pyqtSignal(object)  # Signal to emit model inference probabilities

    def __init__(self, main_app):
        super().__init__()
        self.running = False
        self.cap = None
        self.main_app = main_app
        # Load PyTorch model
        # checkpoint_path = os.path.join(os.path.join(get_working_dir(), "checkpoints"), "checkpoint_epoch_10_dopo_aggiunta_08.pt")
        checkpoint_path = os.path.join(os.path.join(get_working_dir(), "checkpoints"), "checkpoint_epoch_10.pt")
        print(f"loading model with checkpoint: {checkpoint_path} ...")
        self.model = MeasuresNetworkKodalyC1C2_slim()
        self.model.eval()  # Set the model to evaluation mode
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"checkpoint '{checkpoint_path}' successfully loaded!")
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                                  enable_segmentation=True, smooth_segmentation=True,
                                                  model_complexity=1)
        self.velocity_holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                                  enable_segmentation=True, smooth_segmentation=False, smooth_landmarks=False,
                                                  model_complexity=1)

    def compute_velocity(self, results):
        mp_holistic = mp.solutions.holistic
        # Calcola la velocity
        velocity = 0  # Default nel caso di valori di calibrazione non validi
        if results.pose_landmarks and results.right_hand_landmarks:
            # Trova polso, gomito e spalla del braccio destro
            right_elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
            right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            right_wrist = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]

            upper_arm_length = math.sqrt(
                (right_elbow.x - right_shoulder.x) ** 2 +
                (right_elbow.y - right_shoulder.y) ** 2 +
                (right_elbow.z - right_shoulder.z) ** 2
            ) / 1.75

            lower_arm_length = math.sqrt(
                (right_elbow.x - right_wrist.x) ** 2 +
                (right_elbow.y - right_wrist.y) ** 2 +
                (right_elbow.z - right_wrist.z) ** 2
            )

            shoulder_length = math.sqrt(
                (right_shoulder.x - left_shoulder.x) ** 2 +
                (right_shoulder.y - left_shoulder.y) ** 2 +
                (right_shoulder.z - left_shoulder.z) ** 2
            )

            # Calibrazione distanza in X tra polso e gomito
            # Calcolo della distanza lungo l'asse X
            current_distance_x = abs(right_wrist.x - right_elbow.x)

            # Aggiornamento dei valori di calibrazione
            if str(self.main_app.velocity_mode_combo.currentText()) == "Dynamic Adaptive":  # Calibrazione dinamica
                self.main_app.max_elbow_wrist_x_distance = max(self.main_app.max_elbow_wrist_x_distance, current_distance_x)
                self.main_app.min_elbow_wrist_x_distance = min(self.main_app.min_elbow_wrist_x_distance, current_distance_x)
            elif str(self.main_app.velocity_mode_combo.currentText()) == "Upper Arm":  # In base a lunghezza avambraccio
                # print("max_elbow_wrist_x_distance is:", self.main_app.max_elbow_wrist_x_distance)
                self.main_app.max_elbow_wrist_x_distance = max(self.main_app.max_elbow_wrist_x_distance, upper_arm_length)
                self.main_app.min_elbow_wrist_x_distance = min(self.main_app.min_elbow_wrist_x_distance, upper_arm_length / 2)
            elif str(self.main_app.velocity_mode_combo.currentText()) == "Lower Arm":  # In base a lunghezza braccio
                self.main_app.max_elbow_wrist_x_distance = max(self.main_app.max_elbow_wrist_x_distance, lower_arm_length * 0.75)
                self.main_app.min_elbow_wrist_x_distance = min(self.main_app.min_elbow_wrist_x_distance, lower_arm_length / 2)
            elif str(self.main_app.velocity_mode_combo.currentText()) == "Shoulder":  # In base a lunghezza spalle
                self.main_app.max_elbow_wrist_x_distance = shoulder_length * 0.75
                self.main_app.min_elbow_wrist_x_distance = shoulder_length / 3

            # Verifica che i valori di calibrazione siano validi
            if self.main_app.max_elbow_wrist_x_distance > self.main_app.min_elbow_wrist_x_distance:
                # Mappatura della distanza a velocity MIDI (0-127)
                normalized_distance = (current_distance_x - self.main_app.min_elbow_wrist_x_distance) / (
                        self.main_app.max_elbow_wrist_x_distance - self.main_app.min_elbow_wrist_x_distance
                )
                if normalized_distance < 0:
                    normalized_distance = 0
                if normalized_distance > 1:
                    normalized_distance = 1
                velocity = int(((1 - normalized_distance) ** 2) * 127)  # Invertiamo il range
                velocity = max(0, min(velocity, 127))  # Clipping per garantire il range MIDI
                self.main_app.midi_out.send_message([0xB0, 7, velocity])  # 0xB0: Control Change, CC7 (Volume)
                # print("Velocity is: ", velocity)
        return velocity

    def compute_octave(self, results):
        mp_holistic = mp.solutions.holistic
        octave = 0
        if results.pose_landmarks and results.right_hand_landmarks:
            # Trova polso e gomito del braccio destro
            right_elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
            right_wrist = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]

            # Vettore dal gomito al polso
            elbow_wrist_vector = [right_wrist.x - right_elbow.x, right_wrist.y - right_elbow.y]

            # Vettore orizzontale (asse X)
            horizontal_vector = [1, 0]

            # Calcola il prodotto scalare tra il vettore gomito-polso e il vettore orizzontale
            dot_product = elbow_wrist_vector[0] * horizontal_vector[0] + elbow_wrist_vector[1] * horizontal_vector[1]

            # Calcola la norma del vettore gomito-polso e del vettore orizzontale
            norm_elbow_wrist = math.sqrt(elbow_wrist_vector[0] ** 2 + elbow_wrist_vector[1] ** 2)
            norm_horizontal = math.sqrt(horizontal_vector[0] ** 2 + horizontal_vector[1] ** 2)

            # Calcola l'angolo tra il vettore gomito-polso e l'asse orizzontale (in radianti)
            angle = math.acos(dot_product / (norm_elbow_wrist * norm_horizontal))

            # Determina la direzione del braccio (se sopra o sotto)
            # Utilizza il determinante per determinare la "sotto" o "sopra" posizione rispetto all'orizzontale
            cross_product = elbow_wrist_vector[0] * horizontal_vector[1] - elbow_wrist_vector[1] * horizontal_vector[0]

            # Converti l'angolo in gradi
            angle_degrees = math.degrees(angle)

            # Se il cross product è negativo, il braccio è sotto l'orizzontale
            if cross_product < 0 and angle_degrees > 25:
                octave = -1  # braccio abbassato
            elif angle_degrees < 45:  # Braccio orizzontale (angolo tra -45 e 45 gradi)
                octave = 0
            else:  # Braccio alzato (angolo maggiore di 45 gradi rispetto all'orizzontale)
                octave = 1
        # print("Octave: " + str(octave))
        return octave

    def compute_median_midi_note(self, midi_note):
        note = None
        if midi_note is not None:
            self.main_app.note_vector.append(midi_note)
            if len(self.main_app.note_vector) > self.main_app.note_vector_size:
                self.main_app.note_vector = self.main_app.note_vector[1:]
            median = statistics.median(self.main_app.note_vector)
            if self.main_app.note_vector.count(median) >= int(self.main_app.note_vector_size / 2):
                return note
            else:
                return None
        else:
            return None

    def calibrate_velocity(self, frame):
        mp_holistic = mp.solutions.holistic
        frame = cv2.flip(frame, 1)  # Ora la mano destra è la sinistra (e viceversa)
        results = self.velocity_holistic.process(frame)
        if results.right_hand_landmarks:
            right_hand_landmarks = results.right_hand_landmarks.landmark
            # Calcolare la bounding box
            min_x = min(landmark.x for landmark in right_hand_landmarks)
            max_x = max(landmark.x for landmark in right_hand_landmarks)
            min_y = min(landmark.y for landmark in right_hand_landmarks)
            max_y = max(landmark.y for landmark in right_hand_landmarks)
            min_z = min(landmark.z for landmark in right_hand_landmarks)
            max_z = max(landmark.z for landmark in right_hand_landmarks)

            # Convertire le coordinate normalizzate in pixel
            h, w, _ = frame.shape
            bbox = (int(min_x * w), int(min_y * h), int(min_z * h)), (
                int(max_x * w), int(max_y * h), int(max_z * h))
            try:
                right_hand_measures = extract_features(right_hand_landmarks, bbox)
                true_val, output = predict_single_input(self.model, right_hand_measures)
                print("prediction on left hand: ", true_val)
                frame = cv2.flip(frame, 1)  # Ora la mano destra è la destra (e viceversa)
                results = self.velocity_holistic.process(frame)
                if results.pose_landmarks and results.right_hand_landmarks:
                    # Trova polso e gomito del braccio destro
                    right_elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
                    right_wrist = results.right_hand_landmarks.landmark[
                        mp_holistic.HandLandmark.WRIST]  # più preciso rispetto a pose_landmarks
                    if true_val == 0:  # Con nota Do (C) setto la max_distance
                        self.main_app.max_elbow_wrist_x_distance = abs(right_wrist.x - right_elbow.x)
                    if true_val == 6:  # Con nota Si (B) setto la min_distance
                        self.main_app.min_elbow_wrist_x_distance = abs(right_wrist.x - right_elbow.x)
            except:
                pass

    def run(self):
        """Start processing the video and send MIDI messages."""
        if self.main_app.current_device_index < 0:
            print("No MIDI device selected!")
            # return

        # Inizializza la nota come None
        last_midi_note = None
        last_velocity = None

        # Start camera feed
        camera = self.main_app.select_camera_line_edit.text()
        if camera.isdigit():
            camera = int(camera)
        self.cap = cv2.VideoCapture(camera)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            midi_note = None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (960, 540))

            if self.main_app.manual_velocity_calibration_enabled:
                self.calibrate_velocity(frame)

            text = "---"
            results = self.holistic.process(frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (640, 360))  # Resize to QLabel dimensions

            # Calcola la nota
            if results.right_hand_landmarks:
                right_hand_landmarks = results.right_hand_landmarks.landmark
                # Calcolare la bounding box
                min_x = min(landmark.x for landmark in right_hand_landmarks)
                max_x = max(landmark.x for landmark in right_hand_landmarks)
                min_y = min(landmark.y for landmark in right_hand_landmarks)
                max_y = max(landmark.y for landmark in right_hand_landmarks)
                min_z = min(landmark.z for landmark in right_hand_landmarks)
                max_z = max(landmark.z for landmark in right_hand_landmarks)

                # Convertire le coordinate normalizzate in pixel
                h, w, _ = frame.shape
                bbox = (int(min_x * w), int(min_y * h), int(min_z * h)), (
                    int(max_x * w), int(max_y * h), int(max_z * h))
                try:
                    right_hand_measures = extract_features(right_hand_landmarks, bbox)
                    true_val, output = predict_single_input(self.model, right_hand_measures)
                    # print(f"Output type is {type(output)}: {output}")
                    self.probabilities.emit(output)  # Emit the probabilities found by the model
                    self.main_app.osc_client.send_message("/probabilities", output.tolist())
                    octave = self.compute_octave(results)
                    midi_note = self.main_app.current_scale[np.argmax(output)] + self.main_app.base_note + (octave * 12)
                    midi_note = self.main_app.compute_median_midi_note(midi_note)
                    # Calcola la velocity in base alla distanza della proiezione sull'asse X di polso e gomito
                    velocity = self.compute_velocity(results)
                    if midi_note is not None:
                        text = notes_onehot_header[np.argmax(output)] + f"{octave}; v:{velocity}"
                        self.main_app.osc_client.send_message("/note", [int(midi_note), int(velocity), int(octave)])

                    # Disegna a schermo i punti di pose_landmarks di wrist ed elbow
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 250, 0), thickness=2, circle_radius=4))
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                              connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 250, 0),
                                                                                             thickness=2,
                                                                                             circle_radius=4))

                    # Trova polso e gomito del braccio destro
                    right_elbow = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_ELBOW]
                    right_wrist = results.right_hand_landmarks.landmark[
                        self.mp_holistic.HandLandmark.WRIST]  # più preciso rispetto a pose_landmarks
                    h, w, _ = frame.shape  # Ottieni dimensioni del frame

                    # Converti in coordinate pixel
                    elbow_x, elbow_y = int(right_elbow.x * w), int(right_elbow.y * h)
                    wrist_x, wrist_y = int(right_wrist.x * w), int(right_wrist.y * h)

                    # Disegna il gomito
                    cv2.circle(frame, (elbow_x, elbow_y), radius=4, color=(255, 255, 255),
                               thickness=-1, lineType=cv2.LINE_AA)  # contorno bianco
                    cv2.circle(frame, (elbow_x, elbow_y), radius=3, color=(0, 0, 255),
                               thickness=-1, lineType=cv2.LINE_AA)  # interno blu


                    # Calcola l'angolo tra le due linee
                    dx = wrist_x - elbow_x
                    dy = elbow_y - wrist_y  # Inverti Y perché l'origine è in alto a sinistra
                    angle_radians = np.arctan2(dy, dx)  # Angolo in radianti
                    angle_degrees = np.degrees(angle_radians)  # Converti in gradi

                    # Normalizza l'angolo tra 0 e 360 gradi
                    if angle_degrees < 0:
                        angle_degrees += 360

                    # Determina l'intervallo dell'arco (angolo minore tra 0 e 360°)
                    if angle_degrees > 180:
                        start_angle = 0
                        end_angle = angle_degrees - 360  # Angolo negativo per il verso opposto
                    else:
                        start_angle = 0
                        end_angle = angle_degrees
                    end_angle = -end_angle

                    oct_0_color = (0, 255, 0)
                    oct_1_color = (255, 255, 0)
                    oct__1_color = (0, 255, 255)

                    if -22.5 <= end_angle <= 22.5:
                        arc_color = oct_0_color
                    elif end_angle > 22.5:
                        arc_color = oct_1_color
                    else:
                        arc_color = oct__1_color

                    # Disegna una linea tra gomito e polso
                    cv2.line(frame, (elbow_x, elbow_y), (wrist_x, wrist_y), color=arc_color, thickness=1, lineType=cv2.LINE_AA)  # Linea gialla

                    # Disegna una linea orizzontale dal gomito alla proiezione X del polso
                    cv2.line(frame, (elbow_x, elbow_y), (wrist_x, elbow_y), color=arc_color, thickness=1, lineType=cv2.LINE_AA)

                    filled_ellipse_radius = (abs(wrist_x - elbow_x), abs(wrist_x - elbow_x))
                    if -22.5 <= end_angle <= 22.5:
                        draw_filled_ellipse(frame, (elbow_x, elbow_y), filled_ellipse_radius, start_angle, end_angle, oct_0_color, transparency=0.5)
                    elif end_angle > 22.5:
                        draw_filled_ellipse(frame, (elbow_x, elbow_y), filled_ellipse_radius, start_angle, 22.5, oct_1_color, transparency=0.5)
                        draw_filled_ellipse(frame, (elbow_x, elbow_y), filled_ellipse_radius, 22.5, end_angle, oct_1_color, transparency=0.5)
                    else:
                        draw_filled_ellipse(frame, (elbow_x, elbow_y), filled_ellipse_radius, start_angle, -22.5, oct__1_color, transparency=0.5)
                        draw_filled_ellipse(frame, (elbow_x, elbow_y), filled_ellipse_radius, -22.5, end_angle, oct__1_color, transparency=0.5)
                        # draw_filled_ellipse_cartesian(frame, (wrist_x, wrist_y), (elbow_x + 20, elbow_y + 20), (elbow_x, elbow_y), oct__1_color, transparency=0.5)

                    # Disegna l'arco per rappresentare l'angolo
                    cv2.ellipse(
                        frame,
                        (elbow_x, elbow_y),
                        filled_ellipse_radius,
                        0,
                        start_angle,
                        end_angle,  # Disegna solo l'arco tra 0 e l'angolo calcolato
                        arc_color,
                        thickness=1,
                        lineType=cv2.LINE_AA
                    )

                    # Disegna le fette di ellisse per i range di inclinazione delle ottave
                    draw_filled_ellipse_with_text(frame, (elbow_x, elbow_y), (50, 50), -22.5, 22.5, oct_0_color,
                                        transparency=0.5, text="oct=0")  # Verde
                    draw_filled_ellipse_with_text(frame, (elbow_x, elbow_y), (50, 50), -22.5, -67.5, oct__1_color,
                                        transparency=0.5, text="oct=1")  # Giallo
                    draw_filled_ellipse_with_text(frame, (elbow_x, elbow_y), (50, 50), 22.5, 67.5, oct_1_color,
                                        transparency=0.5, text="oct=-1")  # Rosso

                except Exception as e:
                    print("Error on tracking...", e)
            else:
                velocity = 0

            if midi_note is not None:
                try:
                    # Se trovo una nuova nota
                    if last_midi_note != midi_note:
                        if last_midi_note is not None:
                            self.main_app.all_notes_off()
                            self.main_app.osc_client.send_message("/note_off", [])
                        if velocity >= self.main_app.min_velocity_to_trigger_note_on:  # Se la velocity è sopra la soglia minima per triggerare un NoteON
                            self.main_app.midi_out.send_message([0x90, int(midi_note), velocity])  # Note On corrente quando nota cambia
                        else:  # Altrimenti spegni la nota
                            self.main_app.all_notes_off()
                            self.main_app.osc_client.send_message("/note_off", [])
                    # Se continua una nota ma la velocity sale sopra la soglia, manda il NoteON
                    elif last_midi_note == midi_note and last_velocity < self.main_app.min_velocity_to_trigger_note_on <= velocity:
                        self.main_app.midi_out.send_message([0x90, int(midi_note), velocity])  # Note On corrente quando nota cambia
                    # Se continua una nota ma la velocity scende sotto la soglia, manda il NoteOFF
                    elif last_midi_note == midi_note and velocity < self.main_app.min_velocity_to_trigger_note_on <= last_velocity:
                        self.main_app.all_notes_off()
                        self.main_app.osc_client.send_message("/note_off", [])
                except:
                    print("NO MIDI OUT CONNECTED")
            else:
                self.main_app.all_notes_off()
                self.main_app.osc_client.send_message("/note_off", [])
            last_midi_note = midi_note
            last_velocity = velocity
            frame = cv2.flip(frame, 1)  # Mirror horizontally before adding text
            add_text_to_frame(frame, text)
            self.frame_ready.emit(frame)  # Emit the captured frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def stop(self):
        """Stop the camera thread."""
        self.running = False
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
        self.wait()


class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyChiro: Kodály Chironomic Method - ©2024, Francesco Roberto Dani")
        self.setGeometry(100, 100, 980, 360)  # Default window size, will adjust dynamically later
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                min-height: 900px;
            }
            QLabel {
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QLabel#frame {
                color: white;
                font-size: 14px;
                font-weight: bold;
                min-width: 640px;
                min-height: 360px;
                border-radius: 5px;
            }
            QLineEdit, QComboBox, QSpinBox, QPushButton {
                background-color: #333;
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
                padding: 5px;
                font-size: 12px;
            }
            QComboBox::drop-down {
                background-color: #333;
                border: 2px solid #555;
            }
            QComboBox QAbstractItemView {
                background-color: #444;
                color: white;
                border: 2px solid #555;
            }

            QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
                border: 2px solid #88aaff;
            }
            QPushButton {
                background-color: #3b5998;
                color: white;
                font-weight: bold;
            }
            QPushButton:pressed {
                background-color: #2d4373;
            }
            QPushButton:checked {
                background-color: #ff5722;
            }
            QComboBox, QSpinBox {
                min-width: 100px;
                min-height: 15px;
            }
            QVBoxLayout {
                spacing: 10px;
            }
            QHBoxLayout {
                spacing: 15px;
            }
        """)

        # Definisci distanza minima e massima (invertite) tra polso e gomito per calibrazione automatica
        self.max_elbow_wrist_x_distance = 0.0
        self.min_elbow_wrist_x_distance = 1.0
        self.note_vector = []
        self.note_vector_size = 21
        self.manual_velocity_calibration_enabled = False

        # Setup other components as before...
        self.device_label = QLabel("Select MIDI Output Device:")
        self.device_combo = QComboBox()
        self.start_button = QPushButton("Start Video Processing")
        self.start_button.setCheckable(True)
        self.menu_button = QPushButton("MIDI Devices Menu")
        self.menu_scale = QPushButton("Musical Scale")
        self.current_scale = major_scale

        self.base_note_spin = QSpinBox()
        self.base_note_spin.setMinimum(0)
        self.base_note_spin.setMaximum(127)
        self.base_note = 60
        self.base_note_spin.setValue(self.base_note)
        self.base_note_lbl = QLabel("Base Note:")
        base_note_layout = QHBoxLayout()
        base_note_layout.addWidget(self.base_note_lbl)
        base_note_layout.addWidget(self.base_note_spin)

        # Velocity Management section
        velocity_management_label = QLabel("Velocity Management:")
        self.velocity_mode_combo = QComboBox()
        self.velocity_mode_combo.addItems(["Dynamic Adaptive", "Manual Calibration", "Upper Arm", "Lower Arm", "Shoulder"])
        # Velocity Calibration checkbox
        self.checkbox = QCheckBox('Enable Manual Calibration')

        velocity_management_layout = QHBoxLayout()
        velocity_management_layout.addWidget(velocity_management_label)
        velocity_management_layout.addWidget(self.checkbox)
        velocity_management_layout.addWidget(self.velocity_mode_combo)

        self.min_velocity_spin = QSpinBox()
        self.min_velocity_spin.setMinimum(0)
        self.min_velocity_spin.setMaximum(100)
        self.min_velocity_to_trigger_note_on = 40
        self.min_velocity_spin.setValue(self.min_velocity_to_trigger_note_on)
        self.min_velocity_lbl = QLabel("Min Velocity to Trigger Note:")
        min_velocity_layout = QHBoxLayout()
        min_velocity_layout.addWidget(self.min_velocity_lbl)
        min_velocity_layout.addWidget(self.min_velocity_spin)

        # Video selection
        camera_h_layout = QHBoxLayout()
        self.select_camera_label = QLabel("Camera ID or Video path:")
        self.select_camera_line_edit = QLineEdit()
        # self.select_camera_line_edit.setText("/Users/francescodani/Movies/PyChiro_Data/Unprocessed/Notes/A1_08.mov")
        self.select_camera_line_edit.setText("0")
        camera_h_layout.addWidget(self.select_camera_label)
        camera_h_layout.addWidget(self.select_camera_line_edit)
        camera_h_layout.addWidget(self.start_button)

        # Video Display
        self.video_label = QLabel("Video Feed")
        self.video_label.setObjectName("frame")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border-radius: 10px; padding: 10px;")

        # Probability Histogram
        self.prob_hist = ProbabilityHistogram()

        # Central Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.prob_hist)
        layout.addLayout(camera_h_layout)
        layout.addLayout(base_note_layout)
        layout.addLayout(velocity_management_layout)
        layout.addLayout(min_velocity_layout)
        layout.addWidget(self.device_label)
        layout.addWidget(self.device_combo)
        layout.addWidget(self.menu_button)
        layout.addWidget(self.menu_scale)
        # layout.addWidget(self.start_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # MIDI setup
        self.midi_out = rtmidi.MidiOut()
        self.current_device_index = -1
        self.refresh_midi_devices()
        self.change_device(0)

        # OSC setup
        self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 7400)

        # Connections
        self.base_note_spin.valueChanged.connect(self.change_base_note)
        self.min_velocity_spin.valueChanged.connect(self.change_min_velocity)
        self.menu_button.clicked.connect(self.show_device_menu)
        self.menu_scale.clicked.connect(self.show_scale_menu)
        self.device_combo.currentIndexChanged.connect(self.change_device)
        self.start_button.clicked.connect(self.handle_toggle)
        self.checkbox.stateChanged.connect(self.on_checkbox_state_changed)

        # Timer to refresh devices every 2 seconds
        self.device_refresh_timer = QTimer(self)
        self.device_refresh_timer.timeout.connect(self.refresh_midi_devices)
        self.device_refresh_timer.start(2000)

        # Camera thread
        self.camera_thread = CameraThread(self)
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.probabilities.connect(self.update_probabilities)

    def on_checkbox_state_changed(self, state):
        """Funzione chiamata quando lo stato della checkbox cambia."""
        if state == 2:
            self.manual_velocity_calibration_enabled = True
        else:
            self.manual_velocity_calibration_enabled = False

    def change_base_note(self):
        """ Change base note """
        self.base_note = int(self.base_note_spin.value())

    def change_min_velocity(self):
        """ Change min velocity to trigger a NoteON """
        self.min_velocity_to_trigger_note_on = int(self.min_velocity_spin.value())

    def refresh_midi_devices(self):
        """Refresh the list of MIDI devices."""
        available_ports = self.midi_out.get_ports()
        current_ports = [self.device_combo.itemText(i) for i in range(self.device_combo.count())]
        if available_ports != current_ports:
            self.device_combo.clear()
            if available_ports:
                self.device_combo.addItems(available_ports)
            else:
                self.device_combo.addItem("No MIDI devices found")

    def show_device_menu(self):
        """Show a popup menu for selecting the MIDI device."""
        available_ports = self.midi_out.get_ports()
        menu = QMenu(self)
        for i, port in enumerate(available_ports):
            action = menu.addAction(port)
            action.triggered.connect(lambda _, index=i: self.change_device(index))
        menu.exec(self.menu_button.mapToGlobal(self.menu_button.rect().bottomLeft()))

    def show_scale_menu(self):
        """Show a popup menu for selecting the Musical scale."""
        menu = QMenu(self)
        for i, scale in enumerate(["Major", "Minor"]):
            action = menu.addAction(scale)
            action.triggered.connect(lambda _, index=i: self.change_scale(index))
        menu.exec(self.menu_button.mapToGlobal(self.menu_button.rect().bottomLeft()))

    def change_device(self, index):
        """Change the selected MIDI output device."""
        self.current_device_index = index
        if self.midi_out.is_port_open():
            self.midi_out.close_port()
        try:
            self.midi_out.close_port()
        except:
            pass
        if index >= 0:
            self.midi_out.open_port(index)
            print(f"Changed MIDI device to: {self.device_combo.itemText(index)}")

    def handle_toggle(self):
        if not self.start_button.isChecked():
            self.start_button.setText("Start Video Processing")
            print("Stopping video processing...")
            self.stop_camera()
        else:
            self.start_button.setText("Stop Video Processing")
            print("Starting video processing...")
            self.start_camera()

    def change_scale(self, index):
        """Change the selected Musical scale."""
        if index == 0:
            self.current_scale = major_scale
        elif index == 1:
            self.current_scale = minor_scale

    def start_camera(self):
        """Start the camera."""
        self.camera_thread.start()

    def stop_camera(self):
        """Stop the camera."""
        self.camera_thread.stop()

    def compute_median_midi_note(self, midi_note):
        if midi_note is not None:
            self.note_vector.append(midi_note)
            if len(self.note_vector) > self.note_vector_size:
                self.note_vector = self.note_vector[1:]
        return statistics.median(self.note_vector)

    def update_frame(self, frame):
        """ Display current video frame in the QLabel."""
        # Convert BGR to RGB
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        # Convert to QImage
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        # Set QImage to QLabel
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def update_probabilities(self, probabilities):
        """Update the probabilities of the histogram"""
        self.prob_hist.set_probabilities(probabilities)
        

    def all_notes_off(self):
        """ Note Off globale """
        for i in range(127):
            self.midi_out.send_message([0x80, i, 0])

    def closeEvent(self, event):
        """Handle window close event."""
        self.stop_camera()
        event.accept()


# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = CameraApp()
    main_window.show()
    # main_window.start_camera()
    sys.exit(app.exec())

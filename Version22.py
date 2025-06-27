import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
import face_recognition
import numpy as np
import os
import time
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ThreadPoolExecutor
import threading

# --- Initialisation globale ---
executor = ThreadPoolExecutor(max_workers=4)
trackers = {}
trackers_lock = threading.Lock()
next_tracker_id = 0

# --- Fonction pour ajouter une personne ---
def add_person():
    person_name = simpledialog.askstring("Ajouter une personne", "Entrez le nom complet :")
    if not person_name:
        return

    person_folder = os.path.join("images", person_name)
    os.makedirs(person_folder, exist_ok=True)

    existing_files = [f for f in os.listdir(person_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    next_index = len(existing_files) + 1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Erreur", "Impossible d'ouvrir la caméra")
        return

    messagebox.showinfo("Instructions", "Appuyez sur 's' pour sauvegarder une photo.\nAppuyez sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"Enregistrement photos de {person_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, "Appuyez sur 's' pour sauvegarder, 'q' pour quitter", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Ajouter une personne", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            file_path = os.path.join(person_folder, f"{next_index}.jpg")
            cv2.imwrite(file_path, frame)
            next_index += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Thread: reconnaissance faciale asynchrone ---
def recognize_face_async(face_roi, pos, box, tracker_id, known_face_encodings, known_face_names, max_distance):
    global trackers
    rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_face)
    if not encodings:
        return
    encoding = encodings[0]

    name = "Unknown"
    matches = face_recognition.compare_faces(known_face_encodings, encoding)
    distances = face_recognition.face_distance(known_face_encodings, encoding)

    if True in matches:
        idx = np.argmin(distances)
        if distances[idx] < max_distance:
            name = known_face_names[idx]

    with trackers_lock:
        trackers[tracker_id] = {
            'pos': pos,
            'box': box,
            'encoding': encoding,
            'name': name,
            'frames_since_update': 0
        }

# --- Fonction principale reconnaissance ---
def run_recognition():
    global trackers, next_tracker_id

    # Chargement visages connus
    known_face_encodings = []
    known_face_names = []
    for person_name in os.listdir("images"):
        folder = os.path.join("images", person_name)
        if not os.path.isdir(folder): continue

        encodings = []
        for img_file in os.listdir(folder):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image = face_recognition.load_image_file(os.path.join(folder, img_file))
                e = face_recognition.face_encodings(image)
                if e: encodings.append(e[0])

        if encodings:
            known_face_encodings.append(np.mean(encodings, axis=0))
            known_face_names.append(person_name.upper())

    # Détection avec DNN
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    cap = cv2.VideoCapture(0)

    max_distance = 0.6
    max_pos_distance = 100
    max_frames_no_update = 30

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        detected_positions, detected_boxes = [], []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                startX, startY, endX, endY = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                detected_positions.append(((startX + endX) // 2, (startY + endY) // 2))
                detected_boxes.append((startX, startY, endX, endY))

        updated_trackers = {}
        unmatched_detections = set(range(len(detected_positions)))
        with trackers_lock:
            unmatched_trackers = set(trackers.keys())

            if trackers and detected_positions:
                tracker_ids = list(trackers.keys())
                tracker_positions = np.array([trackers[tid]['pos'] for tid in tracker_ids])
                detection_positions = np.array(detected_positions)

                dist_matrix = np.linalg.norm(tracker_positions[:, None, :] - detection_positions[None, :, :], axis=2)
                dist_matrix[dist_matrix > max_pos_distance] = 1e6

                row_ind, col_ind = linear_sum_assignment(dist_matrix)

                for r, c in zip(row_ind, col_ind):
                    if dist_matrix[r, c] > max_pos_distance: continue
                    tid = tracker_ids[r]
                    updated_trackers[tid] = trackers[tid]
                    updated_trackers[tid]['pos'] = detected_positions[c]
                    updated_trackers[tid]['box'] = detected_boxes[c]
                    updated_trackers[tid]['frames_since_update'] = 0
                    unmatched_detections.discard(c)
                    unmatched_trackers.discard(tid)

            for tid in unmatched_trackers:
                trackers[tid]['frames_since_update'] += 1
                if trackers[tid]['frames_since_update'] < max_frames_no_update:
                    updated_trackers[tid] = trackers[tid]

        # Reconnaissance en pool
        for i in list(unmatched_detections):
            startX, startY, endX, endY = detected_boxes[i]
            face_roi = frame[startY:endY, startX:endX]
            if face_roi.size == 0: continue

            tid = next_tracker_id
            next_tracker_id += 1

            executor.submit(recognize_face_async, face_roi, detected_positions[i],
                            detected_boxes[i], tid, known_face_encodings, known_face_names, max_distance)

        with trackers_lock:
            trackers = updated_trackers.copy()
            for tid, data in trackers.items():
                cx, cy = data['pos']
                name = data.get('name', 'Unknown')
                cv2.circle(frame, (cx, cy), 40, (0, 255, 0), 2)
                cv2.putText(frame, name, (cx - 30, cy - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)

        fps = 1 / (time.time() - start_time + 1e-5)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Face Recognition Optimized", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Interface graphique ---
def main_interface():
    root = tk.Tk()
    root.title("Gestion Reconnaissance Faciale")
    root.geometry("300x150")

    def on_add_person():
        root.withdraw()
        add_person()
        root.deiconify()

    def on_start_recognition():
        root.destroy()
        run_recognition()
        main_interface()

    tk.Label(root, text="Choisissez une action :", font=("Arial", 12)).pack(pady=10)
    tk.Button(root, text="Ajouter une personne", width=20, command=on_add_person).pack(pady=5)
    tk.Button(root, text="Lancer la reconnaissance", width=20, command=on_start_recognition).pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main_interface()
import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Fungsi untuk mendapatkan gambar wajah dan label dari semua subfolder
def get_images_and_labels(main_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = []
    labels = []
    label_names = {}
    current_label = 0

    # Pastikan folder dataset ada
    if not os.path.exists(main_path):
        print(f"Error: Folder '{main_path}' tidak ditemukan.")
        return faces, labels, label_names

    # Iterasi melalui semua subfolder di folder utama
    for folder_name in os.listdir(main_path):
        folder_path = os.path.join(main_path, folder_name)

        if os.path.isdir(folder_path):
            label_names[current_label] = folder_name
            print(f"Processing folder: {folder_name} with label {current_label}")

            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                # Membaca gambar dalam grayscale
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Gambar '{image_name}' tidak valid, dilewati.")
                    continue  # Lewati jika gambar tidak valid

                # Deteksi wajah
                faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

                for (x, y, w, h) in faces_detected:
                    # Resize wajah yang terdeteksi dan tambahkan ke list
                    face_resized = cv2.resize(img[y:y+h, x:x+w], (150, 150))
                    faces.append(face_resized)
                    labels.append(current_label)

            current_label += 1

    return faces, labels, label_names


# Path dataset
dataset_path = r'E:\PRAKTIKUM\UAS\dataset_webcam'

# Ambil gambar wajah dan label
faces, labels, label_names = get_images_and_labels(dataset_path)

# Validasi dataset
if len(faces) > 0:
    print("Dataset berhasil dimuat.")
    # Mengubah wajah menjadi array datar (1D) untuk pelatihan SVM
    faces_flattened = [face.flatten() for face in faces]

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Train SVM classifier
    clf = SVC(kernel='linear', probability=True)
    clf.fit(faces_flattened, labels_encoded)

    # Simpan model pelatihan
    np.save('svm_model.npy', clf)
    np.save('label_encoder.npy', le)
    print("Model berhasil disimpan.")

else:
    print("Error: Dataset kosong atau tidak valid. Pastikan dataset berisi gambar wajah.")
    exit()

# Load model pelatihan yang telah disimpan
clf = np.load('svm_model.npy', allow_pickle=True).item()
le = np.load('label_encoder.npy', allow_pickle=True).item()

# Inisialisasi deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mulai kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat mengakses webcam. Pastikan webcam terhubung.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

    for (x, y, w, h) in faces_detected:
        # Resize wajah yang terdeteksi untuk prediksi
        face_resized = cv2.resize(gray[y:y+h, x:x+w], (150, 150)).flatten().reshape(1, -1)

        # Prediksi identitas wajah
        label_encoded = clf.predict(face_resized)

        # Mendapatkan probabilitas dari model SVM
        proba = clf.predict_proba(face_resized)
        confidence = np.max(proba)

        # Decode label
        label = le.inverse_transform(label_encoded)[0]
        name = label_names.get(label, "Unknown")

        # Menampilkan hasil prediksi pada gambar
        cv2.putText(frame, f"{name} ({int(confidence * 100)}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Gambarkan kotak sekitar wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Tampilkan frame video
    cv2.imshow('Face Recognition', frame)

    # Keluar jika tekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

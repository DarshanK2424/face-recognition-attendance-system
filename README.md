# 🎯 Face Recognition Attendance System (OpenCV)

A real-time face recognition-based attendance system built using OpenCV and Python.
The system detects faces from a live webcam, recognizes individuals using a trained model, and logs attendance into a CSV file while preventing duplicate entries.

---

## 🚀 Features

* Real-time face detection using Haar Cascade
* Face recognition using LBPH (Local Binary Pattern Histogram)
* Automatic attendance marking with timestamp
* Duplicate prevention (marks each person only once)
* Simple and lightweight (no heavy ML frameworks required)

---

## 🧠 Project Flow

```
Dataset Collection → Model Training → Face Recognition → Attendance Logging
```

### 1️⃣ Face Detection

* Uses Haar Cascade to detect faces in real-time from webcam.

### 2️⃣ Dataset Collection

* Captures multiple face images of a user.
* Stores them in:

```
dataset/
   └── username/
         ├── 0.jpg
         ├── 1.jpg
         └── ...
```

### 3️⃣ Model Training

* Uses LBPH algorithm to learn facial patterns.
* Generates:

  * `face_model.yml` (trained model)
  * `labels.npy` (mapping of IDs to names)

### 4️⃣ Face Recognition

* Detects face → compares with trained data → predicts identity.

### 5️⃣ Attendance System

* Recognized names are stored in:

```
attendance.csv
```

* Format:

```
Name,Time
user1,18:42:10
```

* Prevents duplicate entries using in-memory + file-based checks.

---

## 📁 Project Structure

```
Face_Attendance_System/
│
├── dataset/
│   └── username/
│
├── face_model.yml
├── labels.npy
├── attendance.csv
│
├── collect_faces.py
├── train_model.py
├── attendance.py
│
└── haarcascade_frontalface_alt.xml
```

---

## ⚙️ Prerequisites

Make sure you have the following installed:

```bash
pip install opencv-python opencv-contrib-python numpy
```

---

## 📦 Required Files

### Haar Cascade File

Download or use OpenCV’s built-in path:

* `haarcascade_frontalface_alt.xml`

👉 Option 1 (Recommended):
Use OpenCV default:

```python
cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
```

👉 Option 2:
Download manually from OpenCV GitHub and place in project folder.

---

## ▶️ How to Run

### Step 1: Collect Face Data

```bash
python collect_faces.py
```

---

### Step 2: Train Model

```bash
python train_model.py
```

---

### Step 3: Run Attendance System

```bash
python attendance.py
```

---

## ⚠️ Notes

* Ensure good lighting for better accuracy
* Capture images with different angles and expressions
* Recognition accuracy depends on dataset quality
* Confidence threshold can be tuned (e.g., 60–75)

---

## 🧠 Key Learnings

* Real-time computer vision using OpenCV
* Face detection vs face recognition
* Data preprocessing and model training
* Handling real-world issues like noise and duplicate entries

---

## 📌 Future Improvements

* Add GUI interface
* Store attendance with date
* Integrate database (SQLite/MySQL)
* Improve accuracy using deep learning (FaceNet, Dlib)

---

## 🏁 Conclusion

This project demonstrates a complete pipeline of a real-world computer vision system — from data collection to deployment — and showcases practical implementation of face recognition for attendance automation.

---


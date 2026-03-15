# 🚗 Real-Time Vehicle Detection using YOLOv8 and Streamlit

## 📌 Project Overview

This project implements a **real-time vehicle detection system** using the YOLOv8 object detection model.
The trained model detects vehicles from **images and videos** and displays the detection results with **bounding boxes and confidence scores**.

The system also includes a **Streamlit web application** that allows users to upload images or videos and run vehicle detection directly in a browser.

This project demonstrates practical skills in **Computer Vision, Deep Learning, and Model Deployment**.

---

## 🎯 Problem Statement

Traffic monitoring and vehicle detection are important tasks in **smart transportation systems**.
Manual monitoring is inefficient, so an automated system is required to detect vehicles from images and videos in real time.

This project solves this problem using a **YOLOv8 deep learning model**.

---

## ✨ Features

• Real-time vehicle detection using YOLOv8
• Detect vehicles from images and videos
• Bounding box visualization
• Confidence score display
• Streamlit web interface for easy usage
• Custom dataset training support
• Model inference directly in the browser

---

## 🧠 Tech Stack

**Programming Language**

• Python

**Libraries & Frameworks**

• YOLOv8 (Ultralytics)
• OpenCV
• NumPy
• Streamlit

**Tools**

• VS Code
• GitHub

---

## 📁 Project Structure

```
Real-Time-Vehicle-Detection/
│
├── app.py
├── train_model.py
├── dataset.yaml
├── README.md
│
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   │
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   │
│   └── test/
│       ├── images/
│       └── labels/
│
└── runs/
    └── detect/
        └── train/
            └── weights/
                └── best.pt
```

---

## ⚙️ Installation

Clone the repository or place the project in your workspace.

Example location:

```
C:\Users\user\Desktop\YOLO-v8-final
```

Install required dependencies:

```bash
pip install ultralytics
pip install streamlit
pip install opencv-python
pip install numpy
pip install protobuf --upgrade
```

---

## 📂 Dataset Structure

Make sure your dataset follows this structure:

```
dataset/
  train/images/
  train/labels/
  val/images/
  val/labels/
  test/images/
  test/labels/
```

---

## 📝 Dataset Configuration

Example `dataset.yaml`:

```yaml
path: dataset

train: train/images
val: val/images
test: test/images

nc: 1
names: ['vehicle']
```

If your dataset contains multiple classes, update `nc` and `names` accordingly.

---

## ▶️ Model Training

Run the training script:

```bash
python train_model.py
```

The model will train using **YOLOv8n** and save the best weights to:

```
runs/detect/train/weights/best.pt
```

---

## 🧪 Model Validation

To evaluate the trained model:

```bash
yolo val model=runs/detect/train/weights/best.pt data=dataset.yaml
```

This will calculate detection metrics such as:

• Precision
• Recall
• mAP (Mean Average Precision)

---

## 🎥 Prediction on Video

Run YOLOv8 inference on a video:

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source="video.mp4" conf=0.5
```

Output will be saved to:

```
runs/detect/predict/
```

---

## 🖥️ Streamlit Web Application

Run the Streamlit interface:

```bash
streamlit run app.py
```

After running the command, open the browser:

```
http://localhost:8501
```

### App Features

The Streamlit app allows users to:

• Upload an image or video
• Run YOLOv8 vehicle detection
• Visualize bounding boxes on detected vehicles
• Display detection confidence scores

---

## 📷 Example Output

The model detects vehicles and highlights them with **bounding boxes and confidence scores**.

Example:

```
Vehicle detected
Confidence: 0.92
```

---

## 📊 Future Improvements

Possible improvements for this project:

• Multi-class vehicle detection
• Real-time webcam detection
• Traffic counting system
• Vehicle type classification (car, truck, bus)
• Deployment on cloud platforms

---

## 👨‍💻 Author

**Deepak Kumar Singh**

Data Science Learner | Computer Vision Enthusiast


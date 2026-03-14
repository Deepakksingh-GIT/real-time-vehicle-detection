# Real-time Vehicle Detections (YOLOv8)

This repository implements YOLOv8 training/validation and a Streamlit app for running real-time vehicle detection on images or video files.

## ✅ What’s included

- `train_model.py`: trains YOLOv8n using a dataset from `dataset/`.
- `dataset.yaml`: dataset config with train/val/test paths and classes.
- `app.py`: Streamlit UI for image/video upload + YOLOv8 inference.
- `runs/detect/`: model logs, weights, and predictions.

## 📌 Setup

1. Clone workspace / place repo at `C:\Users\user\Desktop\YOLO-v8 final`.
2. Install dependencies:

```bash
pip install ultralytics streamlit opencv-python numpy
pip install protobuf --upgrade
```

3. Ensure dataset structure is:

```
dataset/
  train/images/
  train/labels/
  val/images/
  val/labels/
  test/images/
  test/labels/
```

4. Prepare `dataset.yaml` (example already provided):

```yaml
path: dataset
train: train/images
val: val/images
test: test/images

nc: 1
names: ['vehicle']
```

## ▶️ Training

```bash
python train_model.py
```

- Trains `yolov8n.pt` and saves best weights at `runs/detect/train6/weights/best.pt`.
- If class mismatch occurs, adjust `nc` and `names` in `dataset.yaml` or relabel dataset.

## 🧪 Validation

```bash
yolo val model=runs/detect/train6/weights/best.pt data=dataset.yaml
```

If your dataset has more than one label class and model is single-class, create a reduced dataset or retrain with `nc` equal to number of label classes.

## 📺 Prediction on video

```bash
yolo detect predict model=runs/detect/train6/weights/best.pt source="C:\\Users\\user\\Downloads\\yolo_dataset\\archive-7\\TestVideo\\TrafficPolice.mp4" conf=0.5
```

- Output is saved in `runs/detect/predict/TrafficPolice.avi`.

## 🖥️ Streamlit App


```bash
streamlit run app.py
```

1. App title: **Real-time vehicle detections**.
2. Upload image/video file or specify local path.
3. Press **Run detection**.
4. Results shown in page; saved to `runs/detect/streamlit/prediction/`.

## 🛠 Notes

- Model path in `app.py` points to `runs/detect/train6/weights/best.pt`.
- If you retrain with different class count, update `dataset.yaml` and `names` accordingly.

## 📁 Directory snapshot

```
README.md
app.py
train_model.py
dataset.yaml
dataset/
  train/
  val/
  test/
runs/detect/
```

## 🚀 Next steps

- Change `nc` & `names` for 5 classes 0-4 if dataset uses 5 labels.
- Add cross-validation, hyperparameter sweeps, and class-specific metrics.
- Use GPU if available: add `device='0'` argument to `yolo` call.

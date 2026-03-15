import os
import tempfile
from pathlib import Path

import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title='Real-time vehicle detections', layout='wide')

MODEL_PATH = r"C:\Users\user\Desktop\YOLO-v8 final\runs\detect\train11\weights\best.pt"

@st.cache_resource
def load_model(path=MODEL_PATH):
    return YOLO(path)

model = load_model()

# Ensure class names accessible as dict
model_names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}

st.title('Real-time multi-class vehicle detections')

st.write('Upload an image or video file and press Run. The app runs YOLOv8 detection on the file.')

uploaded_file = st.file_uploader('Upload image/video', type=['jpg','jpeg','png','bmp','mp4','mov','avi'])

# Optional local file path input
source_path = st.text_input('Or enter local image/video path', '')

run = st.button('Run detection')

if run:
    if not uploaded_file and not source_path:
        st.warning('Please upload a file or enter a local path.')
    else:
        with st.spinner('Running YOLO detection...'):
            if uploaded_file:
                suffix = Path(uploaded_file.name).suffix.lower()
                tmp_dir = Path(tempfile.mkdtemp())
                input_path = tmp_dir / uploaded_file.name
                with open(input_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
            else:
                input_path = Path(source_path)
                if not input_path.exists():
                    st.error(f'Path not found: {input_path}')
                    st.stop()

            save_dir = Path('runs/detect/streamlit')
            save_dir.mkdir(parents=True, exist_ok=True)

            # Use smaller image size and streaming inference to reduce memory usage
            inference_params = {
                'source': str(input_path),
                'save': True,
                'project': 'runs/detect/streamlit',
                'name': 'prediction',
                'exist_ok': True,
                'conf': 0.4,
                'imgsz': 416,
                'device': 'cpu',
                'stream': True,
                'vid_stride': 2,
            }

            try:
                result = model.predict(**inference_params)
            except Exception as e:
                if 'OutOfMemoryError' in str(e) or 'cv::OutOfMemoryError' in str(e):
                    st.warning('Memory error on video processing; trying lower resolution and stride.')
                    inference_params['imgsz'] = 320
                    inference_params['vid_stride'] = 4
                    result = model.predict(**inference_params)
                else:
                    st.error(f'Error running inference: {e}')
                    raise

            output_file = None
            if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                # copy or load predicted image
                out_img = save_dir / 'prediction' / input_path.name
                if out_img.exists():
                    st.image(str(out_img), caption='Predicted image', use_column_width=True)
                else:
                    st.error('No output image found.')
            else:
                # video output is a .mp4 of same basename
                out_vid = save_dir / 'prediction' / f'{input_path.stem}.mp4'
                if out_vid.exists():
                    st.video(str(out_vid))
                    st.success(f'Prediction saved to {out_vid}')
                else:
                    st.error('No output video found.')

            st.write('YOLO result stats:')
            detections = []

            # If `model.predict(..., stream=True)` returns a generator, iterate through it.
            if hasattr(result, '__iter__') and not isinstance(result, list):
                results = list(result)
            else:
                results = result

            if results:
                for res in results:
                    if res.boxes is None:
                        continue
                    for box in res.boxes:
                        cls_id = int(box.cls)
                        conf = float(box.conf)
                        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                        detections.append({
                            'class_id': cls_id,
                            'class_name': model_names.get(cls_id, str(cls_id)),
                            'confidence': round(conf, 4),
                            'bbox': [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                        })

            st.json(detections)
            if detections:
                st.write('Summary:')
                counts = {}
                for d in detections:
                    counts[d['class_name']] = counts.get(d['class_name'], 0) + 1
                st.write(counts)



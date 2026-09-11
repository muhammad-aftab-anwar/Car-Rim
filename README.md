# 🚗 Rim & Wheel Detection using YOLO Pose

A computer vision project for detecting **wheels and vehicle rims** using a custom-trained **YOLO Pose model** with **4 keypoints per object**.

The project supports image and video inference and includes a **Streamlit web application** for testing the trained model, displaying bounding boxes, keypoints, confidence scores, object counts, and real-time processing FPS.

---

## 📌 Project Overview

This project uses **YOLO Pose estimation** to identify two classes:

| Class ID | Class    |
| -------- | -------- |
| `0`      | Wheel    |
| `1`      | Just Rim |

Each detected object contains:

* Bounding box
* Confidence score
* 4 keypoints
* Keypoint coordinates
* Object class

The trained model can be used for automated wheel/rim analysis from images and videos.

---

## ✨ Features

* 🚗 Wheel detection
* 🛞 Rim detection
* 📍 4-keypoint pose estimation
* 🎯 Bounding box detection
* 📊 Confidence score
* 🎥 Video detection
* 🖼️ Image detection
* 🚀 FPS monitoring
* 🔢 Object counting
* 🎨 Keypoint visualization
* ⚙️ Adjustable confidence threshold
* 📐 Adjustable image size
* 🌐 Streamlit web interface
* ⬇️ Download processed videos
* 🔥 GPU support when available

---

## 🧠 Model

The project uses a **YOLO Pose architecture**.

Dataset configuration:

```yaml
kpt_shape: [4, 3]

names:
  0: wheel
  1: just rim
```

### Keypoint configuration

```text
4 Keypoints
     │
     ├── X coordinate
     ├── Y coordinate
     └── Visibility
```

Therefore, every object contains:

```text
4 × 3 = 12 keypoint values
```

along with its class and bounding-box information.

---

## 📂 Dataset Structure

```text
rim_yolo/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
└── data.yaml
```

---

## 📝 Dataset Configuration

The `data.yaml` file:

```yaml
path: /kaggle/working/rim_yolo

train: images/train
val: images/val
test: images/test

kpt_shape: [4, 3]

names:
  0: wheel
  1: just rim
```

---

## 🔄 Data Preprocessing

Before training, the dataset was checked for:

* Image/label matching
* Corrupt files
* Correct YOLO label format
* Correct train/validation/test organization
* Four keypoints per object

YOLO handles image resizing during training using the selected `imgsz`.

---

## 🧪 Data Augmentation

To improve generalization, augmentation was applied during YOLO training.

Used augmentation includes:

```text
Rotation
Translation
Scaling
Shearing
Perspective transformation
Horizontal flipping
Hue adjustment
Saturation adjustment
Brightness adjustment
```

Example configuration:

```python
degrees=10
translate=0.10
scale=0.50
shear=2.0
perspective=0.0005

fliplr=0.5
flipud=0.0

hsv_h=0.015
hsv_s=0.5
hsv_v=0.3
```

Validation and test images are kept free from training augmentation to provide a more realistic evaluation.

---

# 🏋️ Training

Install Ultralytics:

```bash
pip install -U ultralytics
```

Load the pose model:

```python
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
```

Train:

```python
results = model.train(
    data="/kaggle/working/rim_yolo/data.yaml",

    epochs=100,
    imgsz=640,
    batch=16,
    device=0,

    # Augmentation
    degrees=10,
    translate=0.10,
    scale=0.50,
    shear=2.0,
    perspective=0.0005,

    fliplr=0.5,
    flipud=0.0,

    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,

    # Training control
    patience=20,
    workers=2,

    project="/kaggle/working/rim_training",
    name="rim_pose",

    save=True,
    plots=True
)
```

---

# 📊 Model Evaluation

After training:

```python
from ultralytics import YOLO

model = YOLO(
    "/kaggle/working/rim_training/rim_pose/weights/best.pt"
)

metrics = model.val(
    data="/kaggle/working/rim_yolo/data.yaml",
    imgsz=640,
    device=0
)
```

Important evaluation metrics include:

* Precision
* Recall
* mAP@50
* mAP@50-95
* Pose/keypoint metrics

---

# 🔍 Image Prediction

```python
results = model.predict(
    source="test.jpg",
    conf=0.25,
    imgsz=640,
    save=True
)
```

The output contains:

* Bounding boxes
* Class labels
* Confidence
* Keypoints

---

# 🎥 Video Prediction

The model can also process videos:

```python
results = model.predict(
    source="input.mp4",
    conf=0.25,
    imgsz=640,
    save=True
)
```

The Streamlit application additionally calculates the actual processing FPS.

---

# 🌐 Streamlit Application

The project includes a Streamlit interface for testing the trained model.

### Application features

```text
             Rim & Wheel AI
                    │
       ┌────────────┴────────────┐
       │                         │
     Image                     Video
       │                         │
       ▼                         ▼
 YOLO Pose                  YOLO Pose
       │                         │
       ▼                         ▼
 Bounding Box              Bounding Box
 Keypoints                 Keypoints
 Confidence                Object Count
       │                         │
       │                         ▼
       │                       FPS
       │                         │
       └────────────┬────────────┘
                    ▼
             Detection Result
```

---

# 📁 Application Structure

```text
rim_app/
│
├── app.py
├── best.pt
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/rim-wheel-yolo-pose.git
```

Move into the project:

```bash
cd rim-wheel-yolo-pose
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

Example `requirements.txt`:

```text
ultralytics
streamlit
opencv-python
Pillow
numpy
```

---

# ▶️ Run the Application

Make sure `best.pt` is located beside `app.py`:

```text
rim_app/
├── app.py
└── best.pt
```

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🖼️ Image Detection

The image mode allows users to:

1. Upload an image
2. Select confidence threshold
3. Select image size
4. Run YOLO inference
5. View bounding boxes
6. View keypoints
7. View confidence scores
8. View processing time
9. View estimated FPS

---

# 🎥 Video Detection

The video mode allows users to:

1. Upload a video
2. Run YOLO inference frame-by-frame
3. Detect wheels and rims
4. Display keypoints
5. Display object count
6. Display processing FPS
7. Display progress
8. Download the processed video

---

# 🚀 FPS Monitoring

The application displays:

```text
Original FPS
```

The FPS of the uploaded video.

```text
Processing FPS
```

The speed at which the computer processes the video.

```text
Average FPS
```

The average processing speed across the entire video.

For example:

```text
Original FPS:    30 FPS
Processing FPS:  24 FPS
Average FPS:     23.7 FPS
```

This makes it easy to evaluate real-time inference performance.

---

# 🛠️ Technologies Used

| Technology  | Purpose                     |
| ----------- | --------------------------- |
| Python      | Programming                 |
| YOLO Pose   | Object & keypoint detection |
| Ultralytics | Model training/inference    |
| PyTorch     | Deep learning framework     |
| OpenCV      | Image/video processing      |
| NumPy       | Numerical operations        |
| Pillow      | Image handling              |
| Streamlit   | Web application             |
| Kaggle      | Model training environment  |

---

# 📈 Project Pipeline

```text
Raw Dataset
     │
     ▼
Dataset Organization
     │
     ▼
Label Verification
     │
     ▼
Train / Validation / Test Split
     │
     ▼
Data Augmentation
     │
     ▼
YOLO Pose Training
     │
     ▼
Model Evaluation
     │
     ▼
best.pt
     │
     ▼
Streamlit Application
     │
     ├── Image
     │
     └── Video
           │
           ▼
     Detection + Keypoints
           │
           ▼
        FPS + Results
```

---

# 🎯 Classes

### 1. Wheel

Class ID:

```text
0
```

Used to detect complete wheels.

### 2. Just Rim

Class ID:

```text
1
```

Used to identify the rim separately.

---

# 🔮 Future Improvements

Possible future improvements include:

* [ ] Webcam real-time detection
* [ ] Automatic rim damage detection
* [ ] Rim size estimation
* [ ] Wheel angle estimation
* [ ] Vehicle make/model detection
* [ ] Better keypoint localization
* [ ] Real-time GPU optimization
* [ ] TensorRT deployment
* [ ] ONNX export
* [ ] Cloud deployment
* [ ] Mobile application
* [ ] Automatic wheel/rim measurement

---

# ⚠️ Troubleshooting

### `operator torchvision::nms does not exist`

This usually indicates an incompatible PyTorch/TorchVision installation.

Check:

```bash
python -c "import torch; print(torch.__version__)"
```

and:

```bash
python -c "import torchvision; print(torchvision.__version__)"
```

Install compatible PyTorch and TorchVision versions in the same environment.

---

### `best.pt not found`

Make sure:

```text
best.pt
```

is in the same directory as:

```text
app.py
```

Example:

```text
rim_app/
├── app.py
├── best.pt
└── requirements.txt
```

---

# 📜 License

This project is intended for educational, research, and computer vision development purposes.

Add your preferred license before publishing the repository.

---

# 👨‍💻 Author

**Muhammad Aftab Anwar**

AI / ML Developer | Computer Vision | Deep Learning

---

## ⭐ Acknowledgements

* Ultralytics YOLO
* PyTorch
* OpenCV
* Streamlit
* Kaggle

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ star.

Feel free to fork the project and improve the model or application.

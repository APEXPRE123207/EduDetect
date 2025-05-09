# # EduDetect: Real-Time Shape and Color Recognition for Early Education

**EduDetect** is an interactive computer vision project aimed at helping young children (ages 2–5) learn basic shapes and colors through engaging real-time detection and voice feedback. It combines object detection, color identification, shape recognition, and speech interaction to create an educational experience that is both fun and intuitive.

![EduDetect Demo](assets/demo_screenshot.jpg)

---

## 🎯 Features

- 🧠 **YOLOv3 Object Detection** – Detects objects from a webcam stream (excluding people).
- 🎨 **Color Recognition** – Determines the dominant color of the detected object using region-based RGB sampling.
- 🔷 **Shape Classification** – Classifies the shape using a custom-trained PyTorch CNN.
- 🗣️ **Voice Interaction** – Speaks out the result and prompts the child to repeat it using speech recognition.
- 👶 **Child-Friendly** – Designed for early learners with visual and auditory engagement.

---

## 📚 Dataset

This project uses a custom dataset made by combining:

- [Four Basic Shapes Dataset (Kaggle)](https://www.kaggle.com/datasets/smeschke/four-shapes)
- [Handdrawn Shapes (HDS) Dataset (Kaggle)](https://www.kaggle.com/datasets/frobert/handdrawn-shapes-hds-dataset)

The combined dataset was used to train a CNN for shape classification.

---

## 🗂️ Project Structure

├── Assets/ # Saved webcam images and trained models
│ └── Pics/ # Auto-saved object images
├── Required/ # CSVs and PyTorch model
│ ├── colors4.csv # Color name and RGB data
│ ├── shapes_model_v1.pth # Trained PyTorch shape model
├── Resources/ # YOLO configs, weights, and cvlib
├── color_identification.py # Detects the dominant object color
├── shape_detection_v2.py # Masks the object based on its color
├── test_model.py # Loads and predicts shape using CNN
├── voice.py # Handles speech output and recognition
├── main.py # Real-time detection and interaction loop
├── conda_packages.txt # Conda environment file
├── pip_packages.txt # pip requirements file
├── Requirements.txt # Minimal install file


## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/APEXPRE123207/edudetect.git
cd edudetect 
```

2. Create the environment using Conda and pip. All the requires packages files are in the Required folder.
```
conda create -n edudetect --file conda_packages.txt
conda activate edudetect
```

Then install pip-specific dependencies:
```
pip install -r pip_packages.txt

pip install -r Requirements.txt
```
3. Download YOLOv3 Weights
Place the following files in Resources/.cvlib/object_detection/yolo/yolov3/:

yolov3.cfg

yolov3.weights

coco.names

You can get them from: https://pjreddie.com/darknet/yolo/

🚀 Run the Application
```
python main.py
```

This will:

Open your webcam

Detect objects (excluding people)

Identify color and shape

Provide spoken feedback

Prompt the user to repeat the result aloud

🧠 Technologies Used
Python 3.9

OpenCV (with CUDA support)

PyTorch

YOLOv3 (via cvlib)

pyttsx3, SpeechRecognition, sounddevice

pandas, numpy, matplotlib

👩‍💻 Author
Developed as an educational aid for early learners using modern AI and computer vision techniques.

📜 License
MIT License

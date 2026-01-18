# Crime Detection using CNN–LSTM

This repository implements a **Crime Detection System** using a **CNN–LSTM deep learning architecture** to identify suspicious or criminal activities from video sequences. The model combines spatial feature extraction from Convolutional Neural Networks (CNN) with temporal pattern learning using Long Short-Term Memory (LSTM) networks.

---

## 📌 Project Overview

Traditional crime detection systems rely heavily on manual monitoring of CCTV footage, which is inefficient and error-prone. This project aims to automate crime detection by analyzing video frames and learning temporal motion patterns to classify activities as **normal** or **suspicious/criminal**.

---

## 🧠 Model Architecture

- **CNN (Convolutional Neural Network)**  
  Extracts spatial features from individual video frames.

- **LSTM (Long Short-Term Memory Network)**  
  Captures temporal dependencies across sequences of frames.

**Pipeline:**
Video → Frame Extraction → CNN Feature Extraction → LSTM Sequence Learning → Classification

yaml


---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Deep Learning Framework:** TensorFlow / Keras  
- **Computer Vision:** OpenCV  
- **Libraries:** NumPy, Matplotlib, Scikit-learn  
- **Model Type:** CNN + LSTM  
- **Input:** Video sequences / frame batches  
- **Output:** Crime / No-Crime classification

---

## 📂 Project Structure

crime-detection-cnn-lstm/
│
├── dataset/ # Crime and non-crime video data
├── frames/ # Extracted video frames
├── models/ # Saved trained models
├── notebooks/ # Jupyter notebooks (experiments)
├── src/
│ ├── preprocess.py # Frame extraction & preprocessing
│ ├── model.py # CNN-LSTM model definition
│ ├── train.py # Model training script
│ └── predict.py # Inference on new videos
│
├── requirements.txt
├── README.md
└── .gitignore

yaml


---

## ⚙️ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/crime-detection-cnn-lstm.git
cd crime-detection-cnn-lstm
Create a virtual environment (optional but recommended)

bash

python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
Install dependencies

bash

pip install -r requirements.txt
▶️ Usage
Train the model
bash

python src/train.py
Predict on a video
bash

python src/predict.py --video path/to/video.mp4
📊 Dataset
Public action recognition / crime datasets

Custom-labeled CCTV footage (optional)

⚠️ Ensure proper labeling and ethical use of surveillance data.

📈 Results
Learns both spatial and temporal patterns

Effective for detecting abnormal activities in video streams

Can be extended to real-time CCTV monitoring systems

🚀 Future Improvements
Real-time inference with live CCTV feeds

Multi-class crime classification (fighting, theft, assault, etc.)

Integration with alert systems (email / SMS)

Optimization for low-memory devices

⚠️ Disclaimer
This project is for educational and research purposes only. It should not be used as a sole decision-making system for law enforcement or surveillance without proper validation and ethical considerations.

👨‍💻 Author
Prajwal Temak
Computer Science Student | AI & Computer Vision Enthusiast

⭐ Acknowledgements
Research papers on CNN–LSTM video classification

Open-source crime detection datasets

TensorFlow & OpenCV communities

⭐ If you find this project useful, consider giving it a star!

yaml


---

If you want, I can:
- Simplify it for a **college mini-project**
- Make it **resume-optimized**
- Add **sample results & screenshots**
- Align it with your **Traffic Management / Helmet Detection work**

Just tell me 👍

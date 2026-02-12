# 🧠 SentioVision
### Real-Time Facial Emotion Recognition using Deep Learning
<br><br>

## 🔍 About the Project
**SentioVision** is a real-time **facial emotion recognition system** that identifies human emotions from facial expressions using deep learning.  
The project showcases both **foundational CNN design** and **modern transfer learning techniques** for improved accuracy and robustness.

### 🔧 Core Approaches
* 🧠 **Custom CNN** built from scratch to understand fundamentals  
* ⚡ **EfficientNet (Transfer Learning)** for better generalization and performance
<br><br>

### 😃 Emotion Classes
> 😠 Angry • 😢 Sad • 😄 Happy • 😨 Fear • 😐 Neutral • 🤢 Disgust • 😲 Surprise
<br><br>

This project bridges **theory and practice**, making it suitable for **learning, experimentation, and real-world demos**.
<br><br>

## 🎯 Potential Applications
* 🧘 Mental health & emotion monitoring  
* 🛍️ Customer feedback and sentiment analysis  
* 🤖 Human–Computer Interaction (HCI) systems
<br><br>

## 📘 Project Overview
| Notebook | Description |
|--------|-------------|
| `FER_Code.ipynb` | CNN implemented from scratch (baseline model) |
| `FER_Code_EfficientNet.ipynb` | EfficientNet-based transfer learning model |
| `webcam.ipynb` | Real-time emotion detection using webcam |


> **Note:**  
> Webcam detection requires a trained model.  
> * Use the **pre-trained model** at `models/emotion_model.keras`, **or**  
> * Train your own model using one of the training notebooks.
<br><br>

## 🧩 Dataset Setup (FER2013)
1. Download the **FER2013 dataset** from Kaggle  
2. Create a folder named `original_images` in the project root  
3. Inside it, create `train/` and `test/` directories  
4. Place images accordingly  
<br>


```
original_images/
├── train/
└── test/
```

<br><br>

## ▶️ Usage Guide

### ✅ Option 1: Use Pre-trained Model
1. Ensure `models/emotion_model.keras` exists  
2. Run `webcam.ipynb`  
3. Start real-time emotion recognition  

### 🛠️ Option 2: Train Your Own Model
1. Run:  
   * `FER_Code.ipynb` **(CNN)**  
   * **or** `FER_Code_EfficientNet.ipynb` **(EfficientNet)**  
2. Best model is automatically saved using callbacks  
3. Model is stored as `models/emotion_model.keras`  
4. Run `webcam.ipynb` for live detection  
<br>

> ⚠️ **Warning:**  
> Training multiple notebooks without renaming the model file will overwrite the previous model.
<br><br>

## 📂 Project Structure
```
emotion_recognition_project/
├── code/
│   ├── FER_Code.ipynb
│   ├── FER_Code_EfficientNet.ipynb
│   └── webcam.ipynb
├── models/
│   └── emotion_model.keras
├── original_images/
│   ├── train/
│   └── test/
├── assets/
│   ├── banner.png
│   └── demo.gif
└── README.md
```


<br><br>

## 📸 Demo
<p align="center">
  <img src="assets/demo.gif" alt="Real-time Emotion Detection Demo" width="600"/>
</p>
<br><br>

<h2>📦 Installation &amp; Requirements</h2>

<pre><code>pip install tensorflow keras opencv-python matplotlib seaborn pillow numpy pandas tqdm scikit-learn facenet-pytorch
</code></pre>

<br><br>

<h2>🧠 Notes for Beginners</h2>

<ul>
  <li><b>CNN layers (Conv2D, MaxPooling2D)</b> → used in scratch model</li>
  <li><b>EfficientNet</b> → required only for transfer learning</li>
  <li><b>OpenCV + load_model</b> → used in webcam inference</li>
  <li><b>MTCNN</b> → optional face detection before emotion prediction</li>
</ul>

<br><br>

<h2>📄 License</h2>

<p>
This project is open for <b>learning, research, and experimentation</b>.<br>
Feel free to modify and adapt it.
</p>

<br><br>

<h2>🤝 Contributing</h2>

<p>Contributions are welcome 🚀</p>

<ul>
  <li>Report bugs or request features via Issues</li>
  <li>Submit Pull Requests for improvements</li>
  <li>Share ideas to improve accuracy or performance</li>
</ul>

<br><br>

<p align="center">
  <b>⭐ If you find this project helpful, consider starring the repository!</b>
</p>

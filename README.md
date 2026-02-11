
# **SentioVision**

**Real-time Facial Emotion Recognition using CNNs and EfficientNet-based Transfer Learning**

**SentioVision** is a real-time facial emotion recognition system designed to accurately identify human emotions from facial expressions. The project combines **custom Convolutional Neural Networks (CNNs)** with **EfficientNet-based transfer learning**, achieving high accuracy and strong generalization.

The system classifies facial images into **seven emotion categories**:
**Angry, Sad, Happy, Fear, Neutral, Disgust, Surprise**.

By integrating **from-scratch CNN experimentation** with **state-of-the-art EfficientNet models**, SentioVision demonstrates mastery of both fundamental deep learning concepts and modern optimization techniques.

**Potential Applications:**

* Mental health monitoring
* Customer feedback analysis
* Human–computer interaction

---

## **Project Overview**

| Notebook                      | Description                                                            |
| ----------------------------- | ---------------------------------------------------------------------- |
| `FER_Code.ipynb`              | CNN implemented from scratch. Achieves lower accuracy.                 |
| `FER_Code_EfficientNet.ipynb` | Builds on EfficientNet, giving better accuracy.                        |
| `webcam.ipynb`                | Uses the trained model to detect emotions in real-time using a webcam. |

> **Note:** The webcam code requires a trained model. You can either:
>
> * Use the **pre-trained model** in `models/emotion_model.keras`
> * Or **train your own** by running either `FER_Code.ipynb` or `FER_Code_EfficientNet.ipynb`.

---

## **Dataset Instructions**

1. Download the **FER2013 dataset** from Kaggle: [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
2. Create a folder named `original_images` in the project root.
3. Inside `original_images`, create `train/` and `test/` subfolders.
4. Place training images in `original_images/train/` and test images in `original_images/test/`.

**Folder structure:**

```
original_images/
├─ train/
└─ test/
```

---

## **Usage Instructions**

### **Option 1 – Use Pre-trained Model**

1. Ensure `models/emotion_model.keras` is present.
2. Run `webcam.ipynb` to start **real-time emotion detection**.

### **Option 2 – Train Your Own Model**

1. Run `FER_Code.ipynb` (CNN from scratch) or `FER_Code_EfficientNet.ipynb` (EfficientNet).
2. Training uses **callbacks** to save the best model and optimize training.
3. The best model is saved as `models/emotion_model.keras`.
4. Run `webcam.ipynb` to start **real-time detection**.

> ⚠️ **Important:** If you train multiple notebooks without renaming/moving the saved model, the previous model will be **overwritten**.

---

## **Project Folder Structure**

```
emotion_recognition_project/
├─ code/
│  ├─ FER_Code.ipynb
│  ├─ FER_Code_EfficientNet.ipynb
│  └─ webcam.ipynb
├─ models/
│  └─ emotion_model.keras
├─ original_images/
│  ├─ train/
│  └─ test/
└─ README.md
```

---

## **📦 Required Libraries & Installation**

### **Install Packages**

Run these commands in your terminal:

```bash
# TensorFlow and Keras for model training
pip install tensorflow keras

# OpenCV for webcam input
pip install opencv-python

# Image processing, plotting, and data handling
pip install matplotlib seaborn pillow numpy pandas tqdm

# Metrics and evaluation
pip install scikit-learn

# Optional: Face detection for webcam (MTCNN)
pip install facenet-pytorch
```

> ✅ **Tip:** Install everything at once:

```bash
pip install tensorflow keras opencv-python matplotlib seaborn pillow numpy pandas tqdm scikit-learn facenet-pytorch
```

---

## **Step 1: Import Libraries**

### **For CNN / EfficientNet Training**

```python
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,
    Rescaling, RandomFlip, RandomRotation, RandomZoom
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import image_dataset_from_directory

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from sklearn.metrics import confusion_matrix, classification_report
```

### **For Webcam Emotion Detection**

```python
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model

import cv2  # Webcam
```

---

## **Step 2: Notes for Beginners**

1. **EfficientNet imports** → only needed if using **transfer learning**.
2. **Conv2D, MaxPooling2D, etc.** → for training CNN from scratch.
3. **OpenCV and load_model** → only required for **webcam detection**.
4. **MTCNN** → optional, only if you want **face detection** before emotion prediction.



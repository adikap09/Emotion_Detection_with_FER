# 🎭 Facial Emotion Recognition (FER) using Deep Learning

## Project Overview

This repository contains the complete implementation and analysis for a deep learning project focused on **Facial Emotion Recognition (FER)**. The goal is to classify static facial images into seven core emotional categories using a Convolutional Neural Network (CNN) built with Keras and TensorFlow.

This project was submitted as a Midsemester Report for Semester 1, 2025-26.

|  |  |
|---|---|
| Submitted By: | Adit Kapur, Pranav P E, Vaishnav Kiran (Group 12) |
| Architecture: | Custom Convolutional Neural Network (CNN) |
| Dataset: | FER2013 (35,887 samples, $48 \times 48$ grayscale) |
| Final Validation Accuracy: | 62.4% |
| Key Findings: | Successfully validated CNN viability for the task, identified significant overfitting, and noted confusion between nuanced classes like 'Fear' and 'Sad'. |

## 🚀 Getting Started

Follow these steps to set up the project environment and replicate the model training.

### 1. Prerequisites

You must have Python 3.x and Git installed on your system.

### 2. Clone the Repository

Clone this repository using the following commands:
```
git clone https://github.com/adikap09/Emotion_Detection_with_FER
cd Emotion_Detection_with_FER/
```
### 3. Install Dependencies

Install all required libraries, including TensorFlow, Keras, and scikit-learn. 

```pip install -r requirements.txt```

### 4. Data Acquisition

This project uses the **FER2013 dataset**, which is often distributed as a single CSV file.

1. **Download:** Obtain the ```fer2013.csv``` file from the original Kaggle competition source or a similar archive.

2. **Placement:** Place the ```fer2013.csv``` file in your project's root directory or modify the data loading path in the ```trainmodel.ipynb``` notebook.

- *(Note: The repository uses a directory structure (```images/train```, ```images/test```). If your notebook is designed to create this structure from the CSV, skip manual image file placement.)*

## 🧠 Model Training and Evaluation

The entire workflow, from data loading to saving the final model, is executed via the Jupyter Notebook.

### Training Steps

1. **Start Jupyter:** \
```jupyter notebook```
 
2. **Open ```trainmodel.ipynb```:** Run the notebook cells sequentially.

3. **Data Preprocessing:** The notebook handles the multi-step preprocessing detailed in the report (Pixel String Conversion, Reshaping to (48, 48, 1), Normalization, and One-Hot Encoding).

4. **Model Definition:** The custom CNN architecture is defined using the Keras Sequential API.

5. **Execution:** The ```model.fit()``` command trains the network for 100 epochs using the Adam optimizer and Categorical Cross-Entropy loss.

### Model Architecture Summary

The custom CNN consists of a sequential stack of layers:

- **Feature Extraction:** Multiple blocks of ```Conv2D``` (e.g., 128, 256, 512 filters) + ```MaxPooling2D``` + ```Dropout``` (0.4 rate).

- **Classification:** ```Flatten``` $\to$ ```Dense``` (512 units) $\to$ ```Dense``` (256 units) $\to$ ```Dense``` (7 units, Softmax output).

## 📊 Results and Overfitting Analysis

The model achieved a peak performance on the test set, but exhibited classic signs of overfitting after $\sim$40 epochs.

- **Final Training Accuracy:** 72.86%

- **Final Validation Accuracy:** 62.4% (at the plateau point)

- **Overfitting:** The divergence between the Training Accuracy and Validation Accuracy plots confirms that the model memorized the training data's noise.

### Confusion Matrix Insights (Normalized)

The analysis shows the model's performance on the validation set:

- **Best Performance:** **Happy (86% Correct)**, likely due to high sample representation in the dataset.

- **Worst Performance:** **Fear (41% Correct)** and high confusion between 'Sad' / 'Neutral' and 'Angry' / 'Disgust' due to subtle facial feature overlap.

## 💡 Future Work

To improve performance and achieve project goals, the following steps are planned:

- **Implement Real-Time Detection:** Integrate the saved model (```emotiondetector.h5```) with **OpenCV** to provide live emotion classification via a webcam feed.

- **Experiment with Transfer Learning:** Utilize pre-trained weights from more powerful CNN architectures (e.g., VGG-Face or a larger ResNet model) and fine-tune them on the FER2013 dataset.

- **Data Augmentation:** Implement robust techniques to artificially increase the training data size and variety to combat overfitting.

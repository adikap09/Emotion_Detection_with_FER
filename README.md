# 🎭 Facial Emotion Recognition (FER) using Deep Learning

## Project Overview

This repository contains the complete implementation and analysis for a deep learning project focused on **Facial Emotion Recognition (FER)**. The goal is to classify static facial images into seven core emotional categories using a Convolutional Neural Network (CNN) built with Keras and TensorFlow.

|  |  |
|---|---|
| Submitted By: | Adit Kapur, Pranav P E, Vaishnav Kiran (Group 12) |
| Architecture: | Custom Convolutional Neural Network (CNN) |
| Datasets: | FER2013 (48x48) & Balanced RAF-DB (100x100) |
| Final Validation Accuracy: | 62.4% (Baseline) / ~93% (RAF-DB Transfer Learning) |
| Key Findings: | Successfully validated CNN viability for the task. Significant performance boost achieved using RAF-DB transfer learning to combat overfitting and improve generalization on real-world images. |

## 🚀 Getting Started

Follow these steps to set up the project environment and replicate the model training.

### 1. Prerequisites

You must have Python 3.x and Git installed on your system.

### 2. Clone the Repository

git clone [https://github.com/adikap09/Emotion_Detection_with_FER](https://github.com/adikap09/Emotion_Detection_with_FER) \
cd Emotion_Detection_with_FER/

### 3. Install Dependencies

Install all required libraries, including TensorFlow, Keras, OpenCV, and scikit-learn. **It is highly recommended to use a virtual environment.**

Example: Activate a virtual environment before running this command
### 4. Data Acquisition

This project supports two datasets. You can choose to train on either or both.

#### Option A: FER2013 (Standard Benchmark)

1. **Download:** Obtain the fer2013.csv file from the original Kaggle competition source.

2. **Placement:** Place it in the root directory. The Emotion_detector.ipynb notebook handles the processing.

#### Option B: Balanced RAF-DB (Recommended for Transfer Learning)

1. **Download:** [Balanced RAF-DB Dataset (Kaggle)](https://www.kaggle.com/datasets/dollyprajapati182/balanced-raf-db-dataset-7575-grayscale)

2. **Placement:** Extract the dataset into a folder named RAF_DB_Dataset in your project root so that you have RAF_DB_Dataset/train and RAF_DB_Dataset/test.

## 🧠 Model Training and Evaluation

The entire workflow is executed via Jupyter Notebooks.

### 1. Baseline Model (FER2013)

- **Notebook:** Emotion_detector.ipynb

- **Description:** Trains a custom CNN from scratch on the FER2013 dataset.

- **Process:** Handles pixel string conversion, reshaping, and standard training.

### 2. Advanced Model (RAF-DB Transfer Learning)

- **Notebook:** RAF_DB_trainer.ipynb

- **Description:** Trains a deeper CNN on the higher-quality **RAF-DB** dataset ($100 \times 100$ images).

- **Usage:**

1. Open RAF_DB_trainer.ipynb.

2. Run the cells to preprocess the RAF-DB images.

3. Train the model to generate raf_db_pretrained_model.h5.

4. (Optional) Use this pre-trained model to fine-tune on FER2013 for maximum accuracy.

## 🎥 Real-Time Emotion Detection

This project includes a real-time detector that uses your webcam to recognize emotions live.

- **File:** realtime.ipynb

- **Model Used:** Uses the pre-trained raf_db_pretrained_model.h5 or emotiondetector.h5 (ensure these files exist after running the trainer).

- **Dependencies:** Requires opencv-python (pip install opencv-python).

### How to Run

1. Open realtime.ipynb in Jupyter.

2. Ensure your webcam is connected.

3. Run the code cells. A window named "Output" will appear showing the video feed with bounding boxes and emotion labels.

4. Press **'Esc'** to stop the video feed and close the window.

## 📊 Results and Overfitting Analysis

The model achieved a peak performance on the test set, but exhibited classic signs of overfitting after $\sim$40 epochs.

- **Final Training Accuracy:** 72.86%

- **Final Validation Accuracy:** 62.4% (at the plateau point)

- **Overfitting:** The divergence between the Training Accuracy and Validation Accuracy plots confirms that the model memorized the training data's noise.

### Confusion Matrix Insights (Normalized)

The analysis shows the model's performance on the validation set:

- **Best Performance:** **Happy (86% Correct)**, likely due to high sample representation in the dataset.

- **Worst Performance:** **Fear (41% Correct)** and high confusion between 'Sad' / 'Neutral' and 'Angry' / 'Disgust' due to subtle facial feature overlap.

## 📈 RAF-DB Model Results

The advanced model trained on the Balanced RAF-DB dataset demonstrated significantly improved performance and robustness compared to the baseline FER2013 model.

- **Final Test Accuracy:** **~93%**

- **Generalization:** Due to the higher quality and cleaner labels of RAF-DB, the model shows much stronger generalization to real-world images.

### Classification Report Highlights

| Emotion | Precision | Recall | F1-Score |
|---|---|---|---|
| Angry | 0.97 | 0.99 | 0.98 |
| Fear | 0.98 | 1.00 | 0.99 |
| Happy | 0.96 | 0.88 | 0.92 |
| Neutral | 0.84 | 0.85 | 0.85 |

The RAF-DB model effectively solved the confusion issues present in the baseline model, achieving near-perfect classification for 'Fear' and 'Angry' classes.

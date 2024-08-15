import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
TF_ENABLE_ONEDNN_OPTS=0

def load_images(data_dir, img_size):
    images = []
    labels = []
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        label = 1 if category == 'Malignant' else 0
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    images = images / 255.0  # Normalize to [0, 1]
    return images, labels

def preprocess_main(data_dir, img_size):
    # Load and preprocess data
    images, labels = load_images(data_dir, img_size)
    # load_images(data_dir, img_size)
    images, labels = preprocess_data(images, labels)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    # Filter for .npy files
    npy_files = [f for f in data_dir if f.endswith('.npy')]

    # Preprocessed data files found
    if npy_files:
        print("Found .npy files:")
    # Save preprocessed data if necessary
    else:
        print("No .npy files found in the directory.")
        np.save(data_dir+'/X_train.npy', X_train)
        np.save(data_dir+'/X_val.npy', X_val)
        np.save(data_dir+'/X_test', X_test)
        np.save(data_dir+'/y_train.npy', y_train)
        np.save(data_dir+'/y_val.npy', y_val)
        np.save(data_dir+'/y_test.npy', y_test)
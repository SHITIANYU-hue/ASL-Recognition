import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def process_sign_language_MNIST_dataset(rootdir):
    """
    Link to dataset: https://www.kaggle.com/datamunge/sign-language-mnist

    Args: 
        rootdir(str) - path to the MNIST dataset parent folder

    Returns: 
        data (tensor)
        labels (list) 
    """
    
    # Parse data from csv file
    data = pd.read_csv(rootdir)
    data_labels = data["label"]
    data = data.drop(["label"],axis=1)
    data = data.to_numpy()
    data = data.reshape(data.shape[0], 28,28,1).astype('float64')

    # Reformat data labels to lower case letters
    formatted_labels = []
    letters = "abcdefghijklmnopqrstuvwxyz"
    for label in data_labels.values.tolist():
        formatted_labels.append(letters[label])

    # Resize data
    resizedData = []
    for img in data:
        resizedImg = cv2.resize(img, (50,50), interpolation = cv2.INTER_CUBIC)
        resizedData.append(resizedImg)

    # Plot first 24 images
    col, ax = plt.subplots(4,6) 
    col.set_size_inches(10, 10)
    idx = 0
    for i in range(4):
        for j in range(6):
            ax[i,j].imshow(data[idx] , cmap = "gray")
            idx += 1
        plt.tight_layout()   
    plt.show()

    return torch.tensor(np.array(resizedData)), formatted_labels

def process_massey_gesture_dataset(path_to_files):
    ret_data = [] 
    ret_label = []

    for subdir, dirs, files in os.walk(path_to_files):
        for file in files:
            if "DS_Store" in file:
                continue
            image = os.path.join(subdir, file)
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            resized = cv2.resize(img, (50,50))
            ret_data.append(resized)
            ret_label.append(image)
    
    # Plot first 24 resized images
    col, ax = plt.subplots(4,6) 
    col.set_size_inches(10, 10)
    idx = 0
    for i in range(4):
        for j in range(6):
            ax[i,j].imshow(ret_data[idx] , cmap = "gray")
            idx += 1
        plt.tight_layout()   
    plt.show()

    return ret_data, ret_label

def process_fingerspelling_A_dataset(rootdir):
    """
    Link to dataset: https://empslocal.ex.ac.uk/people/staff/np331/index.php?section=FingerSpellingDataset

    Args: 
        rootdir(str) - path to the FingerSpelling parent folder

    Returns: 
        data (list)
        labels (list) 
    
    """
    data = []
    labels = []

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            # Check if filename contains DS_Store (for MacOS) or depth (we don't want to keep depth images)
            if "DS_Store" in file or "depth" in file:
                continue
            
            # Store the image and the label
            label = os.path.basename(os.path.normpath(subdir))

            # Get the path of the image and read it
            image_path = os.path.join(subdir, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # read as greyscale

            # Resized the image as 50x50
            resized_image = cv2.resize(image, (50,50), interpolation = cv2.INTER_CUBIC)
            
            data.append(resized_image)
            labels.append(label)

            # Show the images
            # cv2.imshow('image', resized_image)
            # cv2.waitKey(0) 

    return data, labels

def one_hot_encoding(labels):
    """
    Performs one-hot-encoding on the labels

    Args: 
        labels (list) - dataset labels all in lowercase ie. ['a','b','c']

    Returns: 
        labels_encodings (tensor) - one-hot-encoding representation of the labels
    """

    # Dictionary that stores the letter to integer class conversion
    letterClass = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8,
                    'k':9, 'l':10, 'm':11, 'n':12, 'o':13, 'p':14, 'q':15, 'r':16, 's':17,
                    't':18, 'u':19, 'v':20, 'w':21, 'x':22, 'y':23}
    
    # Convert lowercase letters to numbers
    labelsClass = []
    for label in labels:
        if label in letterClass:
            labelsClass.append(letterClass[label])
        else:
            print("Incorrect label:", label)

    # Get the one-hot-encodings for the labels
    labels_encodings = torch.nn.functional.one_hot(torch.tensor(labelsClass), num_classes=24)
    
    return labels_encodings

if __name__ == "__main__":
    print("Processing MNIST...")
    data_mnist, labels_mnist = process_sign_language_MNIST_dataset("datasets/mnist/sign_mnist_data.csv")
    print("Processing MNIST done!")
    
    print("Processing Massey...")
    data_massey, labels_massey = process_massey_gesture_dataset("datasets/massey") # path to massey directory. massey directory has subdir 1,2,3,4,5 
    print("Processing Massey done!")

    print("Processing FingerSpelling...")
    data_fs, labels_fs = process_fingerspelling_A_dataset("datasets/fingerspelling")
    print("Processing FingerSpelling done!")


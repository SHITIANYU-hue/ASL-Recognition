import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def process_sign_language_MNIST_dataset(filename):
    
    # Parse data from csv file
    data = pd.read_csv(filename)
    data_labels = data["label"]
    data = data.drop(["label"],axis=1)
    data = data.to_numpy()
    data = data.reshape(data.shape[0], 28,28,1).astype('float64')

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

    # Resize data
    resizedData = []
    for img in data:
        resizedImg = cv2.resize(img, (50,50), interpolation = cv2.INTER_CUBIC)
        resizedData.append(resizedImg)

    # Plot first 24 resized images
    col, ax = plt.subplots(4,6) 
    col.set_size_inches(10, 10)
    idx = 0
    for i in range(4):
        for j in range(6):
            ax[i,j].imshow(resizedData[idx] , cmap = "gray")
            idx += 1
        plt.tight_layout()   
    plt.show()

    return resizedData, data_labels

def process_massey_gesture_dataset(path_to_files):
    #files = ["/Users/kinakim/Desktop/massey/1", "/Users/kinakim/Desktop/massey/2", "/Users/kinakim/Desktop/massey/3", "/Users/kinakim/Desktop/massey/4", "/Users/kinakim/Desktop/massey/5"]
    ret_data = [] 
    ret_label = []

    for list_of_files in os.listdir(path_to_files): 
        filepath = path_to_files+'/'+list_of_files
        if os.path.isdir(filepath):
            os.chdir(filepath)
            for image in os.listdir(filepath):
                img = cv2.imread(image)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (50,50))
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

if __name__ == "__main__":
    data_mnist, labels_mnist = process_sign_language_MNIST_dataset("datasets/sign_mnist_data.csv")
    data_fs, labels_fs = process_fingerspelling_A_dataset("datasets/fingerspelling")
    data_massey, labels_massey = process_massey_gesture_dataset("datasets/massey") # path to massey directory. massey directory has subdir 1,2,3,4,5 


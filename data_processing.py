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

def process_fingerspelling_A_dataset():
    pass

if __name__ == "__main__":
    data, labels = process_sign_language_MNIST_dataset("sign_mnist_data.csv")
    data_massey, labels_massey = process_massey_gesture_dataset("./massey") # path to massey directory. massey directory has subdir 1,2,3,4,5 
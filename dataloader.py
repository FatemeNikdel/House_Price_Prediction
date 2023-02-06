
import cv2
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

class PreProcessing():

    def __init__(self, im_path, txt_path):
        self.im_path = im_path
        self.txt_path = txt_path

    def load_images(self):
        # Variables
        im_data   = []
        rooms = []
        house_number = []
        # Read Dataset
        for i , name in enumerate(glob.glob(self.im_path + "\\*")):
            # Read images
            img = cv2.imread(name)
            # Resize and Normalize
            img = cv2.resize(img, (32, 32))/255.0
            # RGB Color
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    % CV2 read images in GBR colors
            # Create Dataset
            im_data.append(img)
            # Create Labels
            rooms.append(name.split("\\")[-1].split('_')[-1].split('.')[-2]) 
            house_number.append(name.split("\\")[-1].split('_')[0])
            if i % 100 == 0:
                    print(f"[INFO]: {i}/21000 processed!")
        
        return im_data, rooms, house_number

    def read_text(self):
        txt_data = []
        # Read Text file
        df = pd.read_csv(self.txt_path,  sep=" ", 
                  names=["F1", "F2", "F3", "F4", "Price"])
        # Consider the four first Column as inputs
        txt_data = df.loc[:,["F1", "F2", "F3", "F4"]]
        # Consider the  last Column as Label
        labels = df.loc[:,["Price"]]

        return txt_data, labels
    
    def label_binarizer(self, labels):
        LB = LabelBinarizer()
        all_labels = LB.fit_transform(labels)

        return all_labels
    
    def train_test_split(txt_data, im_data, labels):
        txt_train, txt_test,labels_train, labels_test = train_test_split(txt_data, labels, test_size = 0.2 )
        img_train, img_test = train_test_split(im_data, labels, test_size = 0.2 )
        return txt_train, txt_test, img_train, img_test, labels_train, labels_test



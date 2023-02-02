
from sklearn.model_selection import train_test_split
import cv2
import glob
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import pandas as pd

class PreProcessing():

    def __init__(self, im_path, txt_path):
        self.im_path = im_path
        self.txt_path = txt_path

    def load_data(self):
        im_data   = []
        rooms = []
        house_number = []
        for i , name in enumerate(glob.glob(self.im_path + "\\*")):
            # Read images
            img = cv2.imread(name)
            # Resize and Normalize
            img = cv2.resize(img, (32, 32))/255.0
            # RGB Color
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Create Dataset
            im_data.append(img)
            # Create Labels
            rooms.append(name.split("\\")[-1].split('_')[-1].split('.')[-2]) 
            house_number.append(name.split("\\")[-2].split('_')[0])

            if i % 100 == 0:
                print(f"[INFO]: {i}/25000 processed!")
            
            return im_data, rooms, house_number

    def read_text(self):
        txt_data = []
        # Read Text file
        df = pd.read_csv(self.txt_path)
        # Consider the four first Column as inputs
        








    def label_binarizer(house_number):
        color_label = np.array(house_number)
        Color_LB = LabelBinarizer()
        color_label = category_LB.fit_transform(house_number)
        
        return category_label, color_label

    def train_test_split(data, category_label, color_label):
        split = train_test_split(data, category_label, color_label, test_size = 0.2 )
        X_train, X_test, Y_train_category, Y_test_category, Y_train_color, Y_test_color = split

        return X_train, X_test, Y_train_category, Y_test_category, Y_train_color, Y_test_color
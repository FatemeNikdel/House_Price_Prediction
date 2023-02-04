import tensorflow as tf
import glob 
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from load_data import PreProcessing

im_path = r"house_dataset"
txt_path = r"HousesInfo.txt"

Data = PreProcessing(im_path, txt_path)
im_data, rooms, house_number = Data.data()
#txt_data, labels = Data.read_text()
#print(labels)
#labels = Data.label_binarizer(labels)
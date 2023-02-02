import tensorflow as tf
import glob 
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

im_path = r"house_dataset"
txt_path = r"HousesInfo.txt"
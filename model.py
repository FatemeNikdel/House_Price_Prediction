import tensorflow as tf
from keras import layers, models

class FashionNet():

    def __init__(self,txt_train, txt_test, img_train, img_test, labels_train, labels_test, epochs):
        self.txt_train = txt_train
        self.txt_test  = txt_test
        self.img_train = img_train
        self.img_test  = img_test
        self.labels_train = labels_train
        self.labels_test  = labels_test
        self.epochs = epochs
    
    

    
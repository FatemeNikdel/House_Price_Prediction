import tensorflow as tf
from keras import layers, models

class HousePricePrediction():

    def __init__(self,txt_train, txt_test, img_train, img_test, labels_train, labels_test, epochs, delta):
        self.txt_train = txt_train
        self.txt_test  = txt_test
        self.img_train = img_train
        self.img_test  = img_test
        self.labels_train = labels_train
        self.labels_test  = labels_test
        self.epochs = epochs
        self.delta = delta
    
    def step():
        pass()

    ## Step : Define Exclusive Loss Function
    def Huber_loss(self):
        def Huber( y_true, y_pred):
            # Calculate the error
            error = y_true - y_pred
            # Check if error is small, Return True or False
            small_error = tf.abs(error) <= self.delta
            # Use MSE for small error
            small_loss  = tf.square(error)/2
            # Use MAE for big error
            big_loss = self.delta * (error) - (self.delta**2)/2
            return tf.where(small_error, small_loss, big_loss)
        return Huber


    def Train():
        pass()

    def compile():
        pass()
    

    


    
import tensorflow as tf
from keras import layers, models
import custom_layers

class FashionNet():

    def __init__(self, txt_train, txt_test, bathroom_train, bathroom_test, kitchen_train, kitchen_test, frontal_train, frontal_test, bedroom_train, bedroom_test, labels_train, labels_test, epochs):
        self.txt_train      = txt_train
        self.txt_test       = txt_test
        self.bathroom_train = bathroom_train
        self.bathroom_test  = bathroom_test
        self.kitchen_train  = kitchen_train
        self.kitchen_test   = kitchen_test
        self.frontal_train  = frontal_train
        self.frontal_test   = frontal_test
        self.bedroom_train  = bedroom_train
        self.bedroom_test   = bedroom_test
        self.labels_train   = labels_train
        self.labels_test    = labels_test
        self.epochs = epochs
    def bedroom_model(self):
        bedroom_input = layers.Input(shape = (32, 32, 3))
        x = custom_layers.Conv2D(64, (3,3), activation = "relu")(bedroom_input)
        x = custom_layers.Conv2D(64, (3,3), activation = "relu")(x)
        x = custom_layers.MaxPooling()(x)
        x = custom_layers.Conv2D(128, (3,3), activation = "relu")(x)
        x = custom_layers.Conv2D(128, (3,3), activation = "relu")(x)
        x = custom_layers.MaxPooling()(x)
        x = custom_layers.Conv2D(256, (3,3), activation = "relu")(x)
        x = custom_layers.Conv2D(256, (3,3), activation = "relu")(x)
        x = custom_layers.Conv2D(256, (3,3), activation = "relu")(x)
        x = custom_layers.MaxPooling()(x)
        x = custom_layers.Conv2D(512, (3,3), activation = "relu")(x)
        x = custom_layers.Conv2D(512, (3,3), activation = "relu")(x)
        x = custom_layers.Conv2D(512, (3,3), activation = "relu")(x)
        x = custom_layers.MaxPooling()(x)
        x = custom_layers.Conv2D(512, (3,3), strides = (2,2), activation = "relu")(x)
        x = custom_layers.Conv2D(512, (3,3), strides = (2,2),activation = "relu")(x)
        x = custom_layers.Conv2D(512, (3,3), strides = (2,2),activation = "relu")(x)
        x = custom_layers.MaxPooling()(x)
        return x

    def  bathroom_model(self):
           


        # It concatenate the layer in depth axis
        concat_input = layers.concatenate([x, y, z, w], axis = 2)
        flat_input   = layers.flatten()(concat_input)
        out = custom_layers.Dense(50)(flat_input)
        out = custom_layers.relu()(out)
        out = custom_layers.Dense(1)(out)
        out = custom_layers.MAE()(out)

        

    def step(x_train, label):
        with tf.GradientTape() as tape:
            y_prime = model(x_train)
    """ net = models.Model(inputs = input_layer, 
                        outputs = [category_output, color_output],
                        name = "FashionNet")
    
    losses = { "category_output": "categorical_crossentropy",
                "color_output"  : "categorical_crossentropy" }
    loss_weights = { "category_output": 1.0,
                        "color_output"  : 1.0 }
    net.compile(optimizer = "adam",
                loss = losses,
                loss_weights = loss_weights,
                metrics = ['accuracy'])
    H = net.fit(x = self.X_train, 
                y = {"category_output": self.Y_train_category,
                "color_output"  : self.Y_train_color},
                validation_data = (self.X_test,
                {"category_output": self.Y_test_category,
                "color_output"  : self.Y_test_color}),
                epochs = self.epochs,
                verbose = 1)"""
                        

    
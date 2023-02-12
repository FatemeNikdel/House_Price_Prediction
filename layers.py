import tensorflow as tf
from keras.layers import Layer
from keras.losses import Loss

## Define Custom Loss Function
class HuberLoss(Loss):
    # Parameters
    def __init__(self, delta):
        super().__init__()
        self.delta = delta
    # Computations
    def call(self, y_true, y_pred):
        # Calculate the error
            error = y_true - y_pred
            # Check if error is small, Return True or False
            small_error = tf.abs(error) <= self.delta
            # Use MSE for small error
            small_loss  = tf.square(error)/2
            # Use MAE for big error
            big_loss = self.delta * (error) - (self.delta**2)/2
            return tf.where(small_error, small_loss, big_loss)

## Define Custom Dense Layer
class Dense(Layer):
    # Parameters
    def __init__(self, units):
        """
        Parameters
        ___________
        - units: Number of neurons in a layer
        """
        super().__init__()
        self.units = units
    # States
    def build(self, input_shape):
        # Initialize weights
        initialize_weight = tf.random_normal_initializer(mean= 0.0, stddev= 0.05, seed= None)
        self.w = tf.Variable(initial_value = initialize_weight(shape =(input_shape[-1], self.units)),
                            dtype ="float32",
                            trainable = True,
                            name = "weights")
        # Initialize bias
        initialize_bias = tf.zeros_initializer()
        self.b= tf.Variable(initial_value = initialize_bias(shape =(1, self.units)),
                            dtype ="float32",
                            trainable = True,
                            name = "bias") 
    # Computations
    def call(self,inputs):
        return tf.matmul(inputs, self.w) + self.b

## Define Custom Conv2D Layer
class Conv2D:
    # Parameters
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='SAME'):
        """
        Initialize the Conv2D layer with the given parameters.
        Parameters
        ___________
        - filters: integer, number of filters in the Convolution layer
        - kernel_size: tuple of two integers, size of the Convolution kernel
        - strides: tuple of two integers, steps taken by the Convolution kernel
        - padding: string, type of padding to use ('SAME' or 'VALID')
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
    # States    
    def build(self, input_shape):
        """
        Build the Convolution layer.
        Parameters
        ___________
        - input_shape: tuple of integers, shape of the input tensor
        """
        self.kernel = tf.Variable(tf.random.normal(
            shape=(self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters)))
    # Computations   
    def call(self, inputs):
        """
        Apply the Convolution layer to the input tensor.

        Parameters
        ___________
        - inputs: tensor, input to the Convolution layer

        Output
        _______
        - return: tensor, output of the Convolution layer
        """
        return tf.nn.convolution(inputs, self.kernel, self.padding, self.strides)

## Define Custom MaxPooling Layer
class MaxPooling(tf.keras.layers.Layer):
    # Parameters
    def __init__(self, pool_size, strides, padding, **kwargs):
        """
        Initializes the max pooling layer with the given pool size, strides, and padding.
        Parameters
        ___________
        - pool_size: Size of the pooling window
        - strides: Stride of the pooling operation
        - padding: Type of padding to be used
        - kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    # Computations
    def call(self, inputs, training=None):
        """
        Applies the max pooling operation on the input tensor.
        Parameters
        ___________
        - inputs: Input tensor
        - training: Flag indicating if the layer is being run in training mode or not

        Output
        _______
        - return: Result of the max pooling operation
        """
        return tf.nn.max_pool2d(inputs, 
                                ksize=self.pool_size, 
                                strides=self.strides, 
                                padding=self.padding)
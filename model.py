import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Number of frequency buckets during inference
K = 129
# Number of mel frequency bands during loss calculation
M = 80
relu = tf.keras.activations.relu


mel_matrix_what = tf.signal.linear_to_mel_weight_matrix(M, 513, sample_rate=22050)

generator = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=K, kernel_size=40, padding='causal', groups=K, activation=relu),
    tf.keras.layers.Conv1D(filters=K, kernel_size=40, padding='causal', groups=K, activation=relu),
    tf.keras.layers.Conv1D(filters=K, kernel_size=40, padding='causal', groups=K, activation=relu),
    tf.keras.layers.Conv1D(filters=K, kernel_size=40, padding='causal', groups=K)
])

def generator_coherence_loss(big_x, big_y):
    # big_x = tf.signal.stft(x, 1024, 256)
    # big_y = tf.signal.stft(y, 1024, 256)
    
    # need to reduce across only the time dimension; keep separate freqs
    # which axis is that????
    numerator = tf.math.reduce_sum(big_x * big_y, axis=0)
    numerator = tf.math.abs(numerator)
    numerator = tf.tensordot(numerator, mel_matrix_what, 1)
    denominator = tf.math.reduce_sum(tf.square(big_y))
    denominator = denominator * tf.math.reduce_sum(tf.square(big_x))
    denominator = tf.sqrt(denominator)
    denominator = tf.tensordot(denominator, mel_matrix_what, 1)
    # This should be a scalar
    return tf.math.reduce_sum(numerator / denominator) / M

def generator_mel_loss(big_x, big_y):
    # big_x = tf.signal.stft(x, 1024, 256)
    # big_y = tf.signal.stft(y, 1024, 256)
    y_term = tf.tensordot(tf.square(tf.math.abs(big_y)), mel_matrix_what, 1)
    x_term = tf.tensordot(tf.square(tf.math.abs(big_x)), mel_matrix_what, 1)
    # Which axis is time frame count (L)?
    return tf.math.reduce_sum(tf.math.abs(y_term - x_term)) / M / x_term.shape[0]

# Adapted from jik876/hifi-gan, available under the MIT license
# MIT License

# Copyright (c) 2020 Jungil Kong

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

class PeriodDiscriminator(tf.keras.Layer):
    def __init__(self, period):
        super().__init__()
        self.period = period
        
        kernel_size = (5, 1)
        stride = (3, 1)
        L1 = tf.keras.regularizers.L1
        
        self.convs_list = [
            tf.keras.layers.Reshape((period, -1)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, strides=stride, padding='same', activation=tf.nn.leaky_relu, kernel_regularizer=L1(), bias_regularizer=L1()),
            tf.keras.layers.Conv2D(filters=128, kernel_size=kernel_size, strides=stride, padding='same', activation=tf.nn.leaky_relu, kernel_regularizer=L1(), bias_regularizer=L1()),
            tf.keras.layers.Conv2D(filters=512, kernel_size=kernel_size, strides=stride, padding='same', activation=tf.nn.leaky_relu, kernel_regularizer=L1(), bias_regularizer=L1()),
            tf.keras.layers.Conv2D(filters=1024, kernel_size=kernel_size, strides=stride, padding='same', activation=tf.nn.leaky_relu, kernel_regularizer=L1(), bias_regularizer=L1()),
            tf.keras.layers.Conv2D(filters=1024, kernel_size=kernel_size, strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, kernel_regularizer=L1(), bias_regularizer=L1()),
            tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 1), strides=(1, 1), padding='same', kernel_regularizer=L1(), bias_regularizer=L1()),
        ])

    def build(self, input_shape):
        # Which axis is time frame count (L)?
        frame_count = input_shape[0]
        period = self.period
        padding = (period - frame_count % period) % period
        self.convs_list.insert(0, tf.keras.layers.ZeroPadding1D(padding))
        self.convs = tf.keras.Sequential(self.convs_list)
    
    def call(self, inputs):
        return self.convs(inputs)


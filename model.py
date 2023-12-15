import tensorflow as tf
from pathlib import Path

print("TensorFlow version:", tf.__version__)




# https://www.tensorflow.org/tutorials/audio/simple_audio
# Run generate_file_list.py and preprocess.py first to create a simple enough audio dataset to easily import
full_ds = tf.keras.utils.audio_dataset_from_directory('musdb18hq-processed/')
print(full_ds)
assert False    














# Number of frequency buckets during inference
K = 129
# Number of mel frequency bands during loss calculation
M = 80
relu = tf.keras.activations.relu


mel_matrix_what = tf.signal.linear_to_mel_weight_matrix(M, 513, sample_rate=22050)

class Complex2Real(tf.keras.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        real_inputs = tf.math.real(inputs)
        imag_inputs = tf.math.imag(inputs)
        # Help, what shape is the tensor
        assert False

class Real2Complex(tf.keras.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        # Help, what shape is the tensor
        real_inputs = inputs[:,:,::2]
        imag_inputs = inputs[:,:,1::2]
        return tf.complex(real_inputs, imag_inputs)

generator = tf.keras.Sequential([
    Complex2Real(),
    tf.keras.layers.Conv1D(filters=K, kernel_size=40, padding='causal', groups=K, activation=relu),
    tf.keras.layers.Conv1D(filters=K, kernel_size=40, padding='causal', groups=K, activation=relu),
    tf.keras.layers.Conv1D(filters=K, kernel_size=40, padding='causal', groups=K, activation=relu),
    tf.keras.layers.Conv1D(filters=K, kernel_size=40, padding='causal', groups=K),
    Real2Complex()
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
        
        self.convs_list = [
            tf.keras.layers.Reshape((period, -1)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, strides=stride, padding='same'),
            tf.keras.layers.UnitNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(filters=128, kernel_size=kernel_size, strides=stride, padding='same'),
            tf.keras.layers.UnitNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(filters=512, kernel_size=kernel_size, strides=stride, padding='same'),
            tf.keras.layers.UnitNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(filters=1024, kernel_size=kernel_size, strides=stride, padding='same'),
            tf.keras.layers.UnitNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(filters=1024, kernel_size=kernel_size, strides=(1, 1), padding='same'),
            tf.keras.layers.UnitNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 1), strides=(1, 1), padding='same'),
            tf.keras.layers.UnitNormalization()
        ]

    def build(self, input_shape):
        # Which axis is time frame count (L)?
        frame_count = input_shape[0]
        period = self.period
        padding = (period - frame_count % period) % period
        self.convs_list.insert(0, tf.keras.layers.ZeroPadding1D(padding))
        self.convs = tf.keras.Sequential(self.convs_list)
    
    def call(self, inputs):
        return self.convs(inputs)

def append_spectral_normalized_layer(layer_list, layer):
    layer_list.append(tf.keras.layers.SpectralNormalization(layer))

def append_weight_normalized_layer(layer_list, layer):
    layer_list.append(layer)
    layer_list.append(tf.keras.layers.UnitNormalization())

class ScaleDiscriminator(tf.keras.Layer):
    def __init__(self, mean_pool_count, use_spectral_norm):
        super().__init__()
        add_normalized_layer = append_spectral_normalized_layer if use_spectral_norm else append_weight_normalized_layer
        
        convs_list = []
        for _ in range(mean_pool_count):
            convs_list.append(tf.keras.layers.AvgPooling1D(pool_size=4, strides=2, padding='same'))
        add_normalized_layer(convs_list, tf.keras.layers.Conv1D(filters=128, kernel_size=15, strides=1, padding='same'))
        convs_list.append(tf.keras.layers.LeakyReLU(0.1))
        add_normalized_layer(convs_list, tf.keras.layers.Conv1D(filters=128, kernel_size=41, strides=2, padding='same', groups=4))
        convs_list.append(tf.keras.layers.LeakyReLU(0.1))
        add_normalized_layer(convs_list, tf.keras.layers.Conv1D(filters=256, kernel_size=41, strides=2, padding='same', groups=16))
        convs_list.append(tf.keras.layers.LeakyReLU(0.1))
        add_normalized_layer(convs_list, tf.keras.layers.Conv1D(filters=512, kernel_size=41, strides=4, padding='same', groups=16))
        convs_list.append(tf.keras.layers.LeakyReLU(0.1))
        add_normalized_layer(convs_list, tf.keras.layers.Conv1D(filters=1024, kernel_size=41, strides=4, padding='same', groups=16))
        convs_list.append(tf.keras.layers.LeakyReLU(0.1))
        add_normalized_layer(convs_list, tf.keras.layers.Conv1D(filters=1024, kernel_size=41, strides=1, padding='same', groups=16))
        convs_list.append(tf.keras.layers.LeakyReLU(0.1))
        add_normalized_layer(convs_list, tf.keras.layers.Conv1D(filters=1024, kernel_size=5, strides=1, padding='same'))
        convs_list.append(tf.keras.layers.LeakyReLU(0.1))
        add_normalized_layer(convs_list, tf.keras.layers.Conv1D(filters=1, kernel_size=3, strides=1, padding='same'))
        
        self.convs = tf.keras.Sequential(convs_list)

    def call(self, inputs):
        return self.convs(inputs)

# End HiFi-GAN code

def generator_adversarial_loss_part(d_of_y):
    # The generator only cares about how well it was able to fool the discriminator,
    # not how well the discriminator can identify genuine audio.
    return tf.square(d_of_y - tf.ones_like(d_of_y))

def discriminator_adversarial_loss_part(d_of_x, d_of_y):
    # The discriminator cares about both kinds of failures it makes
    return tf.square(d_of_x - tf.ones_like(d_of_x)) + tf.square(d_of_y)


coh_loss_scale = tf.constant(2.5)
mel_loss_scale = tf.constant(5.625)

generator_optimizer = tf.keras.optimizers.AdamW(learning_rate=1.6e-3, beta_1=0.8, beta_2=0.99)
discriminator_optimizer = tf.keras.optimizers.AdamW(learning_rate=1.6e-3, beta_1=0.8, beta_2=0.99)

def train_step(x):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        big_x_gen = tf.signal.stft(x, 116, 58)
        big_y_gen = generator(big_x_gen)
        y = tf.signal.inverse_stft(big_y_gen, 116, 58, window_fn=tf.signal.inverse_stft_window_fn(58))
        
        big_x = tf.signal.stft(x, 1024, 256)
        big_y = tf.signal.stft(y, 1024, 256)
    assert False
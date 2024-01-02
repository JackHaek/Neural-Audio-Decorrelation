#!/usr/bin/env python3
import tensorflow as tf
import datetime
from pathlib import Path
import time


# Spectral normalization layers in Keras were introduced in TF 2.13.0, but ROSIE is running 2.12.0
try:
    from tensorflow.keras.layers import SpectralNormalization
except ImportError:
    from SpectralNormalization import SpectralNormalization

print("TensorFlow version:", tf.__version__)

# https://www.tensorflow.org/tutorials/audio/simple_audio
# Run generate_file_list.py and preprocess.py first to create a simple enough audio dataset to easily import
full_ds = tf.keras.utils.audio_dataset_from_directory('musdb18hq-processed/', class_names=('train', 'validation', 'test'))
# Get rid of the labels and divide into parts
train_ds = full_ds.unbatch().filter(lambda _, x: x == 0).map(lambda x, _: x).batch(4)
val_ds = full_ds.unbatch().filter(lambda _, x: x == 1).map(lambda x, _: x).batch(4)
test_ds = full_ds.unbatch().filter(lambda _, x: x == 2).map(lambda x, _: x).batch(4)

# Each entry taken from one of these datasets has dimensions (batch_size, time frames, channels)
# (in this case, (16, 75400, 2) for full batches)














# Number of frequency buckets during inference
K = 65
# Number of mel frequency bands during loss calculation
M = tf.constant(80)
M_float = tf.cast(M, tf.float32)

mel_xform_matrix = tf.signal.linear_to_mel_weight_matrix(M, 513, sample_rate=22050)

class Complex2Real(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        # Add a another dimension of size 1 to make this easier
        inputs = inputs[..., tf.newaxis]
        real_inputs = tf.math.real(inputs)
        imag_inputs = tf.math.imag(inputs)
        # Stack real and imaginary across the new axis, to put the real and imaginary components next to each other
        output = tf.concat((real_inputs, imag_inputs), axis=3)
        output_shape = output.shape
        output = tf.reshape(output, (output_shape[0], output_shape[1], -1))
        return output
        

class Real2Complex(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        real_inputs = inputs[:,:,::2]
        imag_inputs = inputs[:,:,1::2]
        return tf.complex(real_inputs, imag_inputs)

relu = tf.keras.activations.relu
generator = tf.keras.Sequential([
    Complex2Real(),
    # Even though they're causal convolutions, we expect the caller to init
    # the padding, with useful info or with zeroes as appropriate, so use
    # "valid" padding here.
    tf.keras.layers.Conv1D(filters=K*16, kernel_size=40, padding='valid', groups=K, activation=relu),
    tf.keras.layers.Conv1D(filters=K*16, kernel_size=40, padding='valid', groups=K, activation=relu),
    tf.keras.layers.Conv1D(filters=K*16, kernel_size=40, padding='valid', groups=K, activation=relu),
    tf.keras.layers.Conv1D(filters=K*2,  kernel_size=40, padding='valid', groups=K),
    Real2Complex()
])

def generator_coherence_loss(big_x, big_y):
    # big_x = tf.signal.stft(x, 1024, 256)
    # big_y = tf.signal.stft(y, 1024, 256)
    # The shape of big_x is (batch size * 2, STFT time frames, STFT freq buckets)
    
    # need to reduce across only the time dimension; keep separate freqs
    numerator = tf.reduce_sum(big_x * big_y, axis=1)
    numerator = tf.abs(numerator)
    # tf.print(numerator.shape)
    numerator = tf.tensordot(numerator, mel_xform_matrix, 1)
    # tf.print(numerator.shape)
    denominator = tf.reduce_sum(tf.square(tf.abs(big_y)), axis=1)
    denominator = denominator * tf.reduce_sum(tf.square(tf.abs(big_x)), axis=1)
    denominator = tf.sqrt(denominator)
    denominator = tf.tensordot(denominator, mel_xform_matrix, 1)
    # This should be a scalar
    return tf.math.reduce_sum(numerator / denominator) / M_float

def generator_mel_loss(big_x, big_y):
    # big_x = tf.signal.stft(x, 1024, 256)
    # big_y = tf.signal.stft(y, 1024, 256)
    y_term = tf.tensordot(tf.square(tf.abs(big_y)), mel_xform_matrix, 1)
    x_term = tf.tensordot(tf.square(tf.abs(big_x)), mel_xform_matrix, 1)
    return tf.math.reduce_sum(tf.abs(y_term - x_term)) / M_float / x_term.shape[1]

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

class PeriodDiscriminator(tf.keras.layers.Layer):
    def __init__(self, period):
        super().__init__()
        self.period = period
        
        kernel_size = (5, 1)
        stride = (3, 1)
        
        self.convs_list = [
            tf.keras.layers.Reshape((-1, period, 1)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, strides=stride, padding='same', data_format='channels_last'),
            tf.keras.layers.UnitNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(filters=128, kernel_size=kernel_size, strides=stride, padding='same', data_format='channels_last'),
            tf.keras.layers.UnitNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(filters=512, kernel_size=kernel_size, strides=stride, padding='same', data_format='channels_last'),
            tf.keras.layers.UnitNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(filters=1024, kernel_size=kernel_size, strides=stride, padding='same', data_format='channels_last'),
            tf.keras.layers.UnitNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(filters=1024, kernel_size=kernel_size, strides=(1, 1), padding='same', data_format='channels_last'),
            tf.keras.layers.UnitNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 1), strides=(1, 1), padding='same', data_format='channels_last'),
            tf.keras.layers.UnitNormalization()
        ]

    def build(self, input_shape):
        # Add time frames to the input
        frame_count = input_shape[1]
        period = self.period
        padding = (period - frame_count % period) % period
        self.convs_list.insert(0, tf.keras.layers.ZeroPadding1D((0, padding)))
        self.convs = tf.keras.Sequential(self.convs_list)
    
    def call(self, inputs):
        output = self.convs(inputs)
        # I think we want one neat loss number per signal, so... let's smash this all down
        output = tf.reduce_mean(output, axis=2)
        output = tf.reduce_mean(output, axis=1)
        return output

def append_spectral_normalized_layer(layer_list, layer):
    layer_list.append(SpectralNormalization(layer))

def append_weight_normalized_layer(layer_list, layer):
    layer_list.append(layer)
    layer_list.append(tf.keras.layers.UnitNormalization())

class ScaleDiscriminator(tf.keras.layers.Layer):
    def __init__(self, mean_pool_count, use_spectral_norm):
        super().__init__()
        add_normalized_layer = append_spectral_normalized_layer if use_spectral_norm else append_weight_normalized_layer
        
        convs_list = []
        for _ in range(mean_pool_count):
            convs_list.append(tf.keras.layers.AveragePooling1D(pool_size=4, strides=2, padding='same'))
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
        output = self.convs(inputs)
        # Squash it down I guess
        output = tf.reduce_mean(output, axis=1)
        return output

# End HiFi-GAN code

class Discriminator(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self.p2 = PeriodDiscriminator(2)
        self.p3 = PeriodDiscriminator(3)
        self.p5 = PeriodDiscriminator(5)
        self.p7 = PeriodDiscriminator(7)
        self.p11 = PeriodDiscriminator(11)
        self.s0 = ScaleDiscriminator(0, True)
        self.s1 = ScaleDiscriminator(1, False)
        self.s2 = ScaleDiscriminator(2, False)

    def call(self, inputs):
        op2 = self.p2(inputs)
        op3 = self.p3(inputs)
        op5 = self.p5(inputs)
        op7 = self.p7(inputs)
        op11 = self.p11(inputs)
        os0 = self.s0(inputs)
        os1 = self.s1(inputs)
        os2 = self.s2(inputs)
        outputs = tf.concat((op2, op3, op5, op7, op11, os0, os1, os2), axis=1)
        return outputs

def generator_adversarial_loss(d_of_y):
    # The generator only cares about how well it was able to fool the discriminator,
    # not how well the discriminator can identify genuine audio.
    # Discriminator output of 1 = judged to be original
    # Discriminator output of 0 = judged to be generator output
    mse = tf.square(d_of_y - tf.ones_like(d_of_y))
    return tf.reduce_sum(mse) / 8

def discriminator_adversarial_loss(d_of_x, d_of_y):
    # The discriminator cares about both kinds of failures it makes
    mse = tf.square(d_of_x - tf.ones_like(d_of_x)) + tf.square(d_of_y)
    return tf.reduce_sum(mse) / 8


discriminator = Discriminator()

coh_loss_scale = tf.constant(2.5)
mel_loss_scale = tf.constant(5.625)

generator_optimizer = tf.keras.optimizers.AdamW(learning_rate=1.6e-3, beta_1=0.8, beta_2=0.99)
discriminator_optimizer = tf.keras.optimizers.AdamW(learning_rate=1.6e-3, beta_1=0.8, beta_2=0.99)

train_gen_loss_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
train_disc_loss_metric = tf.keras.metrics.Mean('disc_loss', dtype=tf.float32)
valid_gen_loss_metric = tf.keras.metrics.Mean('validation_generator_loss', dtype=tf.float32)
valid_disc_loss_metric = tf.keras.metrics.Mean('validation_discriminator_loss', dtype=tf.float32)

@tf.function
def train_step(x, train=True):
    print('Tracing! train_step')
    generator_loss = None
    discrim_loss = None
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        x = tf.squeeze(x, [2])
        tf.print(x.shape)
        big_x_gen = tf.signal.stft(x, 116, 58)
        # Each channel of the audio has an FFT size of (1299, 65), and we have one channel (mono audio)
        # (116 fits into 128, meaning the frequencies range 0-64, and (1300 - 1) * 58 STFT frames could be made)
        # (This includes 160 frames of context that we added in the preprocessing stage.)
        # Anyway, the shape of big_x_gen is now (batch size, STFT time frames, STFT freq buckets)
        big_y_gen = generator(big_x_gen)
        y = tf.signal.inverse_stft(big_y_gen, 116, 58, window_fn=tf.signal.inverse_stft_window_fn(58))
        
        x_cut = x[:,-66120:]
        y_cut = y[:,-66120:]
        
        # The discriminators are slow! Don't run them while testing loss calculations
        x_discrim_output = discriminator(x_cut[..., tf.newaxis])
        y_discrim_output = discriminator(y_cut[..., tf.newaxis])
        
        # Say that the discriminator did its job perfectly
        # x_discrim_output = tf.ones((x_0.shape[0], 8))
        # y_discrim_output = tf.zeros((x_0.shape[0], 8))
        
        tf.debugging.assert_shapes([
            (x_discrim_output, (x.shape[0], 8))
        ])
        
        big_x = tf.signal.stft(x_cut, 1024, 256)
        big_y = tf.signal.stft(y_cut, 1024, 256)
        
        coh_loss = generator_coherence_loss(big_x, big_y)
        mel_loss = generator_mel_loss(big_x, big_y)
        adv_loss = generator_adversarial_loss(y_discrim_output)
        
        generator_loss = coh_loss * coh_loss_scale + mel_loss * mel_loss_scale + adv_loss
        discrim_loss = discriminator_adversarial_loss(x_discrim_output, y_discrim_output)

    if train:
        # tf.print('Calculating generator gradients')
        gen_gradients = gen_tape.gradient(generator_loss, generator.trainable_variables)
        # tf.print('Calculating discriminator gradients')
        discrim_gradients = disc_tape.gradient(discrim_loss, discriminator.trainable_variables)
        # tf.print('Applying generator gradients')
        generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
        # tf.print('Applying discriminator gradients')
        discriminator_optimizer.apply_gradients(zip(discrim_gradients, discriminator.trainable_variables))
    
    return generator_loss, discrim_loss

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
valid_log_dir = 'logs/gradient_tape/' + current_time + '/validation'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

checkpoint_prefix = './training_checkpoints/ckpt'
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)

def train(epochs):
    batch_count = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        for batch in train_ds:
            batch_count += 1
            losses = train_step(batch)
            train_gen_loss_metric(losses[0])
            train_disc_loss_metric(losses[1])
            with train_summary_writer.as_default():
                tf.summary.scalar('generator_loss', losses[0], step=batch_count)
                tf.summary.scalar('disc_loss', losses[1], step=batch_count)
        
        for batch in val_ds:
            train_step(batch, train=False)
            valid_gen_loss_metric(losses[0])
            valid_disc_loss_metric(losses[1])
        with valid_summary_writer.as_default():
            tf.summary.scalar('validation_generator_loss', valid_gen_loss_metric.result(), step=batch_count)
            tf.summary.scalar('validation_discriminator_loss', valid_disc_loss_metric.result(), step=batch_count)
        end_time = time.time()
        print('Epoch', epoch+1, 'completed after', end_time - start_time, 'seconds')
        
        template = 'Epoch {}, Training Gen-Loss: {}, Training Disc-Loss: {}, Validation Gen-Loss: {}, Validation Disc-Loss: {}'
        print (template.format(epoch+1,
                         train_gen_loss_metric.result(), 
                         train_disc_loss_metric.result(),
                         valid_gen_loss_metric.result(), 
                         valid_disc_loss_metric.result()))
                         
        # Reset metrics every epoch
        train_gen_loss_metric.reset_states()
        valid_gen_loss_metric.reset_states()
        train_disc_loss_metric.reset_states()
        valid_disc_loss_metric.reset_states()
    checkpoint.save(file_prefix=checkpoint_prefix)
    
    

train(1)
print('It completed')









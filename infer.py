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
# Note that we effectively have batching off now. I want to access them all individually.
test_ds = full_ds.unbatch().filter(lambda _, x: x == 2).map(lambda x, _: x).batch(1)

# Each entry taken from one of these datasets has dimensions (batch_size, time frames, channels)
# (in this case, (4, 75400, 1) for full batches)














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

def generator_straight_spectrogram_loss(big_x, big_y):
    # big_x = tf.signal.stft(x, 1024, 256)
    # big_y = tf.signal.stft(y, 1024, 256)
    y_term = tf.tensordot(tf.square(tf.abs(big_y)), mel_xform_matrix, 1)
    x_term = tf.tensordot(tf.square(tf.abs(big_x)), mel_xform_matrix, 1)
    return tf.math.reduce_sum(tf.abs(y_term - x_term)) / M_float / x_term.shape[1]

def generator_log_spectrogram_loss(big_x, big_y):
    # big_x = tf.signal.stft(x, 1024, 256)
    # big_y = tf.signal.stft(y, 1024, 256)
    epsilon = tf.constant(1e-8)
    ten = tf.constant(10.)
    twenty = tf.constant(20.)
    # Squaring converts from voltage levels to sound power.
    # Then converting to dB, sound power level, is 10 * log10(power)
    # But that's redundant -- squaring just doubles the log value. So fold that into the multiplier
    y_term = tf.tensordot(twenty * tf.math.log(tf.abs(big_y) + epsilon) / tf.math.log(ten), mel_xform_matrix, 1)
    x_term = tf.tensordot(twenty * tf.math.log(tf.abs(big_x) + epsilon) / tf.math.log(ten), mel_xform_matrix, 1)
    return tf.math.reduce_sum(tf.abs(y_term - x_term)) / M_float / x_term.shape[1]


checkpoint_prefix = './training_checkpoints/ckpt'
checkpoint = tf.train.Checkpoint(generator=generator)
status = checkpoint.restore(checkpoint_prefix+'-1')
status.assert_existing_objects_matched()

inference_prefix = './test_results/'

def infer(x, filenum):
    x = tf.squeeze(x, [2])
    big_x_gen = tf.signal.stft(x, 116, 58)
    # Each channel of the audio has an FFT size of (1299, 65), and we have one channel (mono audio)
    # (116 fits into 128, meaning the frequencies range 0-64, and (1300 - 1) * 58 STFT frames could be made)
    # (This includes 160 frames of context that we added in the preprocessing stage.)
    # Anyway, the shape of big_x_gen is now (batch size, STFT time frames, STFT freq buckets)
    big_y_gen = generator(big_x_gen)
    y = tf.signal.inverse_stft(big_y_gen, 116, 58, window_fn=tf.signal.inverse_stft_window_fn(58))
    
    x_cut = x[:,-66120:]
    y_cut = y[:,-66120:]
    
    
    # Mix together the stereo signal to compare to a plain mono signal
    # We use the "middle" and "side" channels, similar to how the authors of the paper did it
    #stereo_result = tf.stack(((x_cut + y_cut) / 2, (x_cut - y_cut) / 2), axis=2)
    # Get rid of the batch size dimension, ensure that a channel dimension exists instead for X
    #stereo_result = tf.squeeze(stereo_result, [0])
    #mono_result = tf.squeeze(x_cut[..., tf.newaxis], [0])
    # Encode as wav
    #stereo_result = tf.audio.encode_wav(stereo_result, tf.constant(22050))
    #mono_result = tf.audio.encode_wav(mono_result, tf.constant(22050))
    # Now, save the mono result and the stereo result with the same base filename, in the same
    # folder, to make them easier to match up with each other
    #tf.io.write_file(inference_prefix + f'{filenum}_orig.wav', mono_result)
    #tf.io.write_file(inference_prefix + f'{filenum}_stereo.wav', stereo_result)
    
    big_x = tf.signal.stft(x_cut, 1024, 256)
    big_y = tf.signal.stft(y_cut, 1024, 256)
    
    return generator_log_spectrogram_loss(big_x, big_y)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
w = tf.summary.create_file_writer(f'logs/infer/{current_time}/test')

losses = []
for i, batch in enumerate(test_ds):
    loss = infer(batch, i)
    if not tf.math.is_finite(loss):
        assert False
    losses.append(loss)

losses = tf.convert_to_tensor(losses)

with w.as_default():
    tf.summary.histogram('log_mel_spectrogram_diffs', losses, step=0)

tf.print('Done!!!!!!')
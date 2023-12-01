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
    denominator *= tf.math.reduce_sum(tf.square(big_x))
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


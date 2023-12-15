from pathlib import Path
import tensorflow as tf
import tensorflow_io as tfio

# Train-validation-test split by song is done by generate_file_list.py
def read_file_names_from_list(list_name):
    with open(list_name, 'r', encoding='utf-8') as f:
        lst = [line.rstrip() for line in f if len(line) != 0]
        return lst

train_songs = read_file_names_from_list('train_list.txt')
validation_songs = read_file_names_from_list('validation_list.txt')
test_songs = read_file_names_from_list('test_list.txt')

preprocessed_dir = Path('./musdb18hq-processed')
preprocessed_dir.mkdir(exist_ok=True)

for songlist, partname in ((train_songs, 'train'), (validation_songs, 'validation'), (test_songs, 'test')):
    this_dir = preprocessed_dir / partname
    this_dir.mkdir(exist_ok=True)
    
    for i, songpath in enumerate(songlist):
        # This code is borrowed from Keras, available under an Apache 2.0 license:
        # https://github.com/keras-team/keras/blob/68f9af408a1734704746f7e6fa9cfede0d6879d8/keras/utils/audio_dataset.py#L392-L410
        wav_file = tf.io.read_file(songpath)
        audio, sample_rate = tf.audio.decode_wav(wav_file)
        if sample_rate != 22050:
            audio = tfio.audio.resample(audio, tf.cast(sample_rate, tf.int64), tf.cast(22050, tf.int64))
        sample_count = audio.shape[0]
        channel_count = audio.shape[1]
        
        for j in range((sample_count + 66149) // 66150):
            segment = None
            if j == 0:
                segment = tf.concat((tf.zeros((9280, channel_count)), audio[:66150]), axis=0)
            elif (j + 1) * 66150 > sample_count:
                segment = audio[66150*j - 9280:]
                segment = tf.concat((segment, tf.zeros((66150 + 9280 - segment.shape[0], channel_count))), axis=0)
            else:
                segment = audio[66150*j - 9280 : 66150*(j+1)]
            
            this_file_path = this_dir / f"{i:03}-{j:05}.wav"
            
            encoded_part = tf.audio.encode_wav(segment, 22050)
            
            tf.io.write_file(str(this_file_path), encoded_part)
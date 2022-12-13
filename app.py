import streamlit as st
import pandas as pd
import os
from pydub import AudioSegment
import tensorflow as tf
import keras
from tensorflow.keras.models import Model
import tensorflow_io as tfio
import numpy as np
st.title("Hello!")

new_model2 = keras.models.load_model('tf2model.h5')


def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


def preprocess(file_path, label):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


def predict_func(input_wav):
    input_wav_pre, label1 = preprocess(input_wav, 0.0)
    input_wav_pre = np.expand_dims(input_wav_pre, axis=0)
    pre_input = new_model2.predict(input_wav_pre)
    pre_input = [1 if prediction > 0.5 else 0 for prediction in pre_input]
    return pre_input


new_model2.compile('Adam', loss='BinaryCrossentropy', metrics=[
                   tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

crc = os.getcwd()


sound = st.file_uploader(
    "Sound File", type=["wav", "mp3"], accept_multiple_files=False)


if sound is not None:
    with open("target.wav", "wb") as f:
        f.write(sound.getbuffer())
    if sound.type == "audio/mpeg":
        src = os.path.join(crc, sound.name)
        lenght = len(src)
        dst = os.path.join(crc, src[:lenght-4] + ".wav")
        wavSound = AudioSegment.from_mp3(src)
        wavSound.export(dst, format="wav")

if sound:
    st.write("Type : ", sound.type)


# process button
Start = st.button("Start Prediction")

if Start:

    a = predict_func("target.wav")
    st.write(a)
    Start = False

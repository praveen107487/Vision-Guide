import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import os
from PIL import Image
import pickle

# Configuration
max_length = 50
vocab_size = 10000
embedding_dim = 256
units = 512
model_path = 'image_caption_model.h5'
tokenizer_path = 'tokenizer.pickle'

# Load the trained model and tokenizer
def load_model(model_path, tokenizer_path):
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

# Preprocess image for inference
def preprocess_image(image_path):
    model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    img = Image.open(image_path).resize((299, 299))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    feature = model.predict(img)
    return feature.flatten()

# Generate caption
def generate_caption(image_path, model, tokenizer, max_length):
    image_feature = preprocess_image(image_path)
    image_feature = np.expand_dims(image_feature, axis=0)
    
    # Start with start token
    start_token = tokenizer.texts_to_sequences(['<start>'])
    if not start_token:
        start_token = [[1]]  # Assuming 1 is start
    generated = start_token[0]
    
    for _ in range(max_length - 1):
        sequence = pad_sequences([generated], maxlen=max_length-1, padding='post')
        predictions = model([image_feature, sequence])
        next_token = np.argmax(predictions[0, -1, :])
        generated.append(next_token)
        if next_token == tokenizer.word_index.get('<end>', 0):
            break
    
    caption = tokenizer.sequences_to_texts([generated])[0]
    return caption

# Main function
def use_model(image_path):
    model, tokenizer = load_model(model_path, tokenizer_path)
    caption = generate_caption(image_path, model, tokenizer, max_length)
    print(f"Caption: {caption}")
    return caption

# Example
# use_model('path/to/image.jpg')

# Note: Ensure the model is trained and saved.

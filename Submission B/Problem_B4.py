# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') and logs.get('val_accuracy') >= 0.92:
            print("\nVal_Akurasi telah mencapai 92%, hentikan pelatihan!")
            self.model.stop_training = True

def pad_sequences_custom(tokenizer, sequences, max_length, trunc_type, padding_type):
    padded_sequences = []
    for sequence in sequences:
        padded_sequence = pad_sequences(
            [tokenizer.texts_to_sequences([sequence])[0]],
            maxlen=max_length,
            truncating=trunc_type,
            padding=padding_type
        )[0]
        padded_sequences.append(padded_sequence)
    return np.array(padded_sequences)

def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"
    
    train_sentence, test_sentence, train_labels, test_labels = train_test_split(
        bbc.text,
        bbc.category,
        train_size=training_portion,
        shuffle=False
    )

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentence)

    train_padded = pad_sequences_custom(tokenizer, train_sentence, max_length, trunc_type, padding_type)
    test_padded = pad_sequences_custom(tokenizer, test_sentence, max_length, trunc_type, padding_type)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(bbc.category)

    train_labels_final = np.array(label_tokenizer.texts_to_sequences(train_labels))
    test_labels_final = np.array(label_tokenizer.texts_to_sequences(test_labels))


    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    epoch=100
    callback=MyCallback()

    history = model.fit(train_padded, train_labels_final, epochs=epoch, validation_data=(test_padded, test_labels_final), callbacks=[callback])

    test_loss, test_accuracy = model.evaluate(test_padded, test_labels_final)
    
    if history.history['accuracy'][-1] > 0.91 and test_accuracy > 0.91:
        print("Desired accuracy criteria met!")
    else:
        print("Desired accuracy criteria not met. Please adjust your model.")

    print(f' Accuracy: {test_accuracy * 100:.2f}%')

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")

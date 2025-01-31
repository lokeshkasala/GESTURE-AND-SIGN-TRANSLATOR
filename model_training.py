from function import *
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

# Define constants
DATA_PATH = 'D:\\mini prog\\test6\\MP_Data'  # Update with your data path
actions = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','Q','R','S','T','U','V','W','X','Y','Z'])
  # Update with your action labels
no_sequences = 30
sequence_length = 30
label_map = {label: num for num, label in enumerate(actions)}

# Initialize variables
sequences, labels = [], []
inconsistent_count = 0
expected_frame_shape = (63,)  # Expected shape of each frame

# Load and check data
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            try:
                res = np.load(res_path, allow_pickle=True)
                if res.shape != expected_frame_shape:
                    print(f"Inconsistent shape for {action}, sequence {sequence}, frame {frame_num}: {res.shape}")
                    inconsistent_count += 1
                    continue
                window.append(res)
            except Exception as e:
                print(f"Error loading {res_path}: {e}")
                inconsistent_count += 1
                continue
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])
        else:
            print(f"Inconsistent sequence length for {action}, sequence {sequence}: {len(window)} frames (expected {sequence_length})")
            inconsistent_count += 1

print(f"Total inconsistencies found: {inconsistent_count}")

# Proceed with model training only if there are no inconsistencies
if inconsistent_count == 0:
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    # Define the model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, expected_frame_shape[0])))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    # Compile the model
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Train the model
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    model.fit(X_train, y_train, epochs=300, callbacks=[tb_callback])

    # Print model summary
    model.summary()

    # Save the model
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save('model.h5')
else:
    print("Inconsistencies found in the data. Please review and fix the dataset before proceeding.")

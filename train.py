import sys
import csv
import pickle
import matplotlib as plt
import numpy as np
from keras import layers, models
from obspy.signal.filter import bandpass
import obspy
import os
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from scipy.fft import fft, fftfreq

"""
NEXT:
- Extract features from data                       DONE
- Figure out window stuff                          DONE
- Extend model to also predict time of arrival     DONE
"""
WINDOW_NUM = 2153
WINDOW_LENGTH = 24
SAMPLING_RATE = 6.625
SAMPLES_PER_WINDOW = int(WINDOW_LENGTH * SAMPLING_RATE) # WINDOW_LENGTH * SAMPLE RATE
TEMP_FEATURES = 6


def energy_ratio(signal):
    low_freq_band = (0.1, 1.0)
    high_freq_band = (2.0, 3.0)

    n = len(signal)
    fft_result = np.fft.fft(signal)

    freqs = np.fft.fftfreq(n, d=1/SAMPLING_RATE)

    low_band_energy = np.sum(np.abs(fft_result[(freqs >= low_freq_band[0]) & (freqs <= low_freq_band[1])])**2)
    high_band_energy = np.sum(np.abs(fft_result[(freqs >= high_freq_band[0]) & (freqs <= high_freq_band[1])])**2)

    total_energy = np.sum(np.abs(fft_result)**2)

    if total_energy == 0:  # To avoid division by zero
        return 0.0
    
    energy_ratio = low_band_energy / total_energy 

    return energy_ratio


def calculate_entropy(window):
    hist, _ = np.histogram(window, bins='auto', density=True)
    
    # Remove zeros from the histogram to avoid log(0) issues
    hist = hist[hist > 0]

    return entropy(hist)


def get_dominant_freq(window):
    yf = fft(window)
    xf = fftfreq(SAMPLES_PER_WINDOW, 1 / SAMPLING_RATE)  # Frequency bins

    power_spectrum = np.abs(yf) ** 2
    peak_index = np.argmax(power_spectrum)
    
    return xf[peak_index]


def extract_data(file_path, window_num, num_sample):
    data = dict()

    for file in os.listdir(file_path):
        if file[-3:] == "csv":
            continue

        st = obspy.read(file_path + "\\" + file)

        tr = st[0]
        tr.data = bandpass(tr.data, freqmin=0.1, freqmax=5, df=tr.stats.sampling_rate)

        windows = tr[-window_num * num_sample:].reshape((window_num, num_sample))  # Info at the end generally more important


        # EXTRACTING NON-TEMPORAL FEATURES
        # non_temp_features = [energy_ratio(tr.data).item()] #, np.angle(np.fft.fft(tr.data)).item()] # Energy ratio, Phase info, 

        # EXTRACTING TEMPORAL FEATURES
        features = [] # Format: tuple of features for each window: (Entropy, RMS Amplitude, Peak Amplitude, Dominant frequency)

        for window in windows:
            features.append((calculate_entropy(window), np.sqrt(np.mean(window**2)), np.max(window), get_dominant_freq(window)))


        data[file[-15:-6]] = (windows, features) #, non_temp_features)   # Extracting evid number
    
    return data

def extract_labels(file_path):
    mq_type_dict = {"impact_mq": 0, "deep_mq": 1, "shallow_mq": 1}

    labels = [] # Label format: evid  time_rel(sec)   mq_type
    with open(file_path + "\\catalogs\\apollo12_catalog_GradeA_final.csv", "r") as label_file:
        rows = csv.reader(label_file)
        next(rows)
        for row in rows:
            labels.append([row[3], row[2], mq_type_dict[row[4]]])

    return labels

def create_model():
    # Define input for velocity data (sequential)
    seq_input = layers.Input(shape=(WINDOW_NUM, SAMPLES_PER_WINDOW), name="sequential_input")
    
    # Pass the sequential data through RNN layers
    x = layers.SimpleRNN(159, return_sequences=True)(seq_input)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.GRU(159, return_sequences=True)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.LSTM(159, return_sequences=True)(x)
    
    # Temporal features input
    temp_feat = layers.Input(shape=(WINDOW_NUM, TEMP_FEATURES), name="features_input")
    # temp_reshaped = layers.Reshape((1, TEMP_FEATURES))(temp_feat)

    # RNN layers for temporal features
    temporal_rnn_output = layers.SimpleRNN(4, return_sequences=True)(temp_feat)
    temporal_rnn_output = layers.LayerNormalization()(temporal_rnn_output)
    temporal_rnn_output = layers.Dropout(0.5)(temporal_rnn_output)

    # Combine the RNN output with the feature input
    combined = layers.concatenate([x, temporal_rnn_output])
    
    # Pass the combined inputs through dense layers
    z = layers.Dense(64, activation='relu')(combined)
    classification_output = layers.Flatten()(z)
    classification_output = layers.Dense(2, activation='sigmoid')(classification_output)
    
    # Create the model
    model = models.Model(inputs=[x, temporal_rnn_output], outputs=[classification_output])

    # Compile the model with different loss functions for each output
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    return model


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("Invalid Input")

    # path = "Python\space_apps_2024_seismic_detection\data\lunar\training"

    data = extract_data(sys.argv[1]) # Data Format: Dict keys: evid, Dict values: (Vel windows, temp features)
                                            #  Label Format: (evid, time_rel(sec), mq_type)

    labels = extract_labels(sys.argv[1], WINDOW_NUM, SAMPLES_PER_WINDOW)

    vel_array = np.array([data[label[0]][0] for label in labels])
    feature_array = np.array([data[label[0]][1] for label in labels])
    label_array = np.array([label[1:] for label in labels])

    # with open("Python\\NASA-Space-Apps\\vel_array.pickle", "rb") as pf:
    #     vel_array = pickle.load(pf)
    # with open("Python\\NASA-Space-Apps\\feature_array.pickle", "rb") as pf:
    #     feature_array = pickle.load(pf)
    # with open("Python\\NASA-Space-Apps\\label_array.pickle", "rb") as pf:
    #     label_array = pickle.load(pf)

    X_velocity_train, X_velocity_test, X_temporal_train, X_temporal_test, y_classification_train, y_classification_test= train_test_split(
                                                                      vel_array, 
                                                                      feature_array, 
                                                                      label_array,  
                                                                      train_size=0.75, stratify=label_array)

    model = create_model()

    model.summary()

    history = model.fit(
        [X_velocity_train, X_temporal_train], 
        y_classification_train,

    validation_data=(
        [X_velocity_test, X_temporal_test],
        [y_classification_test]
    ),
    epochs=50,  
    batch_size=32,
    verbose=1 
)

    results = model.evaluate(
    [X_velocity_test, X_temporal_test],
    [y_classification_test],
    verbose = 1
)

    if len(sys.argv) == 3:
        model.save(sys.argv[2])
        print(f"Model saved to {sys.argv[2]}.")


if __name__ == "__main__":
    main()

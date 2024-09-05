import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import pandas as pd

import joblib
import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.preprocessing import ICA
from mne.io import read_raw_gdf, read_raw_brainvision
from mne.time_frequency import tfr_morlet

from mne_icalabel import label_components

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import pickle

# =============================================================================
# Loading the data and preprocessing to get the epochs and labels
# =============================================================================

vhdr_file_path = 'D:\\Recorder-Data\\Daniela-Shay\\SUB03\\SUB03_MI_Emb.vhdr'
raw = read_raw_brainvision(vhdr_file_path, preload=True)


#gdf_file_path= 'C:\\Users\\shaye\\PIC\\PIC_data_robot\\condition1_first_p\\all_subjects_online_s3\\subject_4_online.gdf'
#raw = read_raw_gdf(gdf_file_path, preload=True)
print(raw.info)
# Get the names of all channels
all_channels = raw.ch_names

# Identify the names of the last three channels
#accelerometer_channels = all_channels[-3:]
accelerometer_channels = all_channels[-6:] 
# Drop the accelerometer channels
raw.drop_channels(accelerometer_channels)

'''
new_names = {
    'Channel 1': 'Fp1', 'Channel 2': 'Fz', 'Channel 3': 'F3', 'Channel 4': 'F7', 'Channel 5': 'FT9',
    'Channel 6': 'FC5', 'Channel 7': 'FC1', 'Channel 8': 'C3', 'Channel 9': 'T7', 'Channel 10': 'TP9',
    'Channel 11': 'CP5', 'Channel 12': 'CP1', 'Channel 13': 'Pz','Channel 14': 'P3', 'Channel 15': 'P7',
    'Channel 16': 'O1', 'Channel 17': 'Oz','Channel 18': 'O2', 'Channel 19': 'P4', 'Channel 20': 'P8',
    'Channel 21': 'TP10', 'Channel 22': 'CP6', 'Channel 23': 'CP2', 'Channel 24': 'Cz', 'Channel 25': 'C4',
    'Channel 26': 'T8', 'Channel 27': 'FT10', 'Channel 28': 'FC6', 'Channel 29': 'FC2', 'Channel 30': 'F4',
    'Channel 31': 'F8', 'Channel 32': 'Fp2'}


# Rename channels
raw.rename_channels(new_names)
'''
# Use the standard_1020 montage
montage = make_standard_montage('standard_1020')

# Apply the montage to your raw data
raw.set_montage(montage)

#raw.plot_sensors(kind='topomap', show_names=True)



####### Preperation for ICA ########

raw.filter(4.0, 30.0, fir_design='firwin')
raw.set_eeg_reference('average', projection=True)

####### ICA ########
'''
# Define ICA parameters
n_components = 32  # The number of components to decompose the signals into; adjust based on your data
method = 'fastica'  # Commonly used algorithm for ICA
ica = ICA(n_components=n_components, method=method, random_state=97) # Initialize ICA object
ica.fit(raw) # Fit ICA to the pre-filtered raw data 
#ica.plot_components() # Plot the components to visually identify artifacts
#ica.plot_sources(raw)

# Use mne_icalabel to automatically label the components
labels_dict = label_components(raw, ica, method='iclabel')
print(labels_dict)

# To access the probabilities and categories for each component
labels= labels_dict['labels']
probabilities = labels_dict['y_pred_proba']

# Define a threshold for excluding components
threshold = 0.9
# From the labels, decide which component categories to exclude

to_exclude = []
for index, (label, prob) in enumerate(zip(labels, probabilities)):
    # Check for 'eye blink' or 'muscle artifact' labels and probability threshold
    if label in ['eye blink', 'muscle artifact', 'line noise'] and prob > threshold:
        to_exclude.append(index)
        
ica.exclude = to_exclude

# You can now apply the ICA solution to the raw data, effectively removing the artifacts
raw_clean = ica.apply(raw)
# raw_clean.plot()
# Exclude identified artifact components
# raw_cleaned = ica.apply(raw.copy(), exclude=ica.exclude)

# Plot the cleaned sources to see the difference
#ica.plot_sources(raw_cleaned)

'''

####### Epoching #######
raw_clean=raw
# Extract events without specifying an event_id mapping
events, event_id_mapping = events_from_annotations(raw_clean)

event_times = events[:, 0] / raw.info['sfreq']

# Manually filter for events 10 and 11
events_of_interest = events[(events[:, 2] == 7) | (events[:, 2] == 8)]

# Specify the EEG channels and exclude bad ones
picks = pick_types(raw_clean.info, meg=False, eeg=True, stim=False, eog=False)

# Epoch extraction
tmin, tmax = -1.5, 4.0  # Define your time window
epochs = Epochs(raw_clean, 
                events_of_interest, 
                event_id={'left': 7, 'right': 8}, 
                tmin=tmin, 
                tmax=tmax,
                baseline=(-1.5, 0.0),
                proj=True, 
                preload=True, 
                picks=picks)

  
# Copy and crop epochs for the training set
epochs_train = epochs.copy().crop(tmin=0.5, tmax=3.0)

# Extract labels for the epochs
labels = epochs.events[:, -1]


# =============================================================================
# ERSP
# =============================================================================


# Assuming 'epochs' is an MNE Epochs object containing your epoch data
frequencies = np.arange(4, 30, 1)  # Define the frequency range
n_cycles = frequencies / 2  # Define the number of cycles for the wavelet transform

# Separate the epochs by condition
epochs_left = epochs['left']
epochs_right = epochs['right']

# Compute the power for both conditions
power_left = tfr_morlet(epochs_left, freqs=frequencies, n_cycles=n_cycles, return_itc=False, average=True)
power_right = tfr_morlet(epochs_right, freqs=frequencies, n_cycles=n_cycles, return_itc=False, average=True)

# Apply baseline correction
baseline = (None, 0)  # Define the baseline interval
power_left.apply_baseline(baseline=baseline, mode='percent')
power_right.apply_baseline(baseline=baseline, mode='percent')

# Extract the TFR data (power) from the TFR objects for the left and right conditions
# These are 3D arrays: (n_epochs, n_channels, n_freqs, n_times)
ersp_left = power_left.data  # Shape: (n_channels, n_freqs, n_times)
ersp_right = power_right.data  # Shape: (n_channels, n_freqs, n_times)

# Select the channel of interest, e.g., 'C3'
channel_of_interest = 'C3'

# Find the index of the channel of interest for the left condition
channel_index_left = power_left.ch_names.index(channel_of_interest)
# Find the index of the channel of interest for the right condition
channel_index_right = power_right.ch_names.index(channel_of_interest)

'''
# Plot the ERSP for the selected channel for both conditions
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

vmin, vmax = -0.6, 0.6

# ERSP for the left condition
im_left = axes[0].imshow(power_left.data[channel_index_left], aspect='auto', origin='lower', 
                         extent=[power_left.times[0], power_left.times[-1], frequencies[0], frequencies[-1]], 
                         cmap='viridis', vmin=vmin, vmax=vmax)
axes[0].set_title(f'{channel_of_interest} - Left Condition')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Frequency (Hz)')

# ERSP for the right condition
im_right = axes[1].imshow(power_right.data[channel_index_right], aspect='auto', origin='lower', 
                          extent=[power_right.times[0], power_right.times[-1], frequencies[0], frequencies[-1]], 
                          cmap='viridis', vmin=vmin, vmax=vmax)
axes[1].set_title(f'{channel_of_interest} - Right Condition')
axes[1].set_xlabel('Time (s)')

# Colorbar
fig.colorbar(im_left, ax=axes[0])
fig.colorbar(im_right, ax=axes[1])

plt.tight_layout()
plt.show()
'''

# =============================================================================
# Fisher Score
# =============================================================================

def calculate_fisher_scores(ersp_left, ersp_right, epochs, frequency_bins, timerange, include_electrodes):
    # Find time indices that correspond to the timerange
    time_inds = (epochs.times >= timerange[0]) & (epochs.times <= timerange[1])
    
    # Get indices of electrodes to include (i.e.,  in include_electrodes)
    include_indices = []
    for i, elec in enumerate(epochs.ch_names):
        if elec in include_electrodes:
            include_indices.append(i)
    
    # Initialize an array to hold the Fisher scores for included electrodes only
    f_scores = np.zeros((len(include_indices), len(frequency_bins)))
    
    # Iterate over included electrodes
    for n_idx, n in enumerate(include_indices):  # Use n_idx for indexing in f_scores
        # Iterate over frequency bins
        for k in range(len(frequency_bins)):
            # Calculate mean and standard deviation for the left and right conditions
            mean_class1 = ersp_left[n, k, time_inds].mean()
            mean_class2 = ersp_right[n, k, time_inds].mean()
            std_class1 = ersp_left[n, k, time_inds].std()
            std_class2 = ersp_right[n, k, time_inds].std()

            # Compute the Fisher score
            FS = abs(mean_class1 - mean_class2) / np.sqrt(std_class1**2 + std_class2**2)

            # Store the Fisher score in the corresponding place
            f_scores[n_idx, k] = FS
    
    # Return the Fisher scores, including only those for the included electrodes
    return f_scores, include_indices


#exclude_electrodes = ['Fp1', 'Fp2', 'O1', 'O2', 'Oz', 'F7', 'F3', 'Fz', 'F4', 'F8', 
#                      'FT9', 'FT10', 'TP9', 'TP10']
include_electrodes = ['FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1','CP2', 'CP6', 'P7',
                    'P3', 'Pz', 'P4', 'P8']
#include_electrodes= ['C3', 'C4', 'CP5', 'P3', 'P8', 'CP6', 'P7', 'Pz', 'FC6', 'CP2', 'P4']

# Define the timerange as from 0 to the end of the epochs
timerange = (0, epochs.times[-1])
# Define the frequency bins (assuming you are using the same bins as for ERSP calculation)
frequency_bins = frequencies

# Compute the Fisher scores
fisher_scores, included_electrode_indices = calculate_fisher_scores(ersp_left, ersp_right, epochs, frequency_bins, timerange, include_electrodes)



# Filter the electrode names to include only those used in Fisher score calculation
used_electrode_names = [epochs.ch_names[i] for i in included_electrode_indices]

# Create the figure and axis for the heatmap
fig, ax = plt.subplots(figsize=(12, 8))

# Plotting the heatmap
cax = ax.imshow(fisher_scores, interpolation='nearest', cmap='jet', aspect='auto')

# Add color bar
cbar = fig.colorbar(cax, ax=ax, format='%.2f')
cbar.set_label('Fisher Score', rotation=270, labelpad=15)

# Set the tick labels to show the used electrodes and frequency bins
ax.set_xticks(np.arange(len(frequency_bins)))
ax.set_yticks(np.arange(len(used_electrode_names)))
ax.set_xticklabels([f'{freq:.1f}' for freq in frequency_bins])
ax.set_yticklabels(used_electrode_names)

# Rotate the tick labels for the x-axis for better readability
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Set axis labels
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Electrodes')

# Add a title to the heatmap
ax.set_title('Fisher Scores by Electrode and Frequency (Used Electrodes)')

plt.tight_layout()
plt.show()



# =============================================================================
# Feature Selelction
# =============================================================================

def select_top_features(fisher_scores, top_n_features, used_electrode_names, frequency_bins):
    # Flatten the Fisher scores array
    flat_scores = fisher_scores.flatten()
    
    # Get the sorted indices of the Fisher scores, in descending order
    sorted_indices = np.argsort(flat_scores)[::-1]
    
    # Select the top N features based on the sorted indices
    top_indices = sorted_indices[:top_n_features]
    
    # Get the corresponding electrode and frequency bin indices
    top_electrode_indices, top_frequency_indices = np.unravel_index(top_indices, fisher_scores.shape)
    
    # Get the corresponding electrode names and frequency bins
    top_features_info = [{
    'electrode': used_electrode_names[e_idx], 
    'frequency_bin': (frequency_bins[f_idx], frequency_bins[f_idx] + 1),  # Example adjustment
    'fisher_score': flat_scores[idx]
    } for e_idx, f_idx, idx in zip(top_electrode_indices, top_frequency_indices, top_indices)]

    
    return top_features_info

# Example usage
top_n_features = 40  # number of top features
frequency_bins = np.arange(4, 30, 1)  # assuming this is your array of frequency bins

top_features = select_top_features(fisher_scores, top_n_features, used_electrode_names, frequency_bins)
print(top_features)
'''
# =============================================================================
# Merging adjuacent bins from the same electrode into one feature
# =============================================================================

def merge_adjacent_bins(features):
    def merge_bins(bins):
        merged_bins = []
        current_bin = bins[0]
        for next_bin in bins[1:]:
            if current_bin[1] == next_bin[0]:  # Adjacent bins
                current_bin = (current_bin[0], next_bin[1])  # Merge bins
            else:
                merged_bins.append(current_bin)
                current_bin = next_bin
        merged_bins.append(current_bin)
        return merged_bins

    # Initial merging
    merged_features = {}
    for feature in features:
        electrode = feature['electrode']
        low_freq, high_freq = feature['frequency_bin']
        if electrode not in merged_features:
            merged_features[electrode] = [(low_freq, high_freq)]
        else:
            merged = False
            for i, (lf, hf) in enumerate(merged_features[electrode]):
                if hf == low_freq:  # Adjacent bins
                    merged_features[electrode][i] = (lf, high_freq)  # Merge bins
                    merged = True
                    break
                elif lf == high_freq:  # Adjacent bins
                    merged_features[electrode][i] = (low_freq, hf)  # Merge bins
                    merged = True
                    break
            if not merged:
                merged_features[electrode].append((low_freq, high_freq))

    # Perform additional pass to ensure all adjacent bins are merged
    for electrode in merged_features:
        merged_features[electrode].sort()  # Sort bins to ensure adjacency can be checked
        merged_features[electrode] = merge_bins(merged_features[electrode])

    # Flatten the dictionary to a list of features
    flattened_features = []
    for electrode, bins in merged_features.items():
        for (low_freq, high_freq) in bins:
            flattened_features.append({'electrode': electrode, 'frequency_bin': (low_freq, high_freq)})

    return flattened_features


merged_features = merge_adjacent_bins(top_features)

print(merged_features)




for feature in top_features:
    print(f"Electrode: {feature['electrode']}, Frequency Bin: {feature['frequency_bin']}, Fisher Score: {feature['fisher_score']}")
'''
######### Saving top features info ###########

# Save the selected features using pickle
feature_num = len(top_features)
with open(f'selected_features_SUB03_{feature_num}f.pkl', 'wb') as f:
    pickle.dump(top_features, f)
    print(f'Features info saves with {feature_num} features')

# =============================================================================
# Finding the best number of features to use
# =============================================================================
'''
# Calculate the TFR for all epochs only once
tfr = tfr_morlet(epochs, freqs=frequencies, n_cycles=n_cycles, return_itc=False, average=False)
tfr.apply_baseline(baseline=baseline, mode='percent')

def train_lda(features_subset, tfr):
    time_indices = tfr.times > 0  # Only times after 0
    X = []
    y = []

    for i in range(len(tfr)):
        feature_vector = []
        for feature in features_subset:
            electrode = feature['electrode']
            low_freq, high_freq = feature['frequency_bin']  # Ensure this is a tuple
            elec_idx = tfr.ch_names.index(electrode)
            freq_idxs = np.where((frequencies >= low_freq) & (frequencies < high_freq))[0]
            power = tfr.data[i, elec_idx, freq_idxs, :][:, time_indices].mean()
            feature_vector.append(power)

        X.append(feature_vector)
        y.append(1 if epochs.events[i, 2] == 11 else 0)

    X = np.array(X)
    y = np.array(y)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)  # Fit LDA model
    return lda.score(X, y)


# Select features with Fisher score above 0.5
all_features = []
for elec_idx, elec in enumerate(used_electrode_names):
    for freq_idx, freq in enumerate(frequency_bins):
        score = fisher_scores[elec_idx, freq_idx]
        if score > 0.5:
            all_features.append({
                'electrode': elec,
                'frequency_bin': (freq, freq + 1),  # Assuming each bin is freq to freq+1
                'fisher_score': score
            })

# Sort features by descending Fisher score
sorted_features = sorted(all_features, key=lambda x: x['fisher_score'], reverse=True)

# List to store the number of features and corresponding scores
num_features = []
scores = []

# Loop over subsets of sorted features
for i in range(1, len(sorted_features) + 1):
    features_subset = sorted_features[:i]
    score = train_lda(features_subset, tfr)
    num_features.append(i)
    scores.append(score)

# Plotting the relationship
plt.figure(figsize=(10, 5))
plt.plot(num_features, scores, marker='o')
plt.title('LDA Performance vs Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Classification Accuracy')
plt.grid(True)
plt.show()


'''
# =============================================================================
# Feature vector creation
# =============================================================================

tfr = tfr_morlet(epochs, freqs=frequencies, n_cycles=n_cycles, return_itc=False, average=False)
tfr.apply_baseline(baseline=baseline, mode='percent')
time_indices = epochs.times > 0  # Only times after 0

X = []  # Initialize feature vector X
y = []  # Initialize label vector y

for i in range(len(epochs)):
    feature_vector = []  # Initialize the feature vector for the current epoch
    
    for feature in top_features:  # Ensure this matches the loaded structure
        electrode = feature['electrode']
        low_freq, high_freq = feature['frequency_bin']  # Now a tuple
        
        elec_idx = tfr.ch_names.index(electrode)  # Get the index of the electrode
        
        # Determine indices within the frequency bin
        freq_idxs = np.where((frequencies >= low_freq) & (frequencies <= high_freq))[0]
        
        # Extract and average the power for the specified frequency bin and electrode, for the time of interest
        power = tfr.data[i, elec_idx, freq_idxs, :][:, time_indices].mean()
        feature_vector.append(power)
    
    X.append(feature_vector)
    # 10 corresponds to left and 11 to right
    y.append(1 if epochs.events[i, 2] == 8 else 0)  

X = np.array(X)
y = np.array(y)


# =============================================================================
# Trainng LDA
# =============================================================================

lda = LinearDiscriminantAnalysis()
lda.fit(X, y) #Fit LDA model

print('Score: ',lda.score(X, y))

# save the model to disk
filename = f'lda_model_SUB03_{feature_num}f.sav'
joblib.dump(lda, filename)
print('model saved!')


# Save the model's coefficients and intercept to disk
coefficients = lda.coef_
intercept = lda.intercept_
classes_=lda.classes_


print(classes_)
# Using numpy to save the coefficients and intercept
np.savez(f'lda_model_SUB03_{feature_num}f_param.npz', coefficients=coefficients, intercept=intercept, classes_=classes_)
print('Model parameters saved!')

'''
# =============================================================================
# Fisher score topomaps
# =============================================================================

all_eeg_channels = all_channels[: 32]
include_electrodes = ['FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1','CP2', 'CP6', 'P7',
                    'P3', 'Pz', 'P4', 'P8']
noise_electrode = [elect for elect in all_eeg_channels if elect not in include_electrodes]
dummy_scores = np.zeros(len(frequency_bins))

for noise in noise_electrode:
    fisher_scores = np.vstack([fisher_scores, dummy_scores])
    used_electrode_names.append(noise)
    
# Create an Info object needed for topomap plotting
info = mne.create_info(ch_names=used_electrode_names, sfreq=raw_clean.info['sfreq'], ch_types='eeg')
info.set_montage(montage)

# Calculate global min and max across all Fisher scores
vmin, vmax = np.min(fisher_scores), np.max(fisher_scores)
norm = colors.Normalize(vmin=vmin, vmax=vmax)  # Create a Normalize object for the color scale


# Number of rows and columns for the subplot
n_rows = 5  # Adjust based on your preference
n_cols = 6  # Adjust based on your preference

# Create a figure
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 15))

# Ensure axes is a flat array for easy iteration
axes = axes.ravel()

for i, freq_bin in enumerate(frequency_bins):
    data = fisher_scores[:, i]
    evoked = mne.EvokedArray(data[:, np.newaxis], info)

    # When vmin and vmax are not accepted, use norm instead
    mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, axes=axes[i], show=False,
                         contours=0, cmap='jet', vlim=(vmin,vmax))
    
    axes[i].set_title(f'Bin {freq_bin} Hz')

for j in range(len(frequency_bins), len(axes)):
    axes[j].axis('off')

# Create an axis for the colorbar
cbar_ax = fig.add_axes([0.7, 0.05, 0.25, 0.02])  # Adjust these values as needed

# Create a colormap object based on your topomap's colormap
cmap = mpl.cm.jet
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# Create a ScalarMappable object with the normalization and colormap
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # You can safely ignore this line; it's a quirk of Matplotlib's API

# Create the colorbar
fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar_ax.set_ylabel('Fisher Score', labelpad=80, rotation=0, horizontalalignment='left')


plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.2)

plt.show()

'''








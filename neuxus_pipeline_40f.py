
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from neuxus.nodes import *
import joblib
import pickle

# Load the LDA parameters
params = np.load('lda_model_SUB03_40f_param.npz')
lda_recreated = LinearDiscriminantAnalysis()
lda_recreated.coef_ = params['coefficients']
lda_recreated.intercept_ = params['intercept']
lda_recreated.classes_ = params['classes_']
joblib.dump(lda_recreated, 'lda_recreated_model.sav')

# Generate stimulation
#config_file = 'protocol_serial.xml'
#markers = stimulator.Stimulator(config_file)
#lsl_send_markers = io.LslSend(markers.output, 'openvibeMarkers', 'Markers', 'int32')


with open('selected_features_SUB03_40f.pkl', 'rb') as f:
    top_features_info = pickle.load(f)

lsl_signal = io.RdaReceive(rdaport=51244, host="10.9.2.100")
#lsl_signal = io.LslReceive('name', 'openvibeSignal', data_type='signal', sync='network')

chan_names = []

# Feature 1
top_f = top_features_info[0]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans1 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter1 = filter.ButterFilter(chans1.output, lowcut, highcut)
time_epoch1 = epoching.TimeBasedEpoching(butter_filter1.output, 1, 0.25)
square_epoch1 = function.ApplyFunction(time_epoch1.output, lambda x: x**2)
average_epoch1 = epoch_function.UnivariateStat(square_epoch1.output, 'mean')
log_power1 = function.ApplyFunction(average_epoch1.output, lambda x: np.log1p(x))

# Feature 2
top_f = top_features_info[1]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans2 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter2 = filter.ButterFilter(chans2.output, lowcut, highcut)
time_epoch2 = epoching.TimeBasedEpoching(butter_filter2.output, 1, 0.25)
square_epoch2 = function.ApplyFunction(time_epoch2.output, lambda x: x**2)
average_epoch2 = epoch_function.UnivariateStat(square_epoch2.output, 'mean')
log_power2 = function.ApplyFunction(average_epoch2.output, lambda x: np.log1p(x))
channel_updater1 = select.ChannelUpdater(log_power1.output, log_power2.output)

# Feature 3
top_f = top_features_info[2]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans3 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter3 = filter.ButterFilter(chans3.output, lowcut, highcut)
time_epoch3 = epoching.TimeBasedEpoching(butter_filter3.output, 1, 0.25)
square_epoch3 = function.ApplyFunction(time_epoch3.output, lambda x: x**2)
average_epoch3 = epoch_function.UnivariateStat(square_epoch3.output, 'mean')
log_power3 = function.ApplyFunction(average_epoch3.output, lambda x: np.log1p(x))
channel_updater2 = select.ChannelUpdater(channel_updater1.output, log_power3.output)

# Feature 4
top_f = top_features_info[3]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans4 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter4 = filter.ButterFilter(chans4.output, lowcut, highcut)
time_epoch4 = epoching.TimeBasedEpoching(butter_filter4.output, 1, 0.25)
square_epoch4 = function.ApplyFunction(time_epoch4.output, lambda x: x**2)
average_epoch4 = epoch_function.UnivariateStat(square_epoch4.output, 'mean')
log_power4 = function.ApplyFunction(average_epoch4.output, lambda x: np.log1p(x))
channel_updater3 = select.ChannelUpdater(channel_updater2.output, log_power4.output)

# Feature 5
top_f = top_features_info[4]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans5 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter5 = filter.ButterFilter(chans5.output, lowcut, highcut)
time_epoch5 = epoching.TimeBasedEpoching(butter_filter5.output, 1, 0.25)
square_epoch5 = function.ApplyFunction(time_epoch5.output, lambda x: x**2)
average_epoch5 = epoch_function.UnivariateStat(square_epoch5.output, 'mean')
log_power5 = function.ApplyFunction(average_epoch5.output, lambda x: np.log1p(x))
channel_updater4 = select.ChannelUpdater(channel_updater3.output, log_power5.output)

# Feature 6
top_f = top_features_info[5]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans6 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter6 = filter.ButterFilter(chans6.output, lowcut, highcut)
time_epoch6 = epoching.TimeBasedEpoching(butter_filter6.output, 1, 0.25)
square_epoch6 = function.ApplyFunction(time_epoch6.output, lambda x: x**2)
average_epoch6 = epoch_function.UnivariateStat(square_epoch6.output, 'mean')
log_power6 = function.ApplyFunction(average_epoch6.output, lambda x: np.log1p(x))
channel_updater5 = select.ChannelUpdater(channel_updater4.output, log_power6.output)

# Feature 7
top_f = top_features_info[6]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans7 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter7 = filter.ButterFilter(chans7.output, lowcut, highcut)
time_epoch7 = epoching.TimeBasedEpoching(butter_filter7.output, 1, 0.25)
square_epoch7 = function.ApplyFunction(time_epoch7.output, lambda x: x**2)
average_epoch7 = epoch_function.UnivariateStat(square_epoch7.output, 'mean')
log_power7 = function.ApplyFunction(average_epoch7.output, lambda x: np.log1p(x))
channel_updater6 = select.ChannelUpdater(channel_updater5.output, log_power7.output)

# Feature 8
top_f = top_features_info[7]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans8 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter8 = filter.ButterFilter(chans8.output, lowcut, highcut)
time_epoch8 = epoching.TimeBasedEpoching(butter_filter8.output, 1, 0.25)
square_epoch8 = function.ApplyFunction(time_epoch8.output, lambda x: x**2)
average_epoch8 = epoch_function.UnivariateStat(square_epoch8.output, 'mean')
log_power8 = function.ApplyFunction(average_epoch8.output, lambda x: np.log1p(x))
channel_updater7 = select.ChannelUpdater(channel_updater6.output, log_power8.output)

# Feature 9
top_f = top_features_info[8]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans9 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter9 = filter.ButterFilter(chans9.output, lowcut, highcut)
time_epoch9 = epoching.TimeBasedEpoching(butter_filter9.output, 1, 0.25)
square_epoch9 = function.ApplyFunction(time_epoch9.output, lambda x: x**2)
average_epoch9 = epoch_function.UnivariateStat(square_epoch9.output, 'mean')
log_power9 = function.ApplyFunction(average_epoch9.output, lambda x: np.log1p(x))
channel_updater8 = select.ChannelUpdater(channel_updater7.output, log_power9.output)

# Feature 10
top_f = top_features_info[9]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans10 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter10 = filter.ButterFilter(chans10.output, lowcut, highcut)
time_epoch10 = epoching.TimeBasedEpoching(butter_filter10.output, 1, 0.25)
square_epoch10 = function.ApplyFunction(time_epoch10.output, lambda x: x**2)
average_epoch10 = epoch_function.UnivariateStat(square_epoch10.output, 'mean')
log_power10 = function.ApplyFunction(average_epoch10.output, lambda x: np.log1p(x))
channel_updater9 = select.ChannelUpdater(channel_updater8.output, log_power10.output)

# Feature 11
top_f = top_features_info[10]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans11 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter11 = filter.ButterFilter(chans11.output, lowcut, highcut)
time_epoch11 = epoching.TimeBasedEpoching(butter_filter11.output, 1, 0.25)
square_epoch11 = function.ApplyFunction(time_epoch11.output, lambda x: x**2)
average_epoch11 = epoch_function.UnivariateStat(square_epoch11.output, 'mean')
log_power11 = function.ApplyFunction(average_epoch11.output, lambda x: np.log1p(x))
channel_updater10 = select.ChannelUpdater(channel_updater9.output, log_power11.output)

# Feature 12
top_f = top_features_info[11]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans12 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter12 = filter.ButterFilter(chans12.output, lowcut, highcut)
time_epoch12 = epoching.TimeBasedEpoching(butter_filter12.output, 1, 0.25)
square_epoch12 = function.ApplyFunction(time_epoch12.output, lambda x: x**2)
average_epoch12 = epoch_function.UnivariateStat(square_epoch12.output, 'mean')
log_power12 = function.ApplyFunction(average_epoch12.output, lambda x: np.log1p(x))
channel_updater11 = select.ChannelUpdater(channel_updater10.output, log_power12.output)

# Feature 13
top_f = top_features_info[12]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans13 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter13 = filter.ButterFilter(chans13.output, lowcut, highcut)
time_epoch13 = epoching.TimeBasedEpoching(butter_filter13.output, 1, 0.25)
square_epoch13 = function.ApplyFunction(time_epoch13.output, lambda x: x**2)
average_epoch13 = epoch_function.UnivariateStat(square_epoch13.output, 'mean')
log_power13 = function.ApplyFunction(average_epoch13.output, lambda x: np.log1p(x))
channel_updater12 = select.ChannelUpdater(channel_updater11.output, log_power13.output)

# Feature 14
top_f = top_features_info[13]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans14 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter14 = filter.ButterFilter(chans14.output, lowcut, highcut)
time_epoch14 = epoching.TimeBasedEpoching(butter_filter14.output, 1, 0.25)
square_epoch14 = function.ApplyFunction(time_epoch14.output, lambda x: x**2)
average_epoch14 = epoch_function.UnivariateStat(square_epoch14.output, 'mean')
log_power14 = function.ApplyFunction(average_epoch14.output, lambda x: np.log1p(x))
channel_updater13 = select.ChannelUpdater(channel_updater12.output, log_power14.output)

# Feature 15
top_f = top_features_info[14]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans15 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter15 = filter.ButterFilter(chans15.output, lowcut, highcut)
time_epoch15 = epoching.TimeBasedEpoching(butter_filter15.output, 1, 0.25)
square_epoch15 = function.ApplyFunction(time_epoch15.output, lambda x: x**2)
average_epoch15 = epoch_function.UnivariateStat(square_epoch15.output, 'mean')
log_power15 = function.ApplyFunction(average_epoch15.output, lambda x: np.log1p(x))
channel_updater14 = select.ChannelUpdater(channel_updater13.output, log_power15.output)

# Feature 16
top_f = top_features_info[15]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans16 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter16 = filter.ButterFilter(chans16.output, lowcut, highcut)
time_epoch16 = epoching.TimeBasedEpoching(butter_filter16.output, 1, 0.25)
square_epoch16 = function.ApplyFunction(time_epoch16.output, lambda x: x**2)
average_epoch16 = epoch_function.UnivariateStat(square_epoch16.output, 'mean')
log_power16 = function.ApplyFunction(average_epoch16.output, lambda x: np.log1p(x))
channel_updater15 = select.ChannelUpdater(channel_updater14.output, log_power16.output)

# Feature 17
top_f = top_features_info[16]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans17 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter17 = filter.ButterFilter(chans17.output, lowcut, highcut)
time_epoch17 = epoching.TimeBasedEpoching(butter_filter17.output, 1, 0.25)
square_epoch17 = function.ApplyFunction(time_epoch17.output, lambda x: x**2)
average_epoch17 = epoch_function.UnivariateStat(square_epoch17.output, 'mean')
log_power17 = function.ApplyFunction(average_epoch17.output, lambda x: np.log1p(x))
channel_updater16 = select.ChannelUpdater(channel_updater15.output, log_power17.output)

# Feature 18
top_f = top_features_info[17]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans18 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter18 = filter.ButterFilter(chans18.output, lowcut, highcut)
time_epoch18 = epoching.TimeBasedEpoching(butter_filter18.output, 1, 0.25)
square_epoch18 = function.ApplyFunction(time_epoch18.output, lambda x: x**2)
average_epoch18 = epoch_function.UnivariateStat(square_epoch18.output, 'mean')
log_power18 = function.ApplyFunction(average_epoch18.output, lambda x: np.log1p(x))
channel_updater17 = select.ChannelUpdater(channel_updater16.output, log_power18.output)

# Feature 19
top_f = top_features_info[18]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans19 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter19 = filter.ButterFilter(chans19.output, lowcut, highcut)
time_epoch19 = epoching.TimeBasedEpoching(butter_filter19.output, 1, 0.25)
square_epoch19 = function.ApplyFunction(time_epoch19.output, lambda x: x**2)
average_epoch19 = epoch_function.UnivariateStat(square_epoch19.output, 'mean')
log_power19 = function.ApplyFunction(average_epoch19.output, lambda x: np.log1p(x))
channel_updater18 = select.ChannelUpdater(channel_updater17.output, log_power19.output)

# Feature 20
top_f = top_features_info[19]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans20 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter20 = filter.ButterFilter(chans20.output, lowcut, highcut)
time_epoch20 = epoching.TimeBasedEpoching(butter_filter20.output, 1, 0.25)
square_epoch20 = function.ApplyFunction(time_epoch20.output, lambda x: x**2)
average_epoch20 = epoch_function.UnivariateStat(square_epoch20.output, 'mean')
log_power20 = function.ApplyFunction(average_epoch20.output, lambda x: np.log1p(x))
channel_updater19 = select.ChannelUpdater(channel_updater18.output, log_power20.output)

# Feature 21
top_f = top_features_info[20]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans21 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter21 = filter.ButterFilter(chans21.output, lowcut, highcut)
time_epoch21 = epoching.TimeBasedEpoching(butter_filter21.output, 1, 0.25)
square_epoch21 = function.ApplyFunction(time_epoch21.output, lambda x: x**2)
average_epoch21 = epoch_function.UnivariateStat(square_epoch21.output, 'mean')
log_power21 = function.ApplyFunction(average_epoch21.output, lambda x: np.log1p(x))
channel_updater20 = select.ChannelUpdater(channel_updater19.output, log_power21.output)

# Feature 22
top_f = top_features_info[21]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans22 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter22 = filter.ButterFilter(chans22.output, lowcut, highcut)
time_epoch22 = epoching.TimeBasedEpoching(butter_filter22.output, 1, 0.25)
square_epoch22 = function.ApplyFunction(time_epoch22.output, lambda x: x**2)
average_epoch22 = epoch_function.UnivariateStat(square_epoch22.output, 'mean')
log_power22 = function.ApplyFunction(average_epoch22.output, lambda x: np.log1p(x))
channel_updater21 = select.ChannelUpdater(channel_updater20.output, log_power22.output)

# Feature 23
top_f = top_features_info[22]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans23 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter23 = filter.ButterFilter(chans23.output, lowcut, highcut)
time_epoch23 = epoching.TimeBasedEpoching(butter_filter23.output, 1, 0.25)
square_epoch23 = function.ApplyFunction(time_epoch23.output, lambda x: x**2)
average_epoch23 = epoch_function.UnivariateStat(square_epoch23.output, 'mean')
log_power23 = function.ApplyFunction(average_epoch23.output, lambda x: np.log1p(x))
channel_updater22 = select.ChannelUpdater(channel_updater21.output, log_power23.output)

# Feature 24
top_f = top_features_info[23]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans24 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter24 = filter.ButterFilter(chans24.output, lowcut, highcut)
time_epoch24 = epoching.TimeBasedEpoching(butter_filter24.output, 1, 0.25)
square_epoch24 = function.ApplyFunction(time_epoch24.output, lambda x: x**2)
average_epoch24 = epoch_function.UnivariateStat(square_epoch24.output, 'mean')
log_power24 = function.ApplyFunction(average_epoch24.output, lambda x: np.log1p(x))
channel_updater23 = select.ChannelUpdater(channel_updater22.output, log_power24.output)

# Feature 25
top_f = top_features_info[24]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans25 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter25 = filter.ButterFilter(chans25.output, lowcut, highcut)
time_epoch25 = epoching.TimeBasedEpoching(butter_filter25.output, 1, 0.25)
square_epoch25 = function.ApplyFunction(time_epoch25.output, lambda x: x**2)
average_epoch25 = epoch_function.UnivariateStat(square_epoch25.output, 'mean')
log_power25 = function.ApplyFunction(average_epoch25.output, lambda x: np.log1p(x))
channel_updater24 = select.ChannelUpdater(channel_updater23.output, log_power25.output)

# Feature 26
top_f = top_features_info[25]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans26 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter26 = filter.ButterFilter(chans26.output, lowcut, highcut)
time_epoch26 = epoching.TimeBasedEpoching(butter_filter26.output, 1, 0.25)
square_epoch26 = function.ApplyFunction(time_epoch26.output, lambda x: x**2)
average_epoch26 = epoch_function.UnivariateStat(square_epoch26.output, 'mean')
log_power26 = function.ApplyFunction(average_epoch26.output, lambda x: np.log1p(x))
channel_updater25 = select.ChannelUpdater(channel_updater24.output, log_power26.output)

# Feature 27
top_f = top_features_info[26]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans27 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter27 = filter.ButterFilter(chans27.output, lowcut, highcut)
time_epoch27 = epoching.TimeBasedEpoching(butter_filter27.output, 1, 0.25)
square_epoch27 = function.ApplyFunction(time_epoch27.output, lambda x: x**2)
average_epoch27 = epoch_function.UnivariateStat(square_epoch27.output, 'mean')
log_power27 = function.ApplyFunction(average_epoch27.output, lambda x: np.log1p(x))
channel_updater26 = select.ChannelUpdater(channel_updater25.output, log_power27.output)

# Feature 28
top_f = top_features_info[27]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans28 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter28 = filter.ButterFilter(chans28.output, lowcut, highcut)
time_epoch28 = epoching.TimeBasedEpoching(butter_filter28.output, 1, 0.25)
square_epoch28 = function.ApplyFunction(time_epoch28.output, lambda x: x**2)
average_epoch28 = epoch_function.UnivariateStat(square_epoch28.output, 'mean')
log_power28 = function.ApplyFunction(average_epoch28.output, lambda x: np.log1p(x))
channel_updater27 = select.ChannelUpdater(channel_updater26.output, log_power28.output)

# Feature 29
top_f = top_features_info[28]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans29 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter29 = filter.ButterFilter(chans29.output, lowcut, highcut)
time_epoch29 = epoching.TimeBasedEpoching(butter_filter29.output, 1, 0.25)
square_epoch29 = function.ApplyFunction(time_epoch29.output, lambda x: x**2)
average_epoch29 = epoch_function.UnivariateStat(square_epoch29.output, 'mean')
log_power29 = function.ApplyFunction(average_epoch29.output, lambda x: np.log1p(x))
channel_updater28 = select.ChannelUpdater(channel_updater27.output, log_power29.output)

# Feature 30
top_f = top_features_info[29]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans30 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter30 = filter.ButterFilter(chans30.output, lowcut, highcut)
time_epoch30 = epoching.TimeBasedEpoching(butter_filter30.output, 1, 0.25)
square_epoch30 = function.ApplyFunction(time_epoch30.output, lambda x: x**2)
average_epoch30 = epoch_function.UnivariateStat(square_epoch30.output, 'mean')
log_power30 = function.ApplyFunction(average_epoch30.output, lambda x: np.log1p(x))
channel_updater29 = select.ChannelUpdater(channel_updater28.output, log_power30.output)

# Feature 31
top_f = top_features_info[30]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans31 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter31 = filter.ButterFilter(chans31.output, lowcut, highcut)
time_epoch31 = epoching.TimeBasedEpoching(butter_filter31.output, 1, 0.25)
square_epoch31 = function.ApplyFunction(time_epoch31.output, lambda x: x**2)
average_epoch31 = epoch_function.UnivariateStat(square_epoch31.output, 'mean')
log_power31 = function.ApplyFunction(average_epoch31.output, lambda x: np.log1p(x))
channel_updater30 = select.ChannelUpdater(channel_updater29.output, log_power31.output)

# Feature 32
top_f = top_features_info[31]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans32 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter32 = filter.ButterFilter(chans32.output, lowcut, highcut)
time_epoch32 = epoching.TimeBasedEpoching(butter_filter32.output, 1, 0.25)
square_epoch32 = function.ApplyFunction(time_epoch32.output, lambda x: x**2)
average_epoch32 = epoch_function.UnivariateStat(square_epoch32.output, 'mean')
log_power32 = function.ApplyFunction(average_epoch32.output, lambda x: np.log1p(x))
channel_updater31 = select.ChannelUpdater(channel_updater30.output, log_power32.output)

# Feature 33
top_f = top_features_info[32]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans33 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter33 = filter.ButterFilter(chans33.output, lowcut, highcut)
time_epoch33 = epoching.TimeBasedEpoching(butter_filter33.output, 1, 0.25)
square_epoch33 = function.ApplyFunction(time_epoch33.output, lambda x: x**2)
average_epoch33 = epoch_function.UnivariateStat(square_epoch33.output, 'mean')
log_power33 = function.ApplyFunction(average_epoch33.output, lambda x: np.log1p(x))
channel_updater32 = select.ChannelUpdater(channel_updater31.output, log_power33.output)

# Feature 34
top_f = top_features_info[33]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans34 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter34 = filter.ButterFilter(chans34.output, lowcut, highcut)
time_epoch34 = epoching.TimeBasedEpoching(butter_filter34.output, 1, 0.25)
square_epoch34 = function.ApplyFunction(time_epoch34.output, lambda x: x**2)
average_epoch34 = epoch_function.UnivariateStat(square_epoch34.output, 'mean')
log_power34 = function.ApplyFunction(average_epoch34.output, lambda x: np.log1p(x))
channel_updater33 = select.ChannelUpdater(channel_updater32.output, log_power34.output)

# Feature 35
top_f = top_features_info[34]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans35 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter35 = filter.ButterFilter(chans35.output, lowcut, highcut)
time_epoch35 = epoching.TimeBasedEpoching(butter_filter35.output, 1, 0.25)
square_epoch35 = function.ApplyFunction(time_epoch35.output, lambda x: x**2)
average_epoch35 = epoch_function.UnivariateStat(square_epoch35.output, 'mean')
log_power35 = function.ApplyFunction(average_epoch35.output, lambda x: np.log1p(x))
channel_updater34 = select.ChannelUpdater(channel_updater33.output, log_power35.output)

# Feature 36
top_f = top_features_info[35]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans36 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter36 = filter.ButterFilter(chans36.output, lowcut, highcut)
time_epoch36 = epoching.TimeBasedEpoching(butter_filter36.output, 1, 0.25)
square_epoch36 = function.ApplyFunction(time_epoch36.output, lambda x: x**2)
average_epoch36 = epoch_function.UnivariateStat(square_epoch36.output, 'mean')
log_power36 = function.ApplyFunction(average_epoch36.output, lambda x: np.log1p(x))
channel_updater35 = select.ChannelUpdater(channel_updater34.output, log_power36.output)

# Feature 37
top_f = top_features_info[36]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans37 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter37 = filter.ButterFilter(chans37.output, lowcut, highcut)
time_epoch37 = epoching.TimeBasedEpoching(butter_filter37.output, 1, 0.25)
square_epoch37 = function.ApplyFunction(time_epoch37.output, lambda x: x**2)
average_epoch37 = epoch_function.UnivariateStat(square_epoch37.output, 'mean')
log_power37 = function.ApplyFunction(average_epoch37.output, lambda x: np.log1p(x))
channel_updater36 = select.ChannelUpdater(channel_updater35.output, log_power37.output)

# Feature 38
top_f = top_features_info[37]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans38 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter38 = filter.ButterFilter(chans38.output, lowcut, highcut)
time_epoch38 = epoching.TimeBasedEpoching(butter_filter38.output, 1, 0.25)
square_epoch38 = function.ApplyFunction(time_epoch38.output, lambda x: x**2)
average_epoch38 = epoch_function.UnivariateStat(square_epoch38.output, 'mean')
log_power38 = function.ApplyFunction(average_epoch38.output, lambda x: np.log1p(x))
channel_updater37 = select.ChannelUpdater(channel_updater36.output, log_power38.output)

# Feature 39
top_f = top_features_info[38]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans39 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter39 = filter.ButterFilter(chans39.output, lowcut, highcut)
time_epoch39 = epoching.TimeBasedEpoching(butter_filter39.output, 1, 0.25)
square_epoch39 = function.ApplyFunction(time_epoch39.output, lambda x: x**2)
average_epoch39 = epoch_function.UnivariateStat(square_epoch39.output, 'mean')
log_power39 = function.ApplyFunction(average_epoch39.output, lambda x: np.log1p(x))
channel_updater38 = select.ChannelUpdater(channel_updater37.output, log_power39.output)

# Feature 40
top_f = top_features_info[39]
electrode_name = top_f['electrode']
chan_names.append(electrode_name)
chans40 = select.ChannelSelector(lsl_signal.output, 'name', [electrode_name])
lowcut, highcut = top_f['frequency_bin']
butter_filter40 = filter.ButterFilter(chans40.output, lowcut, highcut)
time_epoch40 = epoching.TimeBasedEpoching(butter_filter40.output, 1, 0.25)
square_epoch40 = function.ApplyFunction(time_epoch40.output, lambda x: x**2)
average_epoch40 = epoch_function.UnivariateStat(square_epoch40.output, 'mean')
log_power40 = function.ApplyFunction(average_epoch40.output, lambda x: np.log1p(x))
channel_updater39 = select.ChannelUpdater(channel_updater38.output, log_power40.output)


features_aggregated = feature.FeatureAggregator2(channel_updater39.output, chans_names=chan_names)
mi_class = classify.Classify2(features_aggregated.output, 'lda_recreated_model.sav', 'class')
#disp = display.Plot2(mi_class.output, channels=[], way = 'other')
classifier_output = io.LslSend2(mi_class.output, 'classifier_output',1, 'Markers', 'int32')


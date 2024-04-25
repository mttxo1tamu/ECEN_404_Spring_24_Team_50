import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import tensorflow as tf
import keras


def normalize(data):
    envlength = int(data.shape[0] / 24)
    envmax = np.zeros((envlength, data.shape[1]), float)
    envmin = np.zeros((envlength, data.shape[1]), float)
    for i in range(envlength):
        envmax[i, :] = np.ma.max(data[24 * i:24 * (i + 1), :], axis=0)
        envmin[i, :] = np.ma.min(data[24 * i:24 * (i + 1), :], axis=0)
    envamp = envmax - envmin
    envoffs = 0.5 * (envmax - envmin) + envmin
    signorm = data
    for i in range(envlength):
        for j in range(24):
            signorm[(i * 24 + j), :] = (data[(i * 24 + j), :] - envoffs[i, :]) / envamp[i, :]
    return signorm, envamp, envoffs


# def normalize(data):
#     envlength = int(data.shape[0] / 24)
#     envmax = np.zeros((envlength, data.shape[1]), np.complex64)
#     envmin = np.zeros((envlength, data.shape[1]), np.complex64)
#     for i in range(envlength):
#         maxmag = np.ma.max(data[24 * i:24 * (i + 1), :], axis=0)
#         minmag = np.ma.min(data[24 * i:24 * (i + 1), :], axis=0)
#         maxphase = np.exp(1j*np.pi*(1.0/12.0)*np.argmax(data[24 * i:24 * (i + 1), :], axis=0))
#         minphase = np.exp(1j*np.pi*(1.0/12.0)*np.argmin(data[24 * i:24 * (i + 1), :], axis=0))
#         envmax[i, :] = maxmag*maxphase
#         envmin[i, :] = minmag*minphase
#     envamp = envmax - envmin
#     envoffs = 0.5 * (envmax - envmin) + envmin
#     signorm = data
#     for i in range(envlength):
#         for j in range(24):
#             signorm[(i * 24 + j), :] = (data[(i * 24 + j), :] - envoffs[i, :]) / envamp[i, :]
#     return signorm, envmax, envmin, envamp, envoffs


# def denormalize(data, envamp, envoffs):
#     sigdenorm = data
#     for i in range(len(envamp)):
#         for j in range(24):
#             sigdenorm[(i*24 + j), :] = (data[(i*24 + j), :]*envamp[i, :] + envoffs[i, :])

def combFilt(xsig, invert=False, feedback=False):
    Kpls = 25
    qfilt = 0.9
    bcoeffs = np.zeros(Kpls)
    acoeffs = np.zeros(Kpls)
    if feedback:
        bcoeffs[0] = 1
        acoeffs[0] = 1
        acoeffs[-1] = -1*qfilt
    else:
        bcoeffs[0] = 1
        bcoeffs[-1] = qfilt
        acoeffs[0] = 1

    if invert:
        return signal.lfilter(b=acoeffs, a=bcoeffs, x=xsig, axis=0)
    else:
        return signal.lfilter(b=bcoeffs, a=acoeffs, x=xsig, axis=0)


def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    if delta:
        future = delta
    else:
        future = 0
    plt.title(title)
    for i, val in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.show()
    return


def model_train(dsnp, past, future, learning_rate, batch_size, epochs, nlstm, nDeep, train_split, name):
    train_data = dsnp[0:train_split, :]
    val_data = dsnp[train_split:, :]

    start = past + future
    end = start + train_split

    x_train = train_data
    y_train = dsnp[start:end, :]
    sequence_length = int(past)

    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=sequence_length,
        sampling_rate=1,
        batch_size=batch_size,
    )

    x_end = len(val_data) - past - future

    label_start = train_split + past + future

    x_val = val_data[:x_end, :]
    y_val = dsnp[label_start:, :]

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        x_val,
        y_val,
        sequence_length=sequence_length,
        sampling_rate=1,
        batch_size=batch_size,
    )

    for batch in dataset_train.take(1):
        inputs, targets = batch


    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)
    inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
    lstm_out = keras.layers.LSTM(nlstm)(inputs)
    outputs = keras.layers.Dense(nDeep)(lstm_out)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate), loss=tf.keras.losses.MeanAbsolutePercentageError())
    model.summary()
    path_checkpoint = name + "model_checkpoint.weights.h5"
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=3)
    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=[es_callback, modelckpt_callback],
    )

    visualize_loss(history, "Training and Validation Loss")

    model.save(name + "model.keras")

    return model, history, modelckpt_callback

# def generate_predictions(model, historical_data, batch_size, past, num_steps=336):
#     """
#     Generate predictions for the next `num_steps` time steps using batching.
#
#     Parameters:
#         model (keras.Model): Trained Keras model.
#         historical_data (pd.DataFrame): Historical data for selected features.
#         batch_size (int): Batch size for prediction (default is 1).
#         num_steps (int): Number of time steps to predict (default is 336).
#
#     Returns:
#         predictions (np.ndarray): Predictions for the next `num_steps` time steps.
#     """
#
#     predictions = []
#     num_batches = int(num_steps/batch_size)
#     steps = np.arange(num_batches)*batch_size
#     for i in steps:
#         # Extract input data for the current time step
#         input_data = np.broadcast_to(historical_data, (past, past, 8))[i:i+batch_size][:][:]
#         # Predictions for the current time step
#         step_predictions = model.predict(input_data, batch_size=batch_size, verbose=1, steps=num_steps)
#
#         # Append predictions to the list
#         predictions.append(step_predictions)
#
#     # Concatenate predictions into a single array
#     predictions = np.concatenate(predictions).flat
#     return predictions
#%%
#Database connection
from sqlalchemy import create_engine, MetaData
from sqlalchemy.engine import URL
import pypyodbc


###################################

# declares the variables and defines them for the server connections, along with the table names that are going to be assigned
SERVER_NAME = 'tcp:tecafs.database.windows.net,1433'
DATABASE_NAME = 'TecafsSqlDatabase'
TABLE_NAME = 'clean_data'

###################################

# makes the connection to the database with the connection string; has the driver, server name, database name, id, and password
connection_string = f"""
    DRIVER={{ODBC Driver 18 for SQL Server}};
    SERVER={SERVER_NAME};
    DATABASE={DATABASE_NAME};
    Uid={'tecafs2023'};
    Pwd={'Capstone50'};
"""



###################################

# attempted connection string, didn't work though

#connection_string1 = pypyodbc.connect("Driver={ODBC Driver 18 for SQL Server};Server=tcp:tecafs.database.windows.net,1433;Database=TecafsSqlDatabase;Uid=tecafs2023;Pwd={Capstone50};")

###################################


connection_url = URL.create('mssql+pyodbc', query={'odbc_connect': connection_string})
engine = create_engine(connection_url, module=pypyodbc)


metadata = MetaData()
metadata.reflect(bind=engine)

# Print the names of tables
print("Tables in the database:")
for table in metadata.tables.values():
    print(table.name)
#%%
Kpls = 25
qfilt = 0.6
bcoeffs = np.zeros(Kpls)
acoeffs = np.zeros(Kpls)
fb = True
if fb:
    bcoeffs[0] = 1
    acoeffs[0] = 1
    acoeffs[-1] = -1*qfilt
else:
    bcoeffs[0] = 1
    bcoeffs[-1] = qfilt
    acoeffs[0] = 1
# bcoeffs = np.zeros(Kpls)
# bcoeffs[0] = 1
# bcoeffs[-1] = qfilt
# acoeffs = np.zeros(Kpls)
# acoeffs[0] = 1
w, h = signal.freqz(acoeffs, bcoeffs, fs=24)


fig = plt.plot(w, 20 * np.log10(abs(h)), 'b')

plt.title('Comb Filter Frequency Response')

plt.ylabel('Amplitude [dB]')

plt.xlabel('Frequency [rad/day]')
plt.show()
#%%
#datatypes = {"Hour Ending": str, "COAST": np.float64, "EAST": np.float64, "FWEST": np.float64,
#             "NORTH": np.float64, "NCENT": np.float64, "SOUTH": np.float64, "SCENT": np.float64,
#             "WEST": np.float64, "ERCOT": np.float64}

datatypes = {"Hour Ending": str, "Coast": np.float64, "East": np.float64, "Far_West": np.float64,
             "North": np.float64, "North_C": np.float64, "Southern": np.float64, "South_C": np.float64,
             "West": np.float64, "ERCOT": np.float64}

#filePath = str(input("Enter address of .csv file to be analyzed."))

# DEBUG PURPOSES ONLY, REMOVE IN FINAL DEMO AND UNCOMMENT LINE ABOVE

# 2023 Only
#filePath = str('E:/Data/School/Senior Year Part 2/Spring 2024/ECEN 404/Native_Load_2023/2023_FullYear.csv')
filePath = str('F:/Data/School/Senior Year Part 2/Spring 2024/ECEN 404/Native_Load_2023/ERCOT Data 2002-2023 Fixed Dates.csv')

df = pd.read_csv(filePath, header=0, index_col=0, dtype=datatypes, parse_dates=True, skip_blank_lines=True,
                 date_format="np.datetime64", engine='c')
hour_end = df.index
predict_path = str("F:/Data/School/Senior Year Part 2/Spring 2024/ECEN 404/Native_Load_2023/2024Data.csv")
predict_data = pd.read_csv(predict_path, header=0, index_col=0, dtype=datatypes, parse_dates=True, skip_blank_lines=True,
                           date_format="np.datetime64", engine='c')
p_hour = predict_data.index
n_timestep = df.shape[0]

#split_fraction = 0.75
#predict_num = 0
#predict_split = n_timestep - predict_num
#train_split = int(split_fraction * predict_split)

#region_titles = ['COAST', 'EAST', 'FWEST', 'NORTH', 'NCENT', 'SOUTH', 'SCENT', 'WEST', 'ERCOT']
region_titles = ['Coast', 'East', 'Far_West', 'North', 'North_C', 'Southern', 'South_C', 'West', 'ERCOT']
selected_regions = [region_titles[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8]]
int_index = np.arange(df.shape[0])
int_pdex = np.arange(predict_data.shape[0])
regions = df[selected_regions]
data_2024 = predict_data[selected_regions]
regions.index = int_index
regions.head()
data_2024.index = int_pdex
data_2024.head()
regions = regions.interpolate(method='cubic', axis=0)
data_2024 = data_2024.interpolate(method='cubic', axis=0)


#predict_data = pd.DataFrame(regions.to_numpy()[predict_split:])
#regions = regions.to_numpy()[0:predict_split]
regions.index = hour_end
regions = regions.to_numpy()
data_2024.index = p_hour
data_2024 = data_2024.to_numpy()

regions, regamp, regoffs = normalize(regions)
data_2024, pamp, poffs = normalize(data_2024)
spectrumData = regions[:, :]
regions = combFilt(regions, feedback=True)
data_2024 = combFilt(data_2024, feedback=True)
regions = pd.DataFrame(regions)
data_2024 = pd.DataFrame(data_2024)

regions.head()
data_2024.head()

#%%
Wlength = 336
fs = 1
hlength = 24

w = signal.windows.parzen(Wlength, True)
wn = np.arange(len(w))

SFT = signal.ShortTimeFFT(w, hop=24, fs=24, mfft=None, scale_to='magnitude', fft_mode="onesided2X")
SFTn = spectrumData.shape[0]
SFTdays = SFTn/24
print(SFT.invertible)
wd = SFT.dual_win
dt_f = SFT.delta_f*fs

#%%
Sx = SFT.stft(spectrumData[:, 8])
Sxt = np.arange(SFT.p_num(len(spectrumData[:, 8])))

# Sxt[0]: p = -6: December 19, 2001 -  January 1, 2002 Center: December 26, 2001
# Sxt[6]: p = 0: December 25, 2001 - January 7, 2002, Center: January 1, 2002
# Sxt[13]: p = 7: January 1, 2002 - January 14, 2002, Center: January 8, 2002
# First window without padding
# Sxt[8033]: p = 8027 December 18, 2023 - December 31, 2023, Center: December 25, 2023
# Last window without padding
# Sxt[8040]: p = 8034 December 24, 2023 - January 6, 2024, Center: December 31, 2023
# Sxt[8047]: p = 8041 December 31, 2023 - January 13, 2024, Center: January 7, 2024
Sxf =  SFT.f
#Frequencies in cycles per day
Sxfrequencies = [fr * dt_f for fr in Sxf]
Sxtimestamps = np.arange('2001-12-26', '2024-01-08', dtype='datetime64[D]')
envtimestamps = np.arange('2002-01-01', '2024-01-01', dtype='datetime64[D]')

stftMag = pd.DataFrame(abs(Sx), Sxf, Sxtimestamps)
stftPhas = pd.DataFrame(np.angle(Sx), Sxf, Sxtimestamps)

plt.pcolormesh(Sxt, Sxf, np.abs(Sx), norm = "log", shading='gouraud', rasterized = True)

plt.title('ERCOT STFT Magnitude, 2002')

plt.ylabel('Frequency [cycles per day]')

plt.xlabel('Time [days]')

plt.show

#%%
#
# #Spectral decomposition function, currently defunct
# from numba import jit, prange
# @jit(nopython=True, parallel=True)
# def hourly_frequencies(Sx, Sxf, wd, hop, M, STFTn):
#     freqseq = np.zeros((STFTn, len(Sxf)), dtype=np.complex64)
#     for k in prange(STFTn):
#         pvals = np.arange(int(int(k/24) + -1*((M/hop)/2) + 1), int((int(k/24) + (M/hop)/2) + 1))
#         pcast = np.broadcast_to(np.expand_dims(pvals, axis=0), (len(Sxf), len(pvals)))
#         mup = k + int(M/2) - hop*pvals
#         Sxq = Sx[:, pvals]
#         Sxfcast = np.broadcast_to(np.expand_dims(Sxf, axis=1), (len(Sxf), len(pvals)))
#         window = np.broadcast_to(np.expand_dims(wd[mup], axis=0), (len(Sxf), len(pvals)))
#         mupcast = np.broadcast_to(np.expand_dims(mup, axis=0), (len(Sxf), len(pvals)))
#         xqpm = (1/M)*Sxq*np.exp(((2*1j*np.pi*(Sxfcast + (M/2))*mupcast)/M))
#         #rpk = np.transpose([fft.ifft(Sxq[:,i]) for i in range(len(pvals))])*window
#         #xqk = rpk*xqpm
#         freqseq[k, :] = np.sum(xqpm*window, axis=1)
#
#     print("Computation Complete")
#
#     return freqseq

#%%
#
# # Data organization via dicts, work in progress
# Sxtimestamps = np.arange('2001-12-26', '2024-01-08', dtype='datetime64[D]')
# envtimestamps = np.arange('2002-01-01', '2024-01-01', dtype='datetime64[D]')
# #6:8042
# RegionDict = {}
# #Should generalize for arbitrary input timestamps later
#
# for i in range(len(selected_regions)):
#     #Sxb = SFT.stft(spectrumData[:, i])
#     #magnitude = pd.DataFrame(abs(Sx), Sxf, Sxtimestamps)
#     #magnitude = magnitude.T
#     #magnp = pd.DataFrame.to_numpy(magnitude)[6:8041, :]
#     #phase = pd.DataFrame((np.angle(Sx)), Sxf, Sxtimestamps)
#     #phase = phase.T
#     #phasenp = pd.DataFrame.to_numpy(phase)[6:8041, :]
#     envelope_data = pd.DataFrame(data=dict({'amplitude': regamp[:, i], 'offset': regoffs[:, i]}), index=envtimestamps)
#     #Sft = np.array(SFT.stft(spectrumData[:, i]), dtype=np.complex64)
#     #Sxfreq = SFT.f
#     #dual_win = SFT.dual_win
#     #h = SFT.hop
#     #m_num = SFT.m_num
#     #signaln = len(spectrumData[:, i])
#     #freq_hr = hourly_frequencies(Sft, Sxfreq, dual_win, h, m_num, signaln)
#     #k0 = SFT.lower_border_end
#     #k1 = SFT.upper_border_begin(signaln)
#     #freq_checksum = np.sum(freq_hr, axis=1)
#     #iSft = np.array(SFT.istft(Sft), dtype=np.complex64)
#     #reconstruct_check = iSft[0:-312] / freq_checksum
#     #amplitude = pd.DataFrame(data=dict({'amplitude': regamp[:, i]}))
#     #ampnp = pd.DataFrame.to_numpy(amplitude)
#     #offset = pd.DataFrame(data=dict({'offset': regoffs[:, i]}))
#     #offnp = pd.DataFrame.to_numpy(offset)
#     #day_list = pd.DataFrame(data=dict({'day': Sxtimestamps[6:8041]}))
#     ts_hr = pd.Series(data=regions.to_numpy()[:, i], index=hour_end)
#     #fr_hr = pd.DataFrame(freq_hr, index=hour_end, columns=Sxfrequencies)
#     #data_array = np.hstack((ampnp, offnp, magnp, phasenp))
#     #power_data = {"magnitude": magnitude, "phase": phase, "amplitude": amplitude, "offset": offset, "day_list": day_list, "data_array": data_array, "ts_hr": ts_hr}
#     #power_data = {"fr_hr": fr_hr, "amplitude": amplitude, "offset": offset, "day_list": day_list, "ts_hr": ts_hr, "reconstruct_check": reconstruct_check, "istft": iSft, "checksum": freq_checksum}
#     power_data = {"envelope_data": envelope_data, "ts_hr": ts_hr}
#     RegionDict[selected_regions[i]] = power_data
#
# arrays = [RegionDict[selected_regions[i]]["ts_hr"] for i in range(len(selected_regions))]
# #dsnp = np.dstack(arrays)
#


#%%
# Data organization via dicts, work in progress
Sxtimestamps = np.arange('2001-12-26', '2024-01-08', dtype='datetime64[D]')
envtimestamps = np.arange('2002-01-01', '2024-01-01', dtype='datetime64[D]')
#6:8042
RegionDict = {}
#Should generalize for arbitrary input timestamps later
#5 years: 1826 days,43824 hours
#2 years: 730 days, 17520 hours
#1 year: 365 days, 8760 hours
day_split = -1826
hour_split = day_split*24
for i in range(len(selected_regions)):
    #power data
    envelope_data = pd.DataFrame(data=dict({'amplitude': regamp[day_split:, i], 'offset': regoffs[day_split:, i]}), index=envtimestamps[day_split:])
    #amplitude = np.abs(regamp[-17520:, i])
    ts_hr = pd.Series(data=regions.to_numpy()[hour_split:, i], index=hour_end[hour_split:])
    power_data = {"envelope_data": envelope_data, "ts_hr": ts_hr}
    #general data
    array = np.expand_dims(ts_hr.to_numpy(), axis=1)
    split_fraction = 0.75
    train_split = int(split_fraction * array.shape[0])
    name = str(selected_regions[i])
    # (array, past, future, learning_rate, batch_size, epochs, nlstm, nDeep, train_split)
    model, history, chkpt_callback = model_train(array, 720, 336, 0.005, 48, 10, 64, 12, train_split, name)

    RegionDict[selected_regions[i]] = {"model": model, "history": history,  "chkpt_callback": chkpt_callback, "power_data": power_data}




#%%


#%%
# Data organization via dicts, work in progress
# Prediction beta
# Sxtimestamps = np.arange('2001-12-26', '2024-01-08', dtype='datetime64[D]')
# envtimestamps = np.arange('2002-01-01', '2024-01-01', dtype='datetime64[D]')
# #6:8042
# RegionDict = {}
# #Should generalize for arbitrary input timestamps later
#
# for i in range(len(selected_regions)):
#     #power data
#     envelope_data = pd.DataFrame(data=dict({'amplitude': regamp[:, i], 'offset': regoffs[:, i]}), index=envtimestamps[-730:])
#     amplitude = np.abs(regamp[:, i])
#     ts_hr = pd.Series(data=regions.to_numpy()[:, i], index=hour_end)
#     power_data = {"envelope_data": envelope_data, "ts_hr": ts_hr}
#     #general data
#     array = np.expand_dims(ts_hr.to_numpy(), axis=1)
#     split_fraction = 0.75
#     train_split = int(split_fraction * array.shape[0])
#     name = str(selected_regions[i])
#     # (array, past, future, learning_rate, batch_size, epochs, nlstm, nDeep, train_split)
#     model, history, chkpt_callback = model_train(array, 720, 336, 0.001, 336, 10, 32, 1, train_split, name)
#
#     RegionDict[selected_regions[i]] = {"model": model, "history": history, "chkpt_callback": chkpt_callback, "power_data": power_data}

#%%
# past = 336
# future = 168
# learning_rate = 0.001
# batch_size = 168
# epochs = 10
# nlstm = 84
# nDeep = 9
# split_fraction = 0.75
# train_split = int(split_fraction * dsnp.shape[0])
#%%
# train_data = dsnp[0:train_split, :, :]
# val_data = dsnp[train_split:, :, :]
#
# start = past + future
# end = start + train_split
#
# x_train = train_data
# y_train = dsnp[start:end, :, :]
# sequence_length = int(past)
#
# dataset_train = keras.preprocessing.timeseries_dataset_from_array(
#     x_train,
#     y_train,
#     sequence_length=sequence_length,
#     sampling_rate=1,
#     batch_size=batch_size,
# )
#
# x_end = len(val_data) - past - future
#
# label_start = train_split + past + future
#
# x_val = val_data[:x_end, :, :]
# y_val = dsnp[label_start:, :, :]
#
# dataset_val = keras.preprocessing.timeseries_dataset_from_array(
#     x_val,
#     y_val,
#     sequence_length=sequence_length,
#     sampling_rate=1,
#     batch_size=batch_size,
# )
#
# for batch in dataset_train.take(1):
#     inputs, targets = batch
#
#
# print("Input shape:", inputs.numpy().shape)
# print("Target shape:", targets.numpy().shape)
#

#%%
# inputs = keras.layers.Input(shape=(168, 169, 9), batch_size=batch_size)
# reshape_layer = keras.ops.transpose(inputs, (0, 2, 3, 1))
# lstm_layer = keras.layers.LSTM(nlstm)
# dense_layer = keras.layers.Dense(nDeep)
# td1 = keras.layers.TimeDistributed(lstm_layer)(reshape_layer)
#
# outputs = keras.layers.TimeDistributed(dense_layer)(td1)
#
# model = keras.Model(inputs=inputs, outputs=outputs)
# model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate), loss=tf.keras.losses.MeanAbsolutePercentageError())
# model.summary()

#%%
# path_checkpoint = "model_checkpoint.weights.h5"
# es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
# modelckpt_callback = keras.callbacks.ModelCheckpoint(
#     monitor="val_loss",
#     filepath=path_checkpoint,
#     verbose=1,
#     save_weights_only=True,
#     save_best_only=True,
# )
#
# history = model.fit(
#     dataset_train,
#     epochs=epochs,
#     validation_data=dataset_val,
#     callbacks=[es_callback, modelckpt_callback],
# )
#
# visualize_loss(history, "Training and Validation Loss")

#%%
#
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import PolynomialFeatures
#
# x_plot = np.arange(int(n_timestep/24) - 1)
# amp_predictions = np.zeros((int(n_timestep/24), len(selected_regions)), float)
# off_predictions = np.zeros((int(n_timestep/24), len(selected_regions)), float)
# for i in range(len(selected_regions)):
#     y_plot = regamp[:, i]
#     z_plot = regoffs[:, i]
#
#     polynomial_features = PolynomialFeatures(degree=84, include_bias=False)
#     linear_regression = LinearRegression()
#     amp_pipeline = Pipeline(
#         [
#             ("polynomial_features", polynomial_features),
#             ("linear_regression", linear_regression),
#         ]
#     )
#
#     off_pipeline = Pipeline(
#         [
#             ("polynomial_features", polynomial_features),
#             ("linear_regression", linear_regression),
#         ]
#     )
#     amp_pipeline.fit(x_plot[:, np.newaxis], y_plot)
#     off_pipeline.fit(x_plot[:, np.newaxis], z_plot)
#     X_test = np.arange(int(n_timestep/24))
#     amp_predictions[:, i] = amp_pipeline.predict(X_test[:, np.newaxis])
#     off_predictions[:, i] = off_pipeline.predict(X_test[:, np.newaxis])
#
#
#
# plt.plot(X_test, amp_predictions, label="Amplitude Predictions")
# plt.scatter(x_plot, y_plot, edgecolor="b", s=20, label="Samples")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.xlim((0, 1))
# plt.ylim((-2, 2))
# plt.legend(loc="best")

#%%


#%%

# x_predict = regions.iloc[-720:][[i for i in range(9)]].values
# y_predict = predict_data
#
# dataset_predict = keras.preprocessing.timeseries_dataset_from_array(
#     x_predict,
#     y_predict,
#     sequence_length=sequence_length,
#     sampling_rate=1,
#     batch_size=batch_size,
# )
#
# for batch in dataset_predict.take(1):
#     inputs_predict, targets_predict = batch
#
# inputs_predict = keras.layers.Input(shape=(inputs_predict.shape[1], inputs_predict.shape[2]))
# data_predictions = model.predict(inputs_predict, batch_size=24, verbose=1, steps=14, callbacks=modelckpt_callback)



#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import tensorflow as tf
import keras


def normalize(data, train_split):
    data_mean = data[:train_split][:].mean(axis=0)
    data_std = data[:train_split][:].std(axis=0)
    normal_data = (data - data_mean) / data_std
    return normal_data, data_mean, data_std


def denormalize(data, data_mean, data_std):
    return (data * data_std) + data_mean


def combFilt(xsig, invert=False, feedback=False):
    Kpls = 25
    qfilt = 0.9
    bcoeffs = np.zeros(Kpls)
    acoeffs = np.zeros(Kpls)
    if feedback:
        bcoeffs[0] = 1
        acoeffs[0] = 1
        acoeffs[-1] = -1 * qfilt
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


def model_train(weather_array, power_array, past, future, learning_rate, batch_size, epochs, nlstm, nDeep, train_split, name):

    x_train = weather_array[:train_split]
    y_train = np.expand_dims(power_array[:train_split].values, axis=1)
    sequence_length = int(past)

    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=sequence_length,
        sampling_rate=1,
        batch_size=batch_size,
    )

    x_val = weather_array[train_split:]
    y_val = np.expand_dims(power_array[train_split:].values, axis=1)

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
    dense_in = keras.layers.Dense(nDeep)(lstm_out)
    outputs = keras.layers.Dense(batch_size)(dense_in)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate),
                  loss=tf.keras.losses.MeanAbsolutePercentageError())
    model.summary()
    path_checkpoint = name + "model_checkpoint.weights.h5"
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
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


def generate_predictions(model, predict_data, batch_size, past, num_steps=336):
    """
    Generate predictions for the next `num_steps` time steps using batching.

    Parameters:
        model (keras.Model): Trained Keras model.
        predict_data (pd.DataFrame): Historical data for selected features.
        batch_size (int): Batch size for prediction.
        past (int): Number of past time steps used as input to the model.
        num_steps (int): Number of time steps to predict (default is 336).

    Returns:
        predictions (np.ndarray): Predictions for the next `num_steps` time steps.
    """

    predictions = []

    # Calculate the number of batches
    num_batches = num_steps // batch_size
    steps = np.arange(num_batches) * batch_size
    # Iterate through each batch
    for i in steps:
        # Extract input data for the current batch
        input_data = np.broadcast_to(predict_data, (past, past, predict_data.shape[1]))[i:i + batch_size][:][:]

        # Generate predictions for the current batch
        step_predictions = model.predict(input_data, batch_size=batch_size, verbose=0)

        # Append predictions to the list
        predictions.extend(step_predictions[0])

    return np.array(predictions)


def interpolate_missing_hours(data, expected_timestamps):
    data_index = data.index
    # Determine what hours are expected that are not in the data
    missing_index = expected_timestamps.difference(data_index)
    # Create an array of NaN values representing the missing data
    nan_entries = np.full(shape=(len(missing_index), data.shape[1]), fill_value=np.nan, dtype=np.float64)
    # Add the new timestamp values and NaN entries to the index and data array, respectively
    new_index = data_index.append(missing_index)
    interpolate_array = np.concatenate((data.values, nan_entries), axis=0)
    # Ensure the new array is a dataframe and reconnect the index and columns
    # Sort the array so it can be interpolated, interpolate function will throw an error if data isn't sorted
    interpolate_array = pd.DataFrame(interpolate_array, columns=data.columns, index=new_index).sort_index(axis=0)
    # Interpolate data, method argument can be swapped with other options detailed in pandas interpolate documentation
    interpolate_array = interpolate_array.interpolate(method="linear", axis=0)
    return interpolate_array
# %%
# Database connection
import pandas as pd
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

connection_url = URL.create('mssql+pyodbc', query={'odbc_connect': connection_string})
engine = create_engine(connection_url, module=pypyodbc)

metadata = MetaData()
metadata.reflect(bind=engine)

ml_data = pd.read_sql_table("ML_forecast", engine)
# %%

# print("Tables in the database:")
# for table in metadata.tables.values():
#     print(table.name)

hist_data = pd.read_sql_table("hist_data", engine)
vdb_data = pd.read_sql_table("Milvus_forecast", engine)
weather_forecasts = pd.read_sql_table("weather_forecasts", engine)
hist_data.index = pd.to_datetime(hist_data.pop("Time"), format='%Y-%m-%d %H:%M:%S', unit='h')
weather_forecasts.index = pd.to_datetime(weather_forecasts.pop("Time"), format='%Y-%m-%d %H:%M:%S', unit='h')
ideal_timestamps = pd.Index(np.arange("2019-01-01T02:00", "2024-01-01T00:00", dtype='datetime64[h]'))
forecast_timestamps = pd.Index(np.arange("2024-04-25T00:00", "2024-05-09T00:00", dtype='datetime64[h]'))

feature_titles = np.array(['Location', 'Temperature', 'Dew Point', 'Humidity', 'Wind', 'Wind Speed',
                           'Wind Gust', 'Pressure', 'Precip.', 'Condition', 'Power'], dtype=str)

region_titles = ["COAST", "EAST", "FAR_WEST", "NORTH", "NORTH_C", "SOUTHERN", "SOUTH_C", "WEST"]

selected_features = ['Temperature', 'Dew Point', 'Humidity', 'Wind Speed', 'Pressure', 'Precip.', 'Power']
forecast_features = ['Temp.', 'Amount', 'Dew Point', 'Humidity', 'Wind', 'Pressure']
selected_regions = [region_titles[i] for i in [0, 1, 2, 3, 4, 5, 6, 7]]

regions = {}
forecast_regions = {}
selected_data = {}
selected_forecasts = {}
for title in region_titles:
    raw_reg = hist_data[hist_data['Location'] == title].sort_index(axis=0)
    raw_reg.index = pd.to_datetime(raw_reg.index)
    raw_fore = weather_forecasts[weather_forecasts['Location'] == title].sort_index(axis=0)
    raw_fore.index = pd.to_datetime(raw_fore.index)
    regions[title] = raw_reg
    forecast_regions[title] = raw_fore
    raw_reg = raw_reg.truncate(after=np.datetime64("2023-12-31T23:00"))[selected_features]
    raw_fore = raw_fore.truncate(after=np.datetime64("2024-05-08T23:00"))[forecast_features]
    fore_columns = raw_fore.columns.to_list()
    fore_columns = ['Temp.', 'Dew Point', 'Humidity', 'Wind', 'Pressure', 'Amount']
    raw_fore = raw_fore[fore_columns]
    raw_fore = raw_fore.rename(columns={'Temp.': 'Temperature', 'Wind':'Wind Speed', 'Amount': 'Precip.'})
    selected_data[title] = interpolate_missing_hours(raw_reg, ideal_timestamps)
    selected_forecasts[title] = raw_fore

forecast_features = ['Temperature', 'Dew Point', 'Humidity', 'Wind Speed', 'Pressure', 'Precip.']
# %%
RegionDict = {}
load_model = True
produce_forecast = True
for i in selected_regions:
    past = 336
    future = 336
    reg_data = selected_data[i].sort_index(axis=0)
    forecast_data = selected_forecasts[i].sort_index(axis=0)
    reg_timestamps = reg_data.index
    reg_columns = reg_data.columns
    forecast_columns = forecast_data.columns

    split_fraction = 0.75
    train_split = int(split_fraction * (reg_data.shape[0] - past))
    reg_data, reg_mean, reg_std = normalize(reg_data, train_split)
    forecast_data, forecast_mean, forecast_std = normalize(forecast_data, len(forecast_data))
    reg_data = pd.DataFrame(combFilt(reg_data), index=reg_timestamps, columns=reg_columns)
    forecast_data = pd.DataFrame(combFilt(forecast_data), index=forecast_timestamps, columns=forecast_columns)
    reg_data = reg_data.sort_index(axis=0)
    forecast_data = forecast_data.sort_index(axis=0)
    weather_data = reg_data[forecast_features]
    power_data = pd.Series(reg_data["Power"], index=reg_timestamps)
    pred_data = forecast_data
    forecast_timestamps = forecast_data.index

    name = i
    batch_size = 84
    learning_rate = 0.002
    epochs = 20
    nlstm = 24
    nDeep = 64
    if load_model:
        model = keras.saving.load_model(name + "model.keras")
    else:
        model, history, chkpt_callback = model_train(weather_data, power_data, past, future, learning_rate, batch_size,
                                                     epochs, nlstm, nDeep, train_split, name)

    if produce_forecast:
        forecast = generate_predictions(model, pred_data, batch_size, past, future)
        forecast = combFilt(forecast, invert=True)
        forecast = denormalize(forecast, reg_mean['Power'], reg_std['Power'])
        forecast = pd.Series(forecast, index=forecast_timestamps)
        RegionDict[i] = {"model": model, "ts_hr": reg_data, "weather_forecast": forecast_data, "power_forecast": forecast}
    else:
        RegionDict[i] = {"model": model, "ts_hr": reg_data, "weather_forecast": forecast_data}
# %%

# Refresh SQL Connection if needed
engine = create_engine(connection_url, module=pypyodbc)

metadata = MetaData()
metadata.reflect(bind=engine)
# Write Forecasts to SQL Database
write_dict = {"location": [], "time": [], "year": [], "date": [], "hour": [], "forecast power": []}
column_labels = ["location", "time", "year", "date", "hour", "forecast power"]
for title in selected_regions:
    region_forecast = RegionDict[title]["power_forecast"]
    timestamps = region_forecast.index
    years = timestamps.year.astype(int)
    write_dict["year"].append(years)
    days = timestamps.strftime("%m-%d")
    write_dict["date"].append(days)
    hours = timestamps.strftime("%H:%M")
    write_dict["hour"].append(hours)
    #Don't need historical data, just combined datetime, split date components, and forecast
    power_forecast = region_forecast.values
    write_dict["forecast power"].append(power_forecast)
    location_tags = np.full(shape=len(region_forecast), fill_value=title, dtype="S8")
    write_dict["location"].append(location_tags)
    write_dict["time"].append(timestamps.to_list())


for i in write_dict.keys():
    write_dict[i] = np.concatenate(write_dict[i])
write_df = pd.DataFrame(write_dict, columns=column_labels)
write_df.to_sql('ML_forecast', engine, if_exists='replace', index=False)
ml_data = pd.read_sql_table("ML_forecast", engine)
#%% md
# #If I receive rollback error
# connection = engine.connect()
# connection.rollback()
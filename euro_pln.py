import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
import time
from keras.models import load_model


# FUNKCJE
# ======================================================================================================================


def get_data(data, look_back):
    data_x, data_y = [], []
    for i in range(len(data) - look_back - 1):
        data_x.append(data[i : (i + look_back), 0])
        data_y.append(data[i + look_back, 0])
    return np.array(data_x), np.array(data_y)


def create_model(
    kernel_initializer="glorot_uniform",
    optimizer="adam",
    loss="mean_squared_error",
    n_features=1,
    units=100,
    dropout=0.0,
    activation="tanh",
):
    # def create_model(kernel_initializer, optimizer, loss, units, dropout, activation, n_features=1):
    model = Sequential()
    model.add(
        LSTM(
            units=units,
            input_shape=(1, 1),
            dropout=dropout,
            activation=activation,
            kernel_initializer=kernel_initializer,
        )
    )
    model.add(Dense(n_features))
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def random_search_CV(regressor, x_train, y_train):
    batch_size = [10, 20, 40, 60, 80, 100, 200]
    epochs = [10, 50, 100, 200]
    dropout = [0, 0.25, 0.5, 0.75]
    kernel_initializer = [
        "uniform",
        "lecun_uniform",
        "normal",
        "zero",
        "glorot_normal",
        "glorot_uniform",
        "he_normal",
        "he_uniform",
    ]
    loss = ["mse", "log_cosh", "binary_crossentropy"]
    optimizer = ["sgd", "nadam", "adam", "adadelta"]
    activation = [
        "tanh",
        "relu",
        "sigmoid",
        "linear",
        "softmax",
        "softplus",
        "softsign",
        "selu",
        "elu",
        "exponential",
        None,
    ]
    units = [1, 10, 20, 50, 100, 200]
    param_distributions = dict(
        batch_size=batch_size,
        epochs=epochs,
        dropout=dropout,
        kernel_initializer=kernel_initializer,
        loss=loss,
        optimizer=optimizer,
        activation=activation,
        units=units,
    )
    grid = RandomizedSearchCV(
        estimator=regressor,
        param_distributions=param_distributions,
        n_jobs=-1,
        cv=3,
    )
    return grid.fit(x_train, y_train)
    """
    Best:
    
    """


def grid_search_CV(regressor, x_train, y_train):
    units = [50]
    activation = [None]
    optimizer = ["nadam"]
    loss = ["log_cosh"]
    kernel_initializer = ["uniform"]
    batch_size = [10]
    epochs = [50]
    dropout = [0]
    param_grid = dict(
        batch_size=batch_size,
        epochs=epochs,
        dropout=dropout,
        kernel_initializer=kernel_initializer,
        loss=loss,
        optimizer=optimizer,
        activation=activation,
        units=units,
    )
    grid = GridSearchCV(
        estimator=regressor, param_grid=param_grid, n_jobs=-1, cv=3
    )
    return grid.fit(x_train, y_train)
    """
    testy:
    batch_size = [10, 20, 40, 60, 80, 100, 200]
    epochs = [10, 50, 100, 200]
    # Best: -0.000164 using {'batch_size': 10, 'epochs': 50}
    batch_size = [10]
    epochs = [50]
    dropout = [0, 0.25, 0.5, 0.75]
    # Best: -0.000166 using {'batch_size': 10, 'dropout': 0, 'epochs': 50}
    kernel_initializer = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    batch_size = [10]
    epochs = [50]
    dropout = [0]
    # Best: -0.000166 using {'batch_size': 10, 'dropout': 0, 'epochs': 50, 'kernel_initializer': 'uniform'}
    loss = ['mse', 'log_cosh', 'binary_crossentropy']
    kernel_initializer = ['uniform']
    batch_size = [10]
    epochs = [50]
    dropout = [0]
    # Best: -0.000099 using {'batch_size': 10, 'dropout': 0, 'epochs': 50, 'kernel_initializer': 'uniform', 'loss': 'log_cosh'}
    optimizer = ['sgd', 'nadam', 'adam', 'adadelta']
    loss = ['log_cosh']
    kernel_initializer = ['uniform']
    batch_size = [10]
    epochs = [50]
    dropout = [0]
    # Best: -0.000093 using {'batch_size': 10, 'dropout': 0, 'epochs': 50, 'kernel_initializer': 'uniform', 'loss': 'log_cosh', 'optimizer': 'nadam'}
    activation = ['tanh', 'relu', 'sigmoid', 'linear', 'softmax', 'softplus', 'softsign', 'selu', 'elu', 'exponential', None]
    optimizer = ['nadam']
    loss = ['log_cosh']
    kernel_initializer = ['uniform']
    batch_size = [10]
    epochs = [50]
    dropout = [0]
    # Best: -0.000081 using {'activation': None, 'batch_size': 10, 'dropout': 0, 'epochs': 50, 'kernel_initializer': 'uniform', 'loss': 'log_cosh', 'optimizer': 'nadam'}
    units = [1, 10, 20, 50, 100, 200]
    activation = [None]
    optimizer = ['nadam']
    loss = ['log_cosh']
    kernel_initializer = ['uniform']
    batch_size = [10]
    epochs = [50]
    dropout = [0]
    # Best: -0.000083 using {'activation': None, 'batch_size': 10, 'dropout': 0, 'epochs': 50, 'kernel_initializer': 'uniform', 'loss': 'log_cosh', 'optimizer': 'nadam', 'units': 50}
    """


# ======================================================================================================================


if __name__ == "__main__":
    dataset = pd.read_csv(
        "./data/euro-daily-hist_1999_2020.csv", na_values="-"
    )
    # print(dataset.shape)
    # print(dataset.info())

    # podgląd danych
    data = dataset["[Polish zloty ]"]
    data = data.dropna()
    # print(data.shape)
    # print(data.size)  # rozmiar
    # print("Utracono:", 100*(1 - (data.size / len(dataset))), "% danych")
    # plt.plot(data)
    # plt.yticks(np.arange(min(data), max(data) + 0.1, 0.1))
    # plt.show()

    # przygotowanie danych pod model
    data = data.to_numpy().reshape(-1, 1)
    # print(data.shape)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # podział na zbiór testowy(5000) i treningowy(662)
    train = data[:5000]
    test = data[5000:]
    # print(train.shape)
    # print(test.shape)

    # przygotowanie zbiorów treningowego i testowego
    look_back = 1
    x_train, y_train = get_data(train, look_back)
    x_test, y_test = get_data(test, look_back)
    # print(x_test.shape)
    # print(y_test.shape)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    # print(x_train.shape)
    # print(y_train.shape)

    # definiowanie modelu LSTM
    n_features = x_train.shape[1]
    model_regressor = KerasRegressor(build_fn=create_model)

    # strojenie hiperparametrów
    # grid_result = random_search_CV(model_regressor, x_train, y_train)
    # grid_result = grid_search_CV(model_regressor, x_train, y_train)
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # model_regressor.set_params(**grid_result.best_params_)
    # model_regressor.fit(x_train, y_train)

    # zapisywanie i wczytanie modelu
    # model_regressor.model.save(time.strftime("%Y%m%d-%H%M%S") + '.h5')
    model_regressor = tf.keras.models.load_model("20210228-202234.h5")

    # przewidywania i porównanie ze zbiorem testowym
    # przewidywania modelu
    y_pred = model_regressor.predict(x_test)
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred)
    # print(y_pred[:10])

    # zbiór testowy
    y_test = np.array(y_test).reshape(-1, 1)
    y_test = scaler.inverse_transform(y_test)
    # print(y_test[:10])

    # wizualizacja
    plt.figure(figsize=(10, 5))
    plt.title("Exchange Rate EUR - PLN")
    # plt.scatter(y_test, y_pred)
    plt.plot(y_test, label="Actual", color="b")
    plt.plot(y_pred, label="Predicted", color="r")
    plt.legend()
    plt.show()

    print(mean_squared_error(y_test, y_pred))

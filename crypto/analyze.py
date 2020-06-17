import pandas as pd

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale

import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

from pmdarima.arima import auto_arima


def arima_train(actual, P, D, Q):
    model = ARIMA(actual, order=(P, D, Q))
    model_fit = model.fit(disp=0)

    # Explicit 1 step forecasting
    prediction = model_fit.forecast(steps=1)[0]

    return prediction, model_fit


def forecast(actual_data, train_percentage):
    X = actual_data

    size = int(len(X) * train_percentage)
    train, test = X[0:size], X[size:len(X)]
    predictions = list()
    actuals = list()
    history = [x for x in train['value']]

    model = {}

    for timepoint in range(len(test)):
        actual = test['value'].iloc[timepoint]
        time = test['time'].iloc[timepoint]

        # Predict only even timepoints (2 hours ahead)
        if timepoint % 2 == 0:
            history.append(actual)
            continue

        prediction, model = arima_train(history, 0, 1, 2)
        predictions.append([time, prediction[0]])
        history.append(actual)
        actuals.append([time, actual])

    print(model.summary())

    # Print MSE
    actuals = pd.DataFrame(data=actuals, columns=['time', 'value'])
    pred = pd.DataFrame(data=predictions, columns=['time', 'value'])

    actuals_scaled = minmax_scale(actuals['value'], feature_range=(0, 1))
    pred_scaled = minmax_scale(pred['value'], feature_range=(0, 1))

    error = mean_squared_error(actuals_scaled, pred_scaled) * 100
    print('MAE vs observed: %.5f%%' % error)

    return pred, pd.DataFrame(data=train, columns=['time', 'value']), actuals


def get_stationarity(df, coin):
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index(keys=["time"])

    # Rolling Statistics
    rolling_mean = df.rolling(window=30).mean()
    rolling_std = df.rolling(window=30).std()

    plt.plot(df, color='blue', label='Original')
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.plot(rolling_std, color='black', label='Rolling Std')

    plt.legend(loc='best')
    plt.title(f"{coin} Rolling Mean & Rolling Standard Deviation")
    plt.show(block=False)

    # Dickey-Fuller Test
    result = adfuller(df['value'])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))


def get_auto_arima(df):
    stepwise_model = auto_arima(df["value"], start_p=1, start_q=1,
                                max_p=3, max_q=3, m=12,
                                start_P=0, seasonal=True,
                                d=1, D=1, trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)

    print(stepwise_model.aic())


def main():
    print("Starting analytics.")

    coins = ['btc', 'eth']
    hour_count = 250
    train_percentage = 0.66

    print(f"Using {hour_count} total hours of which {int(train_percentage*100)}% is training and {int((1-train_percentage)*100)}% is test.")
    print()

    for coin in coins:
        print()
        print("=============================================")
        print(f"Predicting {coin}.")

        d = pd.read_csv(f"{coin}.csv")

        print("Sample data: ")
        print(d.head().to_string())
        print()

        df = d[0:hour_count].copy()

        #get_auto_arima(df)
        get_stationarity(df, coin)

        predictions, training, actual = forecast(df, train_percentage)

        df['time'] = pd.to_datetime(df['time'], unit='s')

        predictions['time'] = pd.to_datetime(predictions['time'], unit='s')
        training['time'] = pd.to_datetime(training['time'], unit='s')
        actual['time'] = pd.to_datetime(actual['time'], unit='s')

        ax = predictions.plot(kind='line', linestyle="solid", x='time', y='value', colormap='Greens_r',
                              label='prediction')
        training.plot(ax=ax, kind='line', linestyle="solid", x='time', y='value', colormap='Blues_r', label='training')
        actual.plot(ax=ax, kind='line', linestyle="solid", x='time', y='value', colormap='Oranges_r', label='actual')

        plt.title('Forecast vs Actuals')
        plt.legend(loc='upper left', fontsize=8)

        print(f"Done, saving chart to {coin}.pdf.")

        fig = ax.get_figure()
        fig.savefig(f"{coin}.pdf")

        plt.cla()

        print()

    print("Finished analytics.")


if __name__ == "__main__":
    main()

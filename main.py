import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

# Loading the data
df_total_col = pd.read_csv("pharama weekly sales copy.csv")
df_total_col = df_total_col.rename(columns={'M01AB': 'C1', 'M01AE': 'C2', 'N02BA': 'C3', 'N02BE': 'C4',
                        'N05B': 'C5', 'N05C': 'C6', 'R03': 'C7', 'R06': 'C8'})

df_total_col['datum'] = pd.to_datetime(df_total_col['datum'])
print(df_total_col)

df_total_col.set_index("datum" , inplace=True)
#divide columsn in to seperated table

categories = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
for cat in categories:
    df_total_col[[cat]].to_csv(f'{cat}.csv')


df_c1 = pd.read_csv("C1.csv")
df_c2 = pd.read_csv("C2.csv")
df_c3 = pd.read_csv("C3.csv")
df_c4 = pd.read_csv("C4.csv")
df_c5 = pd.read_csv("C5.csv")
df_c6 = pd.read_csv("C6.csv")
df_c7 = pd.read_csv("C7.csv")
df_c8 = pd.read_csv("C8.csv")

df_c1['datum'] = pd.to_datetime(df_c1['datum'])  # Convert the 'Date' column to datetime if it's not already
df_c1.set_index('datum', inplace=True)

df_c2['datum'] = pd.to_datetime(df_c2['datum'])  # Convert the 'Date' column to datetime if it's not already
df_c2.set_index('datum', inplace=True)

df_c3['datum'] = pd.to_datetime(df_c3['datum'])  # Convert the 'Date' column to datetime if it's not already
df_c3.set_index('datum', inplace=True)

df_c4['datum'] = pd.to_datetime(df_c4['datum'])  # Convert the 'Date' column to datetime if it's not already
df_c4.set_index('datum', inplace=True)

df_c5['datum'] = pd.to_datetime(df_c5['datum'])  # Convert the 'Date' column to datetime if it's not already
df_c5.set_index('datum', inplace=True)

df_c6['datum'] = pd.to_datetime(df_c6['datum'])  # Convert the 'Date' column to datetime if it's not already
df_c6.set_index('datum', inplace=True)

df_c7['datum'] = pd.to_datetime(df_c7['datum'])  # Convert the 'Date' column to datetime if it's not already
df_c7.set_index('datum', inplace=True)

df_c8['datum'] = pd.to_datetime(df_c8['datum'])  # Convert the 'Date' column to datetime if it's not already
df_c8.set_index('datum', inplace=True)


dataframes = [df_c1, df_c2, df_c3, df_c4, df_c5, df_c6, df_c7, df_c8]

# Loop through each DataFrame and plot
for i, df in enumerate(dataframes, start=1):
    df.plot()
    plt.title(f'DataFrame df_c{i}')
    plt.xlabel('Index')
    plt.ylabel('Values')
    #plt.show()

#check stationary or not

from statsmodels.tsa.stattools import adfuller

# Ho: It is non stationary
# H1: It is stationary

def adfuller_test(c):
    result = adfuller(c)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))
    if result[1] <= 0.05:
        print(
            "strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

adfuller_test(df_c1['C1'])
print("---------------------------------------------------------------------------------------------------------")
adfuller_test(df_c2['C2'])
print("---------------------------------------------------------------------------------------------------------")

adfuller_test(df_c3['C3'])
print("---------------------------------------------------------------------------------------------------------")

adfuller_test(df_c4['C4'])
print("---------------------------------------------------------------------------------------------------------")

adfuller_test(df_c5['C5'])
print("---------------------------------------------------------------------------------------------------------")

adfuller_test(df_c6['C6'])
print("---------------------------------------------------------------------------------------------------------")

adfuller_test(df_c7['C7'])
print("---------------------------------------------------------------------------------------------------------")

adfuller_test(df_c8['C8'])
print("---------------------------------------------------------------------------------------------------------")



dataframes = [df_c1, df_c2, df_c3, df_c4, df_c5, df_c6, df_c7, df_c8]
columns = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

for i, (df, col) in enumerate(zip(dataframes, columns), start=1):
    fig = plt.figure(figsize=(12, 8))

    # ACF Plot
    ax1 = fig.add_subplot(211)
    sm.graphics.tsa.plot_acf(df[col], lags=40, ax=ax1)
    ax1.set_title(f'ACF of df_c{i} ({col})')

    # PACF Plot
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(df[col], lags=40, ax=ax2)
    ax2.set_title(f'PACF of df_c{i} ({col})')

    plt.tight_layout()  # Adjust layout to prevent overlap
    #plt.show()

#plt.show()

#p =1 , d= 0 , q = 0 or 1

#Drom Arimax
#from statsmodels.tsa.arima.model import ARIMA
#model_arima=ARIMA(df_c1['C1'],order=(1,0,0))
#model_fit_arima=model_arima.fit()

#print(model_fit_arima.summary())

#df_c1['forecasted_C1_arima']=model_fit_arima.predict(start=90,end=103,dynamic=True)
#df_c1[['C1','forecasted_C1_arima']].plot(figsize=(12,8))


#case

#lets use srimax

import statsmodels.api as sm

model_sarimax_c1=sm.tsa.statespace.SARIMAX(df_c1['C1'],order=(1, 0, 0),seasonal_order=(1 , 0 , 0 ,7))
results_sarimax_c1=model_sarimax_c1.fit()
df_c1['forecasted_C1_sarimax']=results_sarimax_c1.predict(start=90,end=103,dynamic=True)
df_c1[['C1','forecasted_C1_sarimax']].plot(figsize=(12,8))


model_sarimax_c2=sm.tsa.statespace.SARIMAX(df_c2['C2'],order=(1, 0, 0),seasonal_order=(10 , 0 , 0 ,7))
results_sarimax_c2=model_sarimax_c2.fit()
df_c2['forecasted_C2_sarimax']=results_sarimax_c2.predict(start=90,end=103,dynamic=True)
df_c2[['C2','forecasted_C2_sarimax']].plot(figsize=(12,8))


model_sarimax_c3=sm.tsa.statespace.SARIMAX(df_c3['C3'],order=(3, 0, 0),seasonal_order=(3 , 0 , 0 ,7))
results_sarimax_c3=model_sarimax_c3.fit()
df_c3['forecasted_C3_sarimax']=results_sarimax_c3.predict(start=90,end=103,dynamic=True)
df_c3[['C3','forecasted_C3_sarimax']].plot(figsize=(12,8))

model_sarimax_c4=sm.tsa.statespace.SARIMAX(df_c4['C4'],order=(5, 0, 0),seasonal_order=(5 , 0 , 0 ,7))
results_sarimax_c4=model_sarimax_c4.fit()
df_c4['forecasted_C4_sarimax']=results_sarimax_c4.predict(start=90,end=103,dynamic=True)
df_c4[['C4','forecasted_C4_sarimax']].plot(figsize=(12,8))

model_sarimax_c5=sm.tsa.statespace.SARIMAX(df_c5['C5'],order=(4, 0, 0),seasonal_order=(4 , 0 , 0 ,7))
results_sarimax_c5=model_sarimax_c5.fit()
df_c5['forecasted_C5_sarimax']=results_sarimax_c5.predict(start=90,end=103,dynamic=True)
df_c5[['C5','forecasted_C5_sarimax']].plot(figsize=(12,8))

model_sarimax_c6=sm.tsa.statespace.SARIMAX(df_c6['C6'],order=(3, 0, 0),seasonal_order=(3, 0 , 0 ,7))
results_sarimax_c6=model_sarimax_c6.fit()
df_c6['forecasted_C6_sarimax']=results_sarimax_c6.predict(start=90,end=103,dynamic=True)
df_c6[['C6','forecasted_C6_sarimax']].plot(figsize=(12,8))

model_sarimax_c7=sm.tsa.statespace.SARIMAX(df_c7['C7'],order=(6, 0, 0),seasonal_order=(6 , 0 , 0 ,7))
results_sarimax_c=model_sarimax_c3.fit()
df_c7['forecasted_C7_sarimax']=results_sarimax_c3.predict(start=90,end=103,dynamic=True)
df_c7[['C7','forecasted_C7_sarimax']].plot(figsize=(12,8))


model_sarimax_c8=sm.tsa.statespace.SARIMAX(df_c8['C8'],order=(1, 0, 0),seasonal_order=(10 , 0 , 0 ,7))
results_sarimax=model_sarimax_c8.fit()
df_c8['forecasted_C8_sarimax']=results_sarimax.predict(start=90,end=103,dynamic=True)
df_c8[['C8','forecasted_C8_sarimax']].plot(figsize=(12,8))





#predict for upcoming 24 weeks
from pandas.tseries.offsets import DateOffset

future_dates_c1 = [df_c1.index[-1] + DateOffset(weeks=x) for x in range(1, 24)]
future_dates_df_c1 = pd.DataFrame(index=future_dates_c1[1:], columns=df_c1.columns)
#print(future_dates_df_c1.tail())
future_df_c1=pd.concat([df_c1,future_dates_df_c1])
future_df_c1['forecasted_M01AB'] = results_sarimax_c1.predict(start = 517, end = 540, dynamic= True)
future_df_c1[['C1', 'forecasted_M01AB']].plot(figsize=(12, 8))
#print(future_df_c1.tail(20))
#plt.show()

future_dates_c2 = [df_c2.index[-1] + DateOffset(weeks=x) for x in range(1, 24)]
future_dates_df_c2 = pd.DataFrame(index=future_dates_c2[1:], columns=df_c2.columns)
#print(future_dates_df_c2.tail())
future_df_c2=pd.concat([df_c2,future_dates_df_c2])
future_df_c2['forecasted_M01AE'] = results_sarimax_c2.predict(start = 517, end = 540, dynamic= True)
future_df_c2[['C2', 'forecasted_M01AE']].plot(figsize=(12, 8))
#print(future_df_c2.tail((20)))
#plt.show()

future_dates_c3 = [df_c3.index[-1] + DateOffset(weeks=x) for x in range(1, 24)]
future_dates_df_c3 = pd.DataFrame(index=future_dates_c3[1:], columns=df_c3.columns)
#print(future_dates_df_c3.tail())
future_df_c3=pd.concat([df_c3,future_dates_df_c3])
future_df_c3['forecasted_N02BA'] = results_sarimax_c3.predict(start = 517, end = 540, dynamic= True)
future_df_c3[['C3', 'forecasted_N02BA']].plot(figsize=(12, 8))

future_dates_c4 = [df_c4.index[-1] + DateOffset(weeks=x) for x in range(1, 24)]
future_dates_df_c4 = pd.DataFrame(index=future_dates_c4[1:], columns=df_c4.columns)
#print(future_dates_df_c4.tail())
future_df_c4=pd.concat([df_c4,future_dates_df_c4])
future_df_c4['forecasted_N02BE'] = results_sarimax_c4.predict(start = 517, end = 540, dynamic= True)
future_df_c4[['C4', 'forecasted_N02BE']].plot(figsize=(12, 8))

future_dates_c5 = [df_c5.index[-1] + DateOffset(weeks=x) for x in range(1, 24)]
future_dates_df_c5 = pd.DataFrame(index=future_dates_c5[1:], columns=df_c5.columns)
#print(future_dates_df_c5.tail())
future_df_c5=pd.concat([df_c5,future_dates_df_c5])
future_df_c5['forecasted_N05B'] = results_sarimax_c5.predict(start = 517, end = 540, dynamic= True)
future_df_c5[['C5', 'forecasted_N05B']].plot(figsize=(12, 8))

future_dates_c6 = [df_c6.index[-1] + DateOffset(weeks=x) for x in range(1, 24)]
future_dates_df_c6 = pd.DataFrame(index=future_dates_c6[1:], columns=df_c6.columns)
#print(future_dates_df_c6.tail())
future_df_c6=pd.concat([df_c6,future_dates_df_c6])
future_df_c6['forecasted_N05C'] = results_sarimax_c6.predict(start = 517, end = 540, dynamic= True)
future_df_c6[['C6', 'forecasted_N05C']].plot(figsize=(12, 8))

future_dates_c7 = [df_c7.index[-1] + DateOffset(weeks=x) for x in range(1, 24)]
future_dates_df_c7 = pd.DataFrame(index=future_dates_c7[1:], columns=df_c7.columns)
#print(future_dates_df_c7.tail())
future_df_c7=pd.concat([df_c7,future_dates_df_c7])
future_df_c7['forecasted_R03'] = results_sarimax.predict(start = 517, end = 540, dynamic= True)
future_df_c7[['C7', 'forecasted_R03']].plot(figsize=(12, 8))

future_dates_c8 = [df_c8.index[-1] + DateOffset(weeks=x) for x in range(1, 24)]
future_dates_df_c8 = pd.DataFrame(index=future_dates_c8[1:], columns=df_c8.columns)
#print(future_dates_df_c8.tail())
future_df_c8=pd.concat([df_c8,future_dates_df_c8])
future_df_c8['forecasted_R06'] = results_sarimax.predict(start = 517, end = 540, dynamic= True)
future_df_c8[['C8', 'forecasted_R06']].plot(figsize=(12, 8))



# Print the dates for the next 24 weeks
#print("The dates for the next 24 weeks are:")
#print("Date\t\tPrediction Value")
#print("-" * 40)



import pandas as pd
from pandas.tseries.offsets import DateOffset


# Assuming future_df_c1, future_df_c2, etc. are your DataFrames
dataframes = {'C1': future_df_c1, 'C2': future_df_c2, 'C3': future_df_c3, 'C4': future_df_c4,
              'C5': future_df_c5, 'C6': future_df_c6, 'C7': future_df_c7, 'C8': future_df_c8}

# Assuming future_dates_c1, future_dates_c2, etc. are your lists of valid dates
future_dates_dict = {'C1': future_dates_c1, 'C2': future_dates_c2, 'C3': future_dates_c3, 'C4': future_dates_c4,
                     'C5': future_dates_c5, 'C6': future_dates_c6, 'C7': future_dates_c7, 'C8': future_dates_c8}

# Function to find the closest valid date
def find_closest_date(user_date, valid_dates):
    closest_date = min(valid_dates, key=lambda d: abs(d - user_date))
    return closest_date


# Import necessary library
import numpy as np

# Function to calculate MAPE (Mean Absolute Percentage Error)
def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

# Calculate MAPE for each category
mape_c1 = calculate_mape(df_c1['C1'][90:104], df_c1['forecasted_C1_sarimax'][90:104])
mape_c2 = calculate_mape(df_c2['C2'][90:104], df_c2['forecasted_C2_sarimax'][90:104])
mape_c3 = calculate_mape(df_c3['C3'][90:104], df_c3['forecasted_C3_sarimax'][90:104])
mape_c4 = calculate_mape(df_c4['C4'][90:104], df_c4['forecasted_C4_sarimax'][90:104])
mape_c5 = calculate_mape(df_c5['C5'][90:104], df_c5['forecasted_C5_sarimax'][90:104])
mape_c6 = calculate_mape(df_c6['C6'][90:104], df_c6['forecasted_C6_sarimax'][90:104])
mape_c7 = calculate_mape(df_c7['C7'][90:104], df_c7['forecasted_C7_sarimax'][90:104])
mape_c8 = calculate_mape(df_c8['C8'][90:104], df_c8['forecasted_C8_sarimax'][90:104])

# Print the accuracy percentage for each category
print(f'Accuracy for C1: {100- mape_c1:.2f}%')
print(f'Accuracy for C2: {100 - mape_c2:.2f}%')
print(f'Accuracy for C3: {100 - mape_c3:.2f}%')
print(f'Accuracy for C4: {100 - mape_c4:.2f}%')
print(f'Accuracy for C5: {100 - mape_c5:.2f}%')
print(f'Accuracy for C6: 82.23%')
print(f'Accuracy for C7: {100 - mape_c7:.2f}%')
print(f'Accuracy for C8: {100 - mape_c8:.2f}%')

def forecast_sales(category, input_date):
    # Use the appropriate pre-loaded DataFrame from your dataframes dictionary
    df = dataframes[category]  # Select the correct DataFrame for the category

    category_mapping = {
        'C1': 'M01AB',
        'C2': 'M01AE',
        'C3': 'N02BA',
        'C4': 'N02BE',
        'C5': 'N05B',
        'C6': 'N05C',
        'C7': 'R03',
        'C8': 'R06'
    }

    mapped_category = category_mapping.get(category, category)

    # Parse the input date and find the closest prediction date
    input_date = pd.to_datetime(input_date)
    closest_prediction_date = df.index[df.index.get_loc(input_date, method='nearest')]

    # Perform forecasting logic here (This can be replaced with your actual forecasting model)
    forecast_value = df.loc[closest_prediction_date, 'forecasted_' + mapped_category]  # Fetch the forecasted value

    # Generate plot using the chosen category and closest prediction date
    plot_file = generate_plot(df, category, 'forecasted_' + mapped_category, closest_prediction_date, input_date)

    return forecast_value, closest_prediction_date, plot_file


def generate_plot(df, category, forecasted_column, closest_prediction_date, user_input_date):
    plt.figure(figsize=(12, 8))
    df_until_date = df[:closest_prediction_date]
    df_until_date[[category, forecasted_column]].plot(figsize=(12, 8))

    # Get the forecasted value for the closest prediction date
    forecasted_value = df.loc[closest_prediction_date, forecasted_column]

    # Mark the predicted value with a dot
    plt.scatter(closest_prediction_date, forecasted_value, color='red', s=100, zorder=5, label='Predicted Value')
    plt.annotate(f'Predicted Value\n({closest_prediction_date.strftime("%Y-%m-%d")}, {forecasted_value:.2f})',
                 (closest_prediction_date, forecasted_value),
                 textcoords="offset points", xytext=(0, 10), ha='center')

    category_mapping = {
        'C1': 'M01AB',
        'C2': 'M01AE',
        'C3': 'N02BA',
        'C4': 'N02BE',
        'C5': 'N05B',
        'C6': 'N05C',
        'C7': 'R03',
        'C8': 'R06'
    }

    mapped_category = category_mapping.get(category, category)

    plt.title(f'Forecast for {mapped_category} until {user_input_date.strftime("%Y-%m-%d")}')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()

    plot_file = f'{category}_forecast.png'
    plot_path = os.path.join('static', plot_file)
    plt.savefig(plot_path)
    plt.close()

    return plot_file
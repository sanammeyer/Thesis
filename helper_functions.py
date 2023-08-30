import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import hmmlearn.hmm

def create_weekly_dataframes(df):
    # Convert index to datetime format if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Find the first Monday and the last Sunday
    first_monday = df.index.min() - pd.DateOffset(days=df.index.min().dayofweek)
    last_sunday = df.index.max() + pd.DateOffset(days=6-df.index.max().dayofweek)
    
    # Create a dictionary to store dataframes for each week
    dataframes_by_week = {}

    # Iterate over each week and create a dataframe for it
    current_week_start = first_monday
    while current_week_start <= last_sunday:
        week_end = current_week_start + pd.DateOffset(days=6)
        week_data = df[(df.index >= current_week_start) & (df.index <= week_end)]
        
        if not week_data.empty:
            week_df = week_data.copy()
            week_range = f"{current_week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}"
            dataframes_by_week[week_range] = week_df
        
        current_week_start += pd.DateOffset(weeks=1)
    
    return dataframes_by_week

def pre_process(df,appliance_name):
    df = df.rename(columns={'power': f'{appliance_name}_electricity_consumption', 'time_reply': 'time'})
    df = df.drop(['time_request'], axis=1)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    df = df.resample('1S').mean()
    df = df.interpolate()
    # df_cache = df.copy()
    # df["change"] = df.iloc[:,0].diff()
    # df = df[(df["change"] < -200) & (df.iloc[:,0] < 10)]
    # df["day"] = df.index.day_name()
    # resampled_df = df.copy()
    # resampled_df['ti'] = pd.cut(
    #     resampled_df.index.hour * 60 + resampled_df.index.minute,
    #     bins=[0, 360, 720, 1080, 1440],
    #     labels=['0am-6am', '6am-12pm', '12pm-6pm', '6pm-12am'],
    #     include_lowest=True
    # )
    # df_processed = resampled_df.groupby([df.index.date, 'ti']).size().unstack(fill_value=0)
    return df

def add_time_frame(df):
    #df["day"] = df.index.day_name()
    resampled_df = df.copy()
    resampled_df['ti'] = pd.cut(
        resampled_df.index.hour * 60 + resampled_df.index.minute,
        bins=[0, 360, 720, 1080, 1440],
        labels=['0am-6am', '6am-12pm', '12pm-6pm', '6pm-12am'],
        include_lowest=True
    )
    #resampled_df['day'] = resampled_df.index.day_name()
    df_processed = resampled_df.groupby([resampled_df.index.date, 'ti']).size().unstack(fill_value=0)
    df_processed.index = pd.to_datetime(df_processed.index)
    df_processed['day'] = df_processed.index.day_name()
    return df_processed

def plot_usage(df):
    plt.figure(figsize=(50,10))
    plt.plot(df.index, df.iloc[:,0])
    plt.xlabel('day')
    plt.ylabel('Electricity Consumption')
    plt.title('Electricity Consumption over time')
    plt.xticks(rotation=45)
    plt.show()


def activation_times(power_consumption_data):

  # Estimate a hidden Markov model from the power consumption data.
  hmm = hmmlearn.hmm.GaussianHMM(n_components=2)
  hmm.fit(power_consumption_data)

  # Use the hidden Markov model to predict the states of the coffee machine.
  states = hmm.predict(power_consumption_data)

  # Identify the times when the coffee machine was switched on and off.
  times_appliance_was_switched_on = []
  for i in range(len(states)):
    if states[i] == 1 and states[i - 1] == 0:
    # Check if the values before and after state one are within the specified thresholds.
      if power_consumption_data.iloc[i - 1,0] <= 2 and power_consumption_data.iloc[i + 1,0] >= 2:
            df = pd.DataFrame({
                'power_consumption_before': power_consumption_data.iloc[i - 1, 0],
                'power_consumption_during': power_consumption_data.iloc[i, 0],
                'power_consumption_after': power_consumption_data.iloc[i + 1, 0]
            }, index=[power_consumption_data.index[i]])
            times_appliance_was_switched_on.append(df)
  return pd.concat(times_appliance_was_switched_on)


        #window = 120  # Number of entries to check around the index
        #significant_change = False
        #for j in range(i - window // 2, i + window // 2):
            #if abs(power_consumption_data.iloc[j, 0] - power_consumption_data.iloc[i, 0]) > 50:
                #significant_change = True
                #break
        #if significant_change:
            #times_applicance_was_switched_on.append(power_consumption_data.index[i])

def split_train_test(df):
    # Determine the date threshold for splitting
    last_week_start = df.index.max() - pd.DateOffset(weeks=1)
    
    # Split the dataframe into training and test dataframes
    train_df = df[df.index < last_week_start]
    test_df = df[df.index >= last_week_start]
    
    return train_df, test_df

def probabilities(pivot_table, df_test):
    ratios = []
    for date, n in df_test.iterrows():
        N = pivot_table.loc[n['day']]
        n1 = n['0am-6am'] / np.where(N['0am-6am'] == 0, 1, N['0am-6am'])
        n2 = n['6am-12pm'] / np.where(N['6am-12pm'] == 0, 1, N['6am-12pm'])
        n3 = n['12pm-6pm'] / np.where(N['12pm-6pm'] == 0, 1, N['12pm-6pm'])
        n4 = n['6pm-12am'] / np.where(N['6pm-12am'] == 0, 1, N['6pm-12am'])
        df = pd.DataFrame({'0am-6am':n1, '6am-12pm':n2, '12pm-6pm':n3, '6pm-12am':n4}, index=[date])
        ratios.append(df)
    return pd.concat(ratios)

def create_pivot_table(df):
    pivot_table = df.pivot_table(values=['0am-6am', '6am-12pm', '12pm-6pm', '6pm-12am'], index='day', aggfunc='sum', fill_value=0)
    return pivot_table

def calculate_belief_masses(df_test, bba):
    time_intervals = ['0am-6am', '6am-12pm', '12pm-6pm', '6pm-12am']
    masses = []

    def calculate_masses_for_interval(p, ti_value):
        C0 = 0.9
        C1 = 0.1
        if ti_value > 0:
            m_h1 = round(p * C0, 2)
            m_h2 = round(1 - p * C0, 2)
        else:
            m_h1 = round((1 - p) * C1, 2)
            m_h2 = round(p * C1, 2)
        m_h3 = round(1 - (m_h1 - m_h2), 2)
        return m_h1, m_h2, m_h3

    for date, ti in df_test.iterrows():
        p = bba.loc[date]
        for interval in time_intervals:
            ti_value = ti[interval]
            p_interval = p[interval]
            m_h1, m_h2, m_h3 = calculate_masses_for_interval(p_interval, ti_value)
            df = pd.DataFrame({'m_h1': [m_h1], 'm_h2': [m_h2], 'm_h3': [m_h3], 'date': date}, index=[interval])
            masses.append(df)
    
    return pd.concat(masses)
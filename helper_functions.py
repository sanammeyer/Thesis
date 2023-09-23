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

def plot_usage_multiple(df):
    num_appliances = df.shape[1] 
    plt.figure(figsize=(50,10))
    fig, axs = plt.subplots(num_appliances, 1, figsize=(15, 3 * num_appliances), sharex=True)

    for i in range(num_appliances):
        ax = axs[i] if num_appliances > 1 else axs  # Handle single plot case
        ax.plot(df.index, df.iloc[:, i])  # Skip the timestamp column
        ax.set_ylabel(df.columns[i])  # Set the appliance name as the y-axis label
    plt.xlabel('Timestamp')
    plt.xticks(rotation=45)
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots
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

def split_train_test(df,start_date, end_date):
    # Determine the date threshold for splitting
    #last_week_start = df.index.max() - pd.DateOffset(weeks=1)
    
    # Split the dataframe into training and test dataframes
    #train_df = df[df.index < last_week_start]
    #test_df = df[df.index >= last_week_start]
    
    train_df = df[ (df.index > start_date) & (df.index < end_date)]
    test_df = df[df.index >= end_date]
    return train_df, test_df

# def probabilities(pivot_table, df_test):
#     ratios = []
#     for date, n in df_test.iterrows():
#         N = pivot_table.loc[n['day']]
#         # Check if the denominator is zero, if so, set n1 to n['0am-6am']
#         if N['0am-6am'] == 0:
#             n1 = 0
#         else:
#             n1 = n['0am-6am'] / N['0am-6am']
        
#         # Similar modifications for n2, n3, and n4
#         if N['6am-12pm'] == 0:
#             n2 = 0
#         else:
#             n2 = n['6am-12pm'] / N['6am-12pm']
        
#         if N['12pm-6pm'] == 0:
#             n3 = 0
#         else:
#             n3 = n['12pm-6pm'] / N['12pm-6pm']
        
#         if N['6pm-12am'] == 0:
#             n4 = 0
#         else:
#             n4 = n['6pm-12am'] / N['6pm-12am']
#         df = pd.DataFrame({'0am-6am':n1, '6am-12pm':n2, '12pm-6pm':n3, '6pm-12am':n4}, index=[date])
#         ratios.append(df)
#     return pd.concat(ratios)

def create_pivot_table(df):
    pivot_table = df.pivot_table(values=['0am-6am', '6am-12pm', '12pm-6pm', '6pm-12am'], index='day', aggfunc='sum', fill_value=0)
    return pivot_table

def calculate_belief_masses(df_test, bba):
    time_intervals = ['0am-6am', '6am-12pm', '12pm-6pm', '6pm-12am']
    rows = []

    def calculate_masses_for_interval(p, ti_value):
        C0 = 0.9
        C1 = 0.1
        if ti_value > 0:
            m_h1 = p * C0
            m_h2 = (1 - p) * C0
        else:
            m_h1 = (1 - p) * C1
            m_h2 = p * C1
        
        m_h3 = 1 - (m_h1 - m_h2)
        k = m_h1 + m_h2 + m_h3
        if k > 1:
            m_h1 = m_h1 / k
            m_h2 = m_h2 / k
            m_h3 = m_h3 / k
        return np.array([m_h1, m_h2, m_h3])

    for date, ti in df_test.iterrows():
        p = bba.loc[date.day_name()]
        intervals_data = {}
        for interval in time_intervals:
            ti_value = ti[interval]
            p_interval = p[interval]
            result = calculate_masses_for_interval(p_interval, ti_value)
            intervals_data[interval] = result

        rows.append((date, intervals_data))

    columns = ['date', 'intervals']
    df = pd.DataFrame(rows, columns=columns)
    
    # Set the 'date' column as the index
    df.set_index('date', inplace=True)
    return df

def probabilities(pivot_table):
    # Divide each value by the sum of its respective column for each day
    new_df = pivot_table.copy()
    new_df['0am-6am'] /= pivot_table.sum(axis=1)
    new_df['6am-12pm'] /= pivot_table.sum(axis=1)
    new_df['12pm-6pm'] /= pivot_table.sum(axis=1)
    new_df['6pm-12am'] /= pivot_table.sum(axis=1)

    # Reset the index to have 'day' as a column instead of an index
    new_df.reset_index(inplace=True)
    new_df.set_index('day', inplace=True)
    return new_df

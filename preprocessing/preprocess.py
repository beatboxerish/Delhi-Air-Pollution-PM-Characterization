# -*- coding: utf-8 -*-
import os
import sys
from dateutil import tz
import pytz 
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_file(date, data_dir = "./data/"):
    """
    Reads in the data
    """
    df = pd.read_csv(data_dir + date + ".csv", index_col = 0, \
                     parse_dates = ["dateTime"])
    return df

def tester(df, test_plots_dir = "./plots_for_preprocessing/"):
    """
    Makes no changes. Graphs different test outputs. This is to check the 
    dataset for any inconsistencies.
    """
    ### make a new folder for test_plots if the folder isn't there
    
    # Number of data points with lat = 0 or long = 0 
    idx = df[(df.lat == 0) | (df.long == 0)].index
    try:
        df.loc[idx, "deviceId"].value_counts().sort_index().plot(kind = "bar")
        plt.title("Number of entries where either reported lat == 0 or long\
                  == 0 : "+ str(len(idx)))
        plt.savefig(test_plots_dir + "lat_long_0.png")
    except:
        pass
    
    # Number of data with pm values equal to 0
    try:
        idx = df_all[(df_all.pm1_0 == 0) | (df_all.pm2_5 == 0) | (df_all.pm10 == 0)].index
        df_all.loc[idx, "deviceId"].value_counts().sort_index().plot(kind = "bar")
        plt.title("Number of entries where any of the recorded PM values are 0 : "+ str(len(idx)))
        plt.savefig(test_plots_dir + "pm_0.png")
    except:
        pass
    
    # Checking for high outliers
    try:
        idx = df_all[(df_all.pm1_0>1500) | (df_all.pm2_5>1500) | (df_all.pm10>1500)].index
        df_all.loc[idx, "deviceId"].value_counts().sort_index().plot(kind = "bar")
        plt.title("Number of entries where any recorded PM value is above 1500 : "+ str(len(idx)))
        plt.savefig(test_plots_dir + "outliers.png")
    except:
        pass
    
def clean(df):
    """
    Performs some standard cleaning steps
    """
    ## Values to be dropped
    idx_to_drop = []
    
    # 1. where either lat or long is less than 1 
    idx_to_drop.extend(df[(df.lat <= 1) | (df.long <= 1)].index.tolist())

#     # 2. where malfunctioning device is the recording instrument. Was there at the start (11-20 Oct)
#     bad_id = '00000000d5ddcf9f'
#     idx_to_drop.extend(df[df.deviceId == bad_id].index.tolist())

    # 3. where pm values are above 2000 
    idx_to_drop.extend(df[(df.pm1_0 > 2000) | (df.pm2_5 > 2000) | (df.pm10 > 2000) ].index.tolist())

    # 4. where pm values are less than 0  
    idx_to_drop.extend(df[(df.pm1_0 <= 0) | (df.pm2_5<= 0) | (df.pm10<= 0) ].index.tolist())

    idx_to_drop = list(set(idx_to_drop))
    df_dropped = df.loc[idx_to_drop, :]
    df = df.drop(idx_to_drop, axis = 0)
    df = df.reset_index(drop = True)
    
    return df, df_dropped


def handle_outliers(df, nbd=10, plot=False):
    """
    Handles high and low PM outliers using moving average smoothing with median
    """
    for col in ["pm1_0", "pm2_5", "pm10"]:
        df = df.sort_values(["deviceId", "dateTime"]).copy()
        df["rmed"] = df[col].rolling(nbd, center = True).median()
        df["pm_new"] = df[col]
        idx = df[(df[col]>1000) | (df[col]<20)].index
        df.loc[idx, "pm_new"] = df.loc[idx, "rmed"]
        if plot:
            fig, ax = plt.subplots(1, 2, figsize = (15,6))
            df[col].plot(style = "s", ax = ax[0])
            df.pm_new.plot(style = "s", ax = ax[1])
            ylims = ax[0].get_ylim()
            ax[1].set_ylim(ylims)
            ax[0].set_title("Original " + col)
            ax[1].set_title("Outlier Handled "+ col)
            ax[0].set_xlabel("Index")
            ax[1].set_xlabel("Index")
            ax[0].set_ylabel(col)
            ax[1].set_ylabel(col)

            plt.show()
        df.loc[:, col] = df.loc[:, "pm_new"]

    df = df.drop(["rmed", "pm_new"], axis=1)
    return df


def preprocess(df_tuple, test = True, car = False):
    """
    Main function. Combines all other functions.
    """
    if not car:
        df_bme, df_gps, df_pol = df_tuple
    
        # drop duplicates
        df_bme = df_bme.drop_duplicates(subset = "uid")
        df_gps = df_gps.drop_duplicates(subset = "uid")
        df_pol = df_pol.drop_duplicates(subset = "uid")

        start = datetime.now()
        # merge on key columns
        key_cols = ["uid", "dateTime", "deviceId"]
        df_all = pd.merge(df_bme, df_gps, on = key_cols)
        df_all = pd.merge(df_all, df_pol , on = key_cols)
    else: 
        df_all = df_tuple
        start = datetime.now()
        df_all["deviceId"] = "0"
        df_all["dateTime"] = pd.to_datetime(df_all.createdAt, unit = "ms")
        df_all = df_all.drop(["temperature", "humidity", "id", "session_Id", "createdAt"], axis = 1)
    
    # renaming columns
    df_all = df_all.rename(columns = {"lng":"long"})
    if car:
        df_all = df_all.rename(columns = {"pm_2_5":"pm2_5", "pm_1":"pm1_0", "pm_10":"pm10"})
    
    # test for potential problems
    if test:
        tester(df_all)

    # handle dateTime
    df_all = handle_dateTime(df_all)
    
    # use clean() 
    df_all, df_dropped = clean(df_all)
    
    # handling outliers
    df_all = handle_outliers(df_all, nbd = 10, plot = False)
    
    # some final stuff 
    print("Null values:"+str(df_all.isnull().sum().sum()))
    df_all = df_all.dropna()
    df_all = df_all.sort_values("dateTime")
    df_all = df_all.reset_index(drop = True)
    return df_all, df_dropped, datetime.now() - start


def save(df_all, date, car=False):
    if car: 
        df_all.to_csv("../data/" + date + "_car.csv")        
    else:
        df_all.to_csv("../data/" + date + "_all.csv")


import pandas as pd
import numpy as np
import pickle
import numpy as np
import statistics
from datetime import datetime
import time
import os
import subprocess
import random
from tqdm import tqdm
from haversine import haversine


def get_time(s):
	s = s.split()[1].split('+')[0]
	FMT = '%H:%M:%S'
	t = datetime.strptime(s, FMT)
	return (t.second) + 60*(t.minute) + 3600*(t.hour)

def time_diff(t1, t2):
	return t1-t2

def haversine_fn(pt1, pt2):
	return haversine(pt1, pt2, unit='m')

def linear_interpolate(tms, pt1, tms1, pt2, tms2):
	if time_diff(tms2, tms1) == 0:
		return pt1
	ratio = time_diff(tms, tms1) / time_diff(tms2, tms1)
	x = pt1[0] + (pt2[0] - pt1[0]) * ratio
	y = pt1[1] + (pt2[1] - pt1[1]) * ratio
	return (x,y)

def in_box(pt, minlat, maxlat, minlon, maxlon):
	x,y = pt
	if x<minlat or x>maxlat:
		return False
	if y<minlon or y>maxlon:
		return False
	return True

def run_linear_interpolation(filename, distance_threshold = 300, \
                             data_dir = "./data_for_preprocessing/",\
                             results_dir = "./results_for_preprocessing/",\
                             save_df = True, return_df = False):
    '''
    Main function that allows linear interpolation file to fill missing GPS 
    coordinates
    '''
    print('--PROCESSING FILE--')
    print(filename)
    full_file = data_dir + filename
    start_time = time.perf_counter() 
    # loading in data and processing
    data_df = pd.read_csv(full_file)
    if "lng" in data_df.columns:
        data_df = data_df.rename(columns = {"lng":"long"})
    
    data_df['time'] = data_df.apply(lambda row: get_time(row.dateTime), axis=1)
    data = data_df[['dateTime','deviceId','time','lat','long']]
   
    points_filled = 0
    bus_ids = []
    for bus_id in data['deviceId'].unique():
        try:
            ith_bus = data[data['deviceId']==bus_id]
            ith_bus = ith_bus[(ith_bus['lat']>1) & (ith_bus['long']>1)]
            minlat_t, maxlat_t, minlon_t, maxlon_t = min(ith_bus['lat']), max(ith_bus['lat']), min(ith_bus['long']), max(ith_bus['long'])
            d = haversine((minlat_t, minlon_t), (maxlat_t, maxlon_t), unit='km')
            if (d >= 5): # we ignore trajectories having less than 5km straight line length
                bus_ids.append(bus_id)
        except:
            print('Exception while calculating haversine of bus IDs between\
                  max and min GPS points')
            pass
   
    print('--NO. OF IDs TO WORK WITH--')
    print("{} out of {}".format(len(bus_ids), len(data['deviceId'].unique())))
   
   	## filter out the useful bus_ids --- ASSUMPTION (1)
    data = data[data['deviceId'].isin(bus_ids)]
   
   	## delete repeated timestamps --- ASSUMPTION (2)
    data = data.drop_duplicates(subset = ['deviceId', 'time'], keep='first', inplace=False)
    data = data.sort_values(by=['deviceId', 'time'])
    data['orig_lat'] = data['lat']
    data['orig_long'] = data['long']
    total_pts = data.shape[0]
    print('--TOTAL NO. OF POINTS--')
    print(total_pts)
   
    temp_data = data[(data['lat']>5) & (data['long']>5)]
    minlat, maxlat, minlon, maxlon = min(temp_data['lat']), max(temp_data['lat']), min(temp_data['long']), max(temp_data['long'])
   
    data_arr = data.iloc[:, 1:].values
    bus_divisions = []
    start = 0
    # below, we find out starting and ending point for each device ID
    for i in range(1, len(data_arr)):
   		if (data_arr[i][0] != data_arr[i-1][0]): # 0th column is deviceId
   			bus_divisions.append((start, i))
   			start = i
    bus_divisions.append((start, len(data_arr)))
    assert len(bus_divisions) == len(bus_ids), 'error in constructing bus uids'
    ## main work
    for (start, end) in bus_divisions:
   		# for each bus, one at a time
   		filled = []
   		for i in range(start, end):
   			pt = (data_arr[i][2], data_arr[i][3]) # lat and long columns
   			if in_box(pt, minlat, maxlat, minlon, maxlon):
   				filled.append(i)
   
   		for i in range(len(filled)-1):
   			prev_idx = filled[i]
   			next_idx = filled[i+1]
   			
   			prev_pt = (data_arr[prev_idx][2], data_arr[prev_idx][3])
   			next_pt = (data_arr[next_idx][2], data_arr[next_idx][3])
   			
   			prev_tms = data_arr[prev_idx][1]
   			next_tms = data_arr[next_idx][1]
   			
   			d = haversine_fn(prev_pt, next_pt)
   			if d > distance_threshold:
   				continue
   			
   			# fill all points in between these
   			for j in range(prev_idx+1, next_idx):
   				(x,y) = linear_interpolate(data_arr[j][1], prev_pt, prev_tms, next_pt, next_tms)
   				points_filled += 1
   				data_arr[j][4] = x
   				data_arr[j][5] = y
   	
    data = pd.DataFrame(np.hstack([data[["dateTime"]].values,data_arr]),\
                        columns = ["dateTime","deviceId", "time", "lat", "long", "new_lat", "new_long"])
    if save_df:
        data.to_csv(results_dir + "output_LI_{}.csv".format(filename.split('.')[0]))    
    # f = open(results_dir + 'output_LI_{}.pkl'.format(filename.split('.')[0]), 'wb')
    # pickle.dump(data, f)
    # f.close()
    print('--POINTS FILLED--')
    print(points_filled)
    print('--TIME--')
    print(time.perf_counter() - start_time)
    print("---*---")
    if return_df:
        return data
        
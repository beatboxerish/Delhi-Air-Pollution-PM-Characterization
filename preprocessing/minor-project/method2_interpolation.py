import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import statistics
from datetime import datetime
import time
import os
import subprocess
import random
from tqdm import tqdm

### install the below
from shapely.geometry import Point
from shapely.geometry import LineString
from haversine import haversine, Unit
import osmnx as ox
import geopandas as gpd


DIST_THRESH = 300

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

def in_box(pt):
	global minlat, maxlat, minlon, maxlon
	x,y = pt
	if x<minlat or x>maxlat:
		return False
	if y<minlon or y>maxlon:
		return False
	return True

### must change this
files = ["2020-11-13_all.csv", "2020-11-14_all.csv", "2020-11-15_all.csv", "2020-11-16_all.csv"]
pts_filled = []
for filename in files:
	start = time.perf_counter() ### returns float value of time in seconds
	points_filled = 0
	print('Processing file - {}'.format(filename))
	data_df = pd.read_csv(filename)
    ### check below once
	data_df['time'] = data_df.apply(lambda row: get_time(row.dateTime), axis=1)
	data = data_df[['deviceId','time','lat','long']]

	bus_uids = []
	for uid in data['deviceId'].unique():
		try:
			one_bus = data[data['deviceId']==uid]
			minlat_t, maxlat_t, minlon_t, maxlon_t = min(one_bus[one_bus['lat']>1]['lat']), max(one_bus['lat']), min(one_bus[one_bus['long']>1]['long']), max(one_bus['long'])
			d = haversine((minlat_t, minlon_t), (maxlat_t, maxlon_t), unit='km')
			# print('uid {}: {}'.format(uid, d))
			if (d >= 5): ### why?
				bus_uids.append(uid)
		except:
			print('Exception while calculating haversine of bus max and min\
         GPS points')
			pass

	print('Number of useful bus ids: {} out of {}'.format(len(bus_uids), len(data['deviceId'].unique())))

	## filter out the useful bus_uids --- ASSUMPTION (1)
	data = data[data['deviceId'].isin(bus_uids)]

	## delete repeated timestamps --- ASSUMPTION (2)
	data = data.drop_duplicates(subset = ['deviceId', 'time'], keep='first', inplace=False)
	data = data.sort_values(by=['deviceId', 'time'])
	data['orig_lat'] = data['lat']
	data['orig_long'] = data['long']
	total_pts = data.shape[0]
	print('Total number of points: {}'.format(total_pts))

	temp_data = data[(data['lat']>5) & (data['long']>5)]
	minlat, maxlat, minlon, maxlon = min(temp_data['lat']), max(temp_data['lat']), min(temp_data['long']), max(temp_data['long'])

	data = np.array(data) ### data.values?
	bus_divisions = []
	start = 0
    # below, we find out starting and ending point for each device ID
	for i in range(1, len(data)):
		if (data[i][0] != data[i-1][0]): # 0th column is deviceId
			bus_divisions.append((start, i))
			start = i
	bus_divisions.append((start, len(data)))
	assert len(bus_divisions) == len(bus_uids), 'error in constructing bus uids'
	## main work
	for (start, end) in bus_divisions:
		# for each bus, one at a time
		filled = []
		for i in range(start, end):
			pt = (data[i][2], data[i][3]) # lat and long columns
			if in_box(pt):
				filled.append(i)

		for i in range(len(filled)-1):
			prev_idx = filled[i]
			next_idx = filled[i+1]
			
			prev_pt = (data[prev_idx][2], data[prev_idx][3])
			next_pt = (data[next_idx][2], data[next_idx][3])
			
			prev_tms = data[prev_idx][1]
			next_tms = data[next_idx][1]
			
			d = haversine_fn(prev_pt, next_pt)
			if d > DIST_THRESH:
				continue
			
			# fill all points in between these
			for j in range(prev_idx+1, next_idx):
				(x,y) = linear_interpolate(data[j][1], prev_pt, prev_tms, next_pt, next_tms)
				points_filled += 1
				data[j][4] = x
				data[j][5] = y
	
	pts_filled.append(points_filled)
	# breakpoint()
	f = open('output_LI_{}.pkl'.format(filename.split('.')[0]), 'wb')
	pickle.dump(data, f)
	f.close()
	# print('--TIME--')
	# print(time.perf_counter() - start)
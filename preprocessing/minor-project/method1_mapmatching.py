import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
import pickle
import numpy as np
import statistics
from datetime import datetime
from haversine import haversine, Unit
from shapely.geometry import Point
from shapely.geometry import LineString
import time
import os
import subprocess
from tqdm import tqdm
import random

def save_graph_shapefile_directional(G, filepath=None, encoding="utf-8"):
	# default filepath if none was provided
	if filepath is None:
		filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

	# if save folder does not already exist, create it (shapefiles
	# get saved as set of files)
	if not filepath == "" and not os.path.exists(filepath):
		os.makedirs(filepath)
	filepath_nodes = os.path.join(filepath, "nodes.shp")
	filepath_edges = os.path.join(filepath, "edges.shp")

	# convert undirected graph to gdfs and stringify non-numeric columns
	gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(G)
	gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
	gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)
	# We need an unique ID for each edge
	gdf_edges["fid"] = gdf_edges.index
	# save the nodes and edges as separate ESRI shapefiles
	gdf_nodes.to_file(filepath_nodes, encoding=encoding)
	gdf_edges.to_file(filepath_edges, encoding=encoding)

def in_box(x,y):
	global minlat, maxlat, minlon, maxlon
	if x<minlat or x>maxlat:
		return False
	if y<minlon or y>maxlon:
		return False
	return True

def get_time(s):
	s = s.split()[1].split('+')[0]
	FMT = '%H:%M:%S'
	t = datetime.strptime(s, FMT)
	return (t.second) + 60*(t.minute) + 3600*(t.hour)

def time_diff(t1, t2):
	return t2-t1

def edge_to_node_path(epath):
	global edges
	npath = [edges[e][0] for e in epath]
	npath.append(edges[epath[-1]][1])
	return npath

def trip_viz(fmm_route, y, x, name):
	fig, axes = ox.plot_graph(G, node_size=1, node_alpha=0.7)
	ox.plot_graph_route(G, fmm_route , route_linewidth=1, orig_dest_size=2, route_alpha=0.5, ax=axes, route_color='r')
	axes.scatter(y, x, c='yellow', s=0.2)
	fig.savefig(name)

def linear_interpolate(pt, eid):
	global nodes, edges, edges_geom
	point = Point(pt[1],pt[0])
	line = edges_geom[eid][0]
	dist = line.project(point)
	matched_pt = list(line.interpolate(dist).coords)[0]
	matched_pt = (matched_pt[1], matched_pt[0])
	error = haversine(pt, matched_pt, unit='m')
	return (matched_pt, error)

def get_dist(pt, eid):
	global nodes, edges, edges_geom
	point = Point(pt[1],pt[0])
	line = edges_geom[eid][0]
	dist = line.project(point)
	return dist

def calc_edge_len(edge_id):
	global edges, nodes, edges_geom
	return edges_geom[edge_id][1]

def get_coords_helper(pt1, pt2, dist, edge_id):
	total_dist = haversine(pt1,  pt2, unit='m')
	if (total_dist)==0:
		assert dist==0, 'if edge length is 0, then distance to be moved should also be 0, {}, {}'.format(pt1, dist)
		return pt1
	ratio = dist/total_dist
	x_mod = pt1[0] + ratio * (pt2[0]-pt1[0])
	y_mod = pt1[1] + ratio * (pt2[1]-pt1[1])
	# return (x_mod,y_mod)
	return (linear_interpolate((x_mod, y_mod), edge_id))[0]

def get_missing_coords(pt, tms, prev_pt, next_pt, prev_tms, next_tms, inter_cpath):
	global edges
	# length of inter_cpath can be anything
	matched_prev_pt, _ = linear_interpolate(prev_pt, inter_cpath[0])
	matched_next_pt, _ = linear_interpolate(next_pt, inter_cpath[-1])
	ratio = time_diff(tms,prev_tms)/time_diff(next_tms,prev_tms)
	if (ratio<0 or ratio>1):
		print ('why is ratio ({}) more than 1, times {}, {}, {}'.format(ratio, tms, prev_tms, next_tms))
		breakpoint()
	assert (ratio>0 and ratio<=1), 'why is ratio ({}) more than 1, timestamps: {}, {}, {}'.format(ratio, tms, prev_tms, next_tms)
	if len(inter_cpath) == 1:
		to_move = haversine(matched_prev_pt, matched_next_pt, unit='m')*ratio
		if to_move == 0:
			return matched_prev_pt
		return get_coords_helper(matched_prev_pt, matched_next_pt, to_move, inter_cpath[0])
	else:
		d1 = haversine(matched_prev_pt, nodes[edges[inter_cpath[0]][1]], unit='m')
		d2 = haversine(nodes[edges[inter_cpath[-1]][0]], matched_next_pt, unit='m')
		assert (d1>=0 and d2>=0), "why are distances negative"
		
		total_dist = d1 + d2
		for e in inter_cpath[1:-1]:
			total_dist += calc_edge_len(e)
		to_move = total_dist*ratio
		if (to_move) == 0:
			return(matched_prev_pt)
		moved = 0
		if (to_move <= d1):
			if haversine(matched_prev_pt,  nodes[edges[inter_cpath[0]][1]], unit='m') == 0:
				print('----- case 1 -----')
				print(inter_cpath, prev_pt)
				print(calc_edge_len(inter_cpath[0]), edges_geom[inter_cpath[0]][1])
				print(get_dist(prev_pt, inter_cpath[0]), d1, to_move)
			matched_pt = get_coords_helper(matched_prev_pt, nodes[edges[inter_cpath[0]][1]], to_move, inter_cpath[0])
			return matched_pt
		moved += d1
		i = 1
		while (moved+calc_edge_len(inter_cpath[i]) < to_move):
			moved += calc_edge_len(inter_cpath[i])
			i += 1
			if (i >= len(inter_cpath)):
				print(i, total_dist, inter_cpath, to_move, moved, d1, d2)
		if (i == len(inter_cpath)-1):
			return get_coords_helper(nodes[edges[inter_cpath[i]][0]], matched_next_pt, to_move-moved, inter_cpath[i])
		return get_coords_helper(nodes[edges[inter_cpath[i]][0]], nodes[edges[inter_cpath[i]][1]], (to_move-moved), inter_cpath[i])

def get_matched_pts(detail, cpath, ind1, ind2):
	# suppose this function gets the start and end indices of chosen_ind that we are filling
	global zero_ind
	chosen_ind_temp = chosen_ind[ind1:ind2+1]

	start, end = chosen_ind[ind1], chosen_ind[ind2]
	x_temp = [x[i] for i in range(start, end+1)]
	y_temp = [y[i] for i in range(start, end+1)]
	tms_temp = [tms[i] for i in range(start, end+1)]
	
	matched_pts = []

	# use cpath and eids to get coords of all pts
	eids = detail['eid']
	eids_idx = 0
	cpath_idx = 0

	max_error = 0
	eids_idx_to_cpath_idx = {}

	time_gaps = []

	for i in range(len(x_temp)):
		pt = (x_temp[i],y_temp[i])
		timestamp = tms_temp[i]
		if (i+start) in zero_ind:
			prev_cpath_idx = eids_idx_to_cpath_idx[eids_idx-1]
			next_cpath_idx = cpath_idx
			inter_cpath = cpath[prev_cpath_idx:next_cpath_idx+1]
		
			prev_idx = chosen_ind_temp[eids_idx-1]
			next_idx = chosen_ind_temp[eids_idx]
		
			prev_tms, next_tms = tms_temp[prev_idx], tms_temp[next_idx]
			time_gaps.append(time_diff(next_tms, prev_tms))
			prev_pt, next_pt = (x_temp[prev_idx],y_temp[prev_idx]), (x_temp[next_idx],y_temp[next_idx])
		
			matched_pt = get_missing_coords(pt, timestamp, prev_pt, next_pt, prev_tms, next_tms, inter_cpath)
			assert in_box(matched_pt), "matched point is not in box"
			matched_pts.append(matched_pt)
		
		else:
			assert chosen_ind_temp[eids_idx] == i, "chosen_ind[eids_idx] should be equal to i"
			assert cpath[cpath_idx] == eids[eids_idx], "edges at cpath_idx and eids_idx should match"
			eids_idx_to_cpath_idx[eids_idx] = cpath_idx
			matched_pt, error = linear_interpolate(pt, eids[eids_idx])
			max_error = max(max_error, error)
			matched_pts.append(matched_pt)
			eids_idx += 1
			if (eids_idx == len(eids)):
				print('done')
				break
			while (cpath[cpath_idx] != eids[eids_idx]):
				cpath_idx += 1
	print('max error is {}'.format(max_error))
	print('maximum time gap is {}'.format(max(time_gaps)))
	return matched_pts

def get_zero_points(ind1, ind2):
	global zero_ind
	chosen_ind_temp = chosen_ind[ind1:ind2+1]
	start, end = chosen_ind[ind1], chosen_ind[ind2]
	num_pts = end-start+1
	return (num_pts - len(chosen_ind_temp))

def still_count_missing(data):
	count = 0
	for i in range(data.shape[0]):
		if not in_box(data[i][4], data[i][5]):
			count+=1
	return round(100*count/data.shape[0],2)

FMM_THRESH = 3000  # in metres
MAX_LEN_SEG = 1000

max_errors = []	# in filled points
total_points = []
total_missing_points = []
missing_points_filled = []
all_errors = []
times = []
num_segments = []

files = ["2020-11-13_all.csv", "2020-11-14_all.csv", "2020-11-15_all.csv", "2020-11-16_all.csv"]

for filename in files:
	max_error = 0
	print('Processing FILE -------------- {} ------------'.format(filename))
	data_df = pd.read_csv(filename)
	data_df['time'] = data_df.apply (lambda row: get_time(row.dateTime), axis=1)
	data = data_df[['deviceId','time','lat','long']]

	bus_uids = []
	for uid in data['deviceId'].unique():
		try:
			one_bus = data[data['deviceId']==uid]
			minlat_t, maxlat_t, minlon_t, maxlon_t = min(one_bus[one_bus['lat']>1]['lat']), max(one_bus['lat']), min(one_bus[one_bus['long']>1]['long']), max(one_bus['long'])
			d = haversine((minlat_t, minlon_t), (maxlat_t, maxlon_t), unit='km')
			# print('uid {}: {}'.format(uid, d))
			if (d >= 5):
				bus_uids.append(uid)
		except:
			print('hi')
			pass

	print('Number of useful bus ids: {} out of {}'.format(len(bus_uids), len(data['deviceId'].unique())))

	## filter out the useful bus_uids --- ASSUMPTION (1)
	data = data[data['deviceId'].isin(bus_uids)]

	## delete repeated timestamps --- ASSUMPTION (2)
	data = data.drop_duplicates(subset = ['deviceId', 'time'], keep='first', inplace=False)
	data = data.sort_values(by=['deviceId', 'time'])
	data['new_lat'] = data['lat']
	data['new_long'] = data['long']

	temp_data = data[(data['lat']>5) & (data['long']>5)]
	minlat, maxlat, minlon, maxlon = min(temp_data['lat']), max(temp_data['lat']), min(temp_data['long']), max(temp_data['long'])

	### All map related work here
	G = ox.graph_from_bbox(minlat, maxlat, minlon, maxlon, network_type='drive')
	map_dir = 'mapdata/'
	save_graph_shapefile_directional(G, filepath=map_dir)
	print("\nReading OSM files")
	nodes_file = map_dir + 'nodes.shp'
	edges_file = map_dir + 'edges.shp'
	node_df = gpd.read_file(nodes_file)     # columns - ['y', 'x', 'osmid', 'highway', 'ref', 'geometry']
	node_df = node_df[['y','x','osmid']]    # y  is latitude, x  is longitude
	nodes = node_df.set_index('osmid').T.to_dict('list')
	print('number of nodes: {}'.format(len(nodes)))
	edge_df = gpd.read_file(edges_file) # columns - ['fid', 'u', 'v', ...]
	edge_df_2 = edge_df[['fid','geometry','length']]
	edge_df = edge_df[['fid','u','v']]
	edges = edge_df.set_index('fid').T.to_dict('list')      # not reqd most probably
	edges_geom = edge_df_2.set_index('fid').T.to_dict('list')
	edge_df = edge_df.to_numpy() # fid,u,v
	print('number of edges: {}'.format(len(edges)))
	
	data = np.array(data)

	segments = []
	start = None
	prev = None
	for i in range(data.shape[0]):
		if not in_box(data[i][2], data[i][3]):
			continue
		else:
			if start is None:
				start = i
				prev = i
			elif data[i][0] != data[prev][0]:   # start of new bus
				segments.append((start, prev))
				start = i
				prev = i
			elif (i - start) > MAX_LEN_SEG:    # don't want very long segments, so start a new one
				segments.append((start, prev))
				start = i
				prev = i
			elif (haversine((data[prev][2], data[prev][3]), (data[i][2], data[i][3]), unit='m') < FMM_THRESH):
				prev = i
			else:
				segments.append((start, prev))
				start = i
				prev = i
	segments.append((start, prev))
	print('Number of segments: {}'.format(len(segments)))
	num_segments.append(len(segments))

	## number of points not matched because of FMM_THRESH
	count1 = 0
	for i in range(len(segments)-1):
		count1 += (segments[i+1][0] - segments[i][1] - 1)
	print('Number of points that cannot be matched: {}'.format(count1))

	to_match = []
	for (start, end) in segments:
		trip = []
		x = data[start:end+1, 2]
		y = data[start:end+1, 3]
		tms = data[start:end+1, 1]
		
		chosen_ind = []
		zero_ind = set()
		for i,(lat, lon) in enumerate(zip(x,y)):
			if not in_box(lat,lon):
				zero_ind.add(i)
			else:
				chosen_ind.append(i)
				trip.append((lat,lon))
		to_match.append(trip)
		
	f = open('wkt.pkl', 'wb')
	pickle.dump(to_match, f, protocol=2)
	f.close()

	############### MAP MATCHING #################
	start  = time.time()
	bashCommand = "python2 mapmatch_new.py"
	output = subprocess.check_output(['bash','-c', bashCommand])
	end = time.time()
	#############################################

	print('Done map matching in time: {} seconds'.format(end-start))
	times.append(end-start)
	f = open('wkt_mm.pkl', 'rb')
	matched_out = pickle.load(f)
	f.close()
	assert len(matched_out) == len(to_match), 'input and output for FMM have diff lengths'

	count_failed = 0
	unmatched = []
	errors = []
	for index, (start, end) in enumerate(segments):
		x = data[start:end+1, 2]
		y = data[start:end+1, 3]
		tms = data[start:end+1, 1]

		if not (sorted(tms) == tms).all():
			breakpoint()	# this should not happen
		
		chosen_ind = []
		zero_ind = set()
		for i,(lat, lon) in enumerate(zip(x,y)):
			if not in_box(lat,lon):
				zero_ind.add(i)
			else:
				chosen_ind.append(i)
				
		matched_pts = []
		
		cpath = matched_out[index][1]
		candidates = matched_out[index][0]
		detail = pd.DataFrame(candidates, columns=["eid","source","target","error","length","offset"])
		eids = detail['eid']
		
		if len(eids) == 0:
			count_failed += 1
			unmatched.append((start,end))
			## map-match failed - keep the points unchanged
			for (lat,lon) in zip(x,y):
				matched_pts.append((lat,lon))
			continue
			
		if len(eids) == 1:
			## retaining the original input
			data[start][4] = data[start][2]
			data[start][5] = data[start][3]
			continue
		
		eids_idx = 0
		cpath_idx = 0
		eids_idx_to_cpath_idx = {}

		time_gaps = []
		for i in range(len(x)):
			pt = (x[i],y[i])
			timestamp = tms[i]
			if i in zero_ind:
				prev_cpath_idx = eids_idx_to_cpath_idx[eids_idx-1]
				next_cpath_idx = cpath_idx
				inter_cpath = cpath[prev_cpath_idx:next_cpath_idx+1]
			
				prev_idx = chosen_ind[eids_idx-1]
				next_idx = chosen_ind[eids_idx]
			
				prev_tms, next_tms = tms[prev_idx], tms[next_idx]
				time_gaps.append(time_diff(next_tms, prev_tms))
				prev_pt, next_pt = (x[prev_idx], y[prev_idx]), (x[next_idx], y[next_idx])
			
				matched_pt = get_missing_coords(pt, timestamp, prev_pt, next_pt, prev_tms, next_tms, inter_cpath)
				# errors.append(haversine((matched_pt), (x_orig[i], y_orig[i]), unit='m'))
				matched_pts.append(matched_pt)
			
			else:
				assert chosen_ind[eids_idx] == i, "chosen_ind[eids_idx] should be equal to i"
				assert cpath[cpath_idx] == eids[eids_idx], "edges at cpath_idx and eids_idx should match"
				eids_idx_to_cpath_idx[eids_idx] = cpath_idx
				matched_pt, error = linear_interpolate(pt, eids[eids_idx])
				max_error = max(max_error, error)
				matched_pts.append(matched_pt)
				# if error>300:
				#	print(i, pt, matched_pt, error, eids[eids_idx])
				eids_idx += 1
				if (eids_idx == len(eids)):
					break
				while (cpath[cpath_idx] != eids[eids_idx]):
					cpath_idx += 1
		# fill these matched points in data
		idx_2 = 0
		assert len(matched_pts) == (end-start+1), 'matched output length not correct'
		for idx in range(start, end+1, 1):
			data[idx][4] = matched_pts[idx_2][0]
			data[idx][5] = matched_pts[idx_2][1]
			idx_2 += 1
	
	per = still_count_missing(data)
	print('done')
	print('FMM failed for {} number of input sub-trajs'.format(count_failed))
	print('Percentage points not filled: {}%'.format(per))

	f = open('output_{}.pkl'.format(filename.split('.')[0]), 'wb')
	pickle.dump(data, f)
	f.close()

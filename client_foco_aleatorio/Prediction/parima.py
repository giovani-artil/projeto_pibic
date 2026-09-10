from pyexpat import model
#from turtle import width

#from sympy import fps

from river import linear_model
from river import compose
from river import compat
from river import metrics
from river import model_selection
from river import optim
from river import preprocessing
from river import stream
from sklearn import datasets
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np 
import math
import pickle
import random
import warnings
import time

def get_model_state(model):
    return {
        "weights_x": dict(model.weights_x),
        "weights_y": dict(model.weights_y),
        "intercept_x": model.intercept_x,
        "intercept_y": model.intercept_y
    }


def set_model_state(model, state):
    if state is None:
        return model

    model.weights_x = state.get("weights_x", {})
    model.weights_y = state.get("weights_y", {})
    model.intercept_x = state.get("intercept_x", 0.0)
    model.intercept_y = state.get("intercept_y", 0.0)
    return model

def clip_model_weights(model, limit=1e4):
    for k in model.weights_x:
        model.weights_x[k] = np.clip(model.weights_x[k], -limit, limit)

    for k in model.weights_y:
        model.weights_y[k] = np.clip(model.weights_y[k], -limit, limit)

    model.intercept_x = np.clip(model.intercept_x, -limit, limit)
    model.intercept_y = np.clip(model.intercept_y, -limit, limit)

    return model

def pred_frames(data, model, metric_X, metric_Y, frames, prev_frames, tile_manhattan_error, act_tiles, pred_tiles, count, width, height, nrow_tiles, ncol_tiles):
	x_pred, y_pred = 0,0
	series = []
	shift_x = False

	for k in range(len(prev_frames)):
		[inp_k, x_act, y_act] = data[prev_frames[k]]
		a = [-1 for i in range(2)]
		for key, value in inp_k.items():
			value = np.clip(value, -1e3, 1e3)
			if key == 'VIEWPORT_x':
				a[0] = value
			if key == 'VIEWPORT_y':
				a[1] = value

		if k==0:
			series.append(a)
			continue

		[prev_x,prev_y] = series[-1]
		
		if prev_x < a[0]:
			new_x_dif = a[0]-width
			if a[0] - prev_x > prev_x - new_x_dif:
				a[0] = new_x_dif
		else:
			new_x_dif = a[0]+width
			if prev_x - a[0] > new_x_dif - prev_x:
				a[0] = new_x_dif

		if a[0]<0:
			shift_x = True

		series.append(a)

	if shift_x == True:
		for i in range(len(series)):
			series[i][0] = series[i][0]+width

	series = np.array(series, dtype=np.float64)
	series_copy=series

	result = len(series[:, 0]) > 0 and all(elem == series[0][0] for elem in series[:, 0])
	if(result):
		series[:, 0] = [elem + random.random()/10 for elem in series[:, 0]]
	for i in range(len(series[:, 0])):
		if series[i, 0] == 0:
			series[i, 0] += random.random()
	series[:, 0] = np.clip(series[:, 0], 1e-6, None)
	series_log = np.log(series[:, 0])
	series_x = np.diff(series_log, 1)

	result = len(series[:, 1]) > 0 and all(elem == series[0][1] for elem in series[:, 1])
	if(result):
		series[:, 1] = [elem + random.random()/10 for elem in series[:, 1]]
	for i in range(len(series[:, 1])):
		if series[i, 1] == 0:
			series[i, 1] += random.random()
	series[:, 1] = np.clip(series[:, 1], 1e-6, None)
	series_log = np.log(series[:, 1])
	series_y = np.diff(series_log, 1)

	series_new = []
	for i in range(len(series_x)):
		series_new.append([series_x[i], series_y[i]])


	with warnings.catch_warnings():
		warnings.filterwarnings("ignore")
		
		model_x = SARIMAX(series_x, order=(2,0,1))
		model_fit_x = model_x.fit(maxiter=1000, disp=0, method='nm')
		model_pred_x = model_fit_x.forecast(len(frames))

		model_y = SARIMAX(series_y, order=(3,0,0))
		model_fit_y = model_y.fit(maxiter=1000, disp=0, method='nm')
		model_pred_y = model_fit_y.forecast(len(frames))

	model_pred_x = np.clip(model_pred_x, -10, 10)
	model_pred_y = np.clip(model_pred_y, -10, 10)

	# print(model_pred_x)
	x_pred_list, y_pred_list = [], []
	for k in range(len(model_pred_x)):
		model_pred_x[k] = np.clip(model_pred_x[k], -7, 7)
		model_pred_y[k] = np.clip(model_pred_y[k], -7, 7)
		if k == 0:
			x_val = np.exp(model_pred_x[k]) * series_copy[-1][0]
			y_val = np.exp(model_pred_y[k]) * series_copy[-1][1]
		else:
			x_val = np.exp(model_pred_x[k]) * x_pred_list[k-1]
			y_val = np.exp(model_pred_y[k]) * y_pred_list[k-1]

		x_val = np.clip(x_val, -1e4, 1e4)
		y_val = np.clip(y_val, -1e4, 1e4)

		x_pred_list.append(x_val)
		y_pred_list.append(y_val)

	for k in range(len(frames)):
		[inp_k, x_act, y_act] = data[frames[k]]
		for key in inp_k:
			inp_k[key] = np.clip(inp_k[key], -1e3, 1e3)

		x_act = np.clip(x_act, 0, width)
		y_act = np.clip(y_act, 0, height)
		if(k == 0):
			x_pred, y_pred = model.predict_one(inp_k, True, x_act, y_act)
		else:
			if(shift_x==True):
				x_pred_list[k] = x_pred_list[k] - width
			inp_k['VIEWPORT_x'] = x_pred_list[k-1]
			inp_k['VIEWPORT_y'] = y_pred_list[k-1]
			for key in inp_k:
				inp_k[key] = np.clip(inp_k[key], -1e3, 1e3)
			x_pred, y_pred = model.predict_one(inp_k, False, None, None)	

		shift = 0
		if(x_act > x_pred):
			if(abs(x_act - x_pred) > abs(x_act - (x_pred+width))):
				x_pred = x_pred+width
				shift = 1
		else:
			if(abs(x_act - x_pred) > abs(x_act - (x_pred-width))):
				x_pred = x_pred-width
				shift = 2

		metric_X = metric_X.update(x_act, x_pred)
		metric_Y = metric_Y.update(y_act, y_pred)

		if(shift == 0):
			actual_tile_col = int(x_act * ncol_tiles / width)
			actual_tile_row = int(y_act * nrow_tiles / height)
			pred_tile_col = int(x_pred * ncol_tiles / width)
			pred_tile_row = int(y_pred * nrow_tiles / height)
		elif(shift == 1):
			actual_tile_col = int(x_act * ncol_tiles / width)
			actual_tile_row = int(y_act * nrow_tiles / height)
			pred_tile_col = int((x_pred - width) * ncol_tiles / width)
			pred_tile_row = int(y_pred * nrow_tiles / height)
		else:
			actual_tile_col = int(x_act * ncol_tiles / width)
			actual_tile_row = int(y_act * nrow_tiles / height)
			pred_tile_col = int((x_pred + width) * ncol_tiles / width)
			pred_tile_row = int(y_pred * nrow_tiles / height)

		actual_tile_row = actual_tile_row-nrow_tiles if(actual_tile_row >= nrow_tiles) else actual_tile_row
		actual_tile_col = actual_tile_col-ncol_tiles if(actual_tile_col >= ncol_tiles) else actual_tile_col
		actual_tile_row = actual_tile_row+nrow_tiles if actual_tile_row < 0 else actual_tile_row
		actual_tile_col = actual_tile_col+ncol_tiles if actual_tile_col < 0 else actual_tile_col

		######################################################
		# print("x: "+str(x_act))
		# print("x_pred: "+str(x_pred))
		# print("y: "+str(y_act))	
		# print("y_pred: "+str(y_pred))
		# print("("+str(actual_tile_row)+","+str(actual_tile_col)+"),("+str(pred_tile_row)+","+str(pred_tile_col)+")")
		# # ######################################################
		
		act_tiles.append((actual_tile_row, actual_tile_col))
		pred_tiles.append((pred_tile_row, pred_tile_col))

		tile_col_dif = ncol_tiles

		if actual_tile_col < pred_tile_col:
			tile_col_dif = min(pred_tile_col - actual_tile_col, actual_tile_col + ncol_tiles - pred_tile_col)
		else:
			tile_col_dif = min(actual_tile_col - pred_tile_col, ncol_tiles + pred_tile_col - actual_tile_col)

		tile_manhattan_error += abs(actual_tile_row - pred_tile_row) + abs(tile_col_dif)
		count = count+1

	return metric_X, metric_Y, tile_manhattan_error, count, act_tiles, pred_tiles



def build_model(data, frame_nos, max_frame, tot_objects, width, height, nrow_tiles, ncol_tiles, fps, pred_nframe, global_state=None,save_path=None,start_frame_id=0,end_frame_id=0,warmup=True,method="fedavg",mu=0.01,c_global=None,c_local=None):
	model = linear_model.PARegressor(C=0.01, mode=2, eps=0.001, data=data, learning_rate=0.1, rho=0.9)
	model = set_model_state(model, global_state)
	metric_X = metrics.MAE()
	metric_Y = metrics.MAE()
	manhattan_error = []
	x_mae = []
	y_mae = []
	count=0

	i=0
	tile_manhattan_error=0
	act_tiles, pred_tiles = [],[]
	act_tiles_by_chunk = []
	chunk_frames = []
	start_idx = next((i for i, f in enumerate(frame_nos) if f >= start_frame_id), len(frame_nos) - 1)
	end_idx = next((i for i, f in enumerate(frame_nos) if f >= end_frame_id), len(frame_nos))

	def apply_strategy(model):
		if method == "fedavg" or global_state is None:
			return model
		
		local_state = get_model_state(model)
	
		if method == "fedprox":
			for k in local_state["weights_x"]:
				local_state["weights_x"][k] -= mu * (
                    local_state["weights_x"][k] - global_state.get("weights_x", {}).get(k, 0.0)
                )

			for k in local_state["weights_y"]:
				local_state["weights_y"][k] -= mu * (
                    local_state["weights_y"][k] - global_state.get("weights_y", {}).get(k, 0.0)
                )
		elif method == "scaffold" and c_global is not None and c_local is not None:
			for k in local_state["weights_x"]:
				correction = (
                    c_global.get("weights_x", {}).get(k, 0.0)
                    - c_local.get("weights_x", {}).get(k, 0.0)
                )
				local_state["weights_x"][k] += correction

			for k in local_state["weights_y"]:
				correction = (
                    c_global.get("weights_y", {}).get(k, 0.0)
                    - c_local.get("weights_y", {}).get(k, 0.0)
                )
				local_state["weights_y"][k] += correction

		return set_model_state(model, local_state)

	#prev_frames = {0}
	if warmup:
		for i in range(start_idx, end_idx):
			#curr_frame=frame_nos[i]
			#prev_frames.add(i)
				[inp_i,x,y]=data[i]
				for key in inp_i:
					inp_i[key] = np.clip(inp_i[key], -1e3, 1e3)

				x_pred_tmp, y_pred_tmp = model.predict_one(inp_i, False, None, None)
				x_pred_tmp = np.clip(x_pred_tmp, 0, width)
				y_pred_tmp = np.clip(y_pred_tmp, 0, height)
				x = x_pred_tmp + np.clip(x - x_pred_tmp, -100, 100)
				y = y_pred_tmp + np.clip(y - y_pred_tmp, -100, 100)
				model = model.learn_one(inp_i,x,y)
				model = apply_strategy(model)
				model = clip_model_weights(model)

		if save_path is not None:
			with open(save_path, 'wb') as f:
				pickle.dump(get_model_state(model), f)

		return [], [], [], [], [], []
	#prev_frames = sorted(prev_frames)
	#cnt = 0

	prev_frames = []

	# Predicting frames and update model
	for i in range(start_idx, end_idx):
		curr_frame = frame_nos[i]
		nframe = min(pred_nframe, max_frame - frame_nos[i])

		if(nframe < 1):
			break

		frames = {i}
		k=i
		for k in range(i+1, len(frame_nos)):
			if(frame_nos[k] < curr_frame + nframe):
				frames.add(k)
			else:
				i=k
				break
		i=k

		if(i==(len(frame_nos)-1)):
			break
		frames = sorted(frames)

		if len(prev_frames) < 2:
			for idx in frames:
				inp_k, x_k, y_k = data[idx]
				for key in inp_k:
					inp_k[key] = np.clip(inp_k[key], -1e3, 1e3)
				x_k = np.clip(x_k, 0, width)
				y_k = np.clip(y_k, 0, height)
				model = model.learn_one(inp_k, x_k, y_k)
				model = apply_strategy(model)
				model = clip_model_weights(model)
			prev_frames.extend(frames)
			continue

		chunk_frames.append(frames)

		before = len(act_tiles)
		metric_X, metric_Y, tile_manhattan_error, count, act_tiles, pred_tiles = pred_frames(data, model, metric_X, metric_Y, frames, prev_frames, tile_manhattan_error, act_tiles, pred_tiles, count, width, height, nrow_tiles, ncol_tiles)
		act_tiles_by_chunk.append(act_tiles[before:])
		#model = model.learn_n(frames)
		for idx in frames:
			inp_k, x_k, y_k = data[idx]
			for key in inp_k:
					inp_k[key] = np.clip(inp_k[key], -1e3, 1e3)
			x_pred_tmp, y_pred_tmp = model.predict_one(inp_k, False, None, None)

			x_pred_tmp = np.clip(x_pred_tmp, 0, width)
			y_pred_tmp = np.clip(y_pred_tmp, 0, height)

			x_k = x_pred_tmp + np.clip(x_k - x_pred_tmp, -200, 200)
			y_k = y_pred_tmp + np.clip(y_k - y_pred_tmp, -200, 200)
			model = model.learn_one(inp_k, x_k, y_k)
			model = apply_strategy(model)
			model = clip_model_weights(model)

		prev_frames.extend(frames)
		if len(prev_frames) > 5 * fps:
			prev_frames = prev_frames[-5 * fps:]
		manhattan_error.append(tile_manhattan_error*1.0 / count)
		x_mae.append(metric_X.get())
		y_mae.append(metric_Y.get())

		#print("Manhattan Tile Error: "+str(tile_manhattan_error*1.0 / count))
		#print(metric_X, metric_Y)
		#print("\n")
		#cnt = cnt + 1
		#if cnt == 60:
		#	break

	if save_path is not None:
		with open(save_path, 'wb') as f:
			pickle.dump(get_model_state(model), f)

	return act_tiles_by_chunk, pred_tiles, chunk_frames, manhattan_error, x_mae, y_mae
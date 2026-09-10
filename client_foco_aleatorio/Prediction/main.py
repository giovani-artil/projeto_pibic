import time
from tracemalloc import start

import numpy as np 
import math
import pickle
import random
import argparse
import json
from parima import build_model
from bitrate import alloc_bitrate
from qoe import calc_qoe
from pathlib import Path
import requests
import os
import threading

SERVER_URL = os.environ.get("SERVER_URL", "http://server:5000")
SILO_ID = os.environ.get("SILO_ID", "aleatorio")
pending_global = {"fedavg": None, "fedprox": None, "scaffold": None}
pending_global_lock = threading.Lock()

def async_sync_with_server(local_avg, local_prox, local_scaf, silo_id):
    try:
        fedavg, fedprox, scaffold = sync_with_server(local_avg, local_prox, local_scaf, silo_id)
        with pending_global_lock:
            pending_global["fedavg"] = fedavg
            pending_global["fedprox"] = fedprox
            pending_global["scaffold"] = scaffold
        print(f"[FL] Modelo global armazenado para aplicação no próximo chunk.")
    except Exception as e:
        print(f"[FL] Erro na thread de sincronização: {e}")

def sync_with_server(local_avg, local_prox, local_scaf, silo_id):
	with open(local_avg, "rb") as f:
		state_avg = pickle.load(f)
	with open(local_prox, "rb") as f:
		state_prox = pickle.load(f)
	with open(local_scaf, "rb") as f:
		state_scaf = pickle.load(f)

	payload = {
		"silo_id": silo_id,
		"fedavg": state_avg,
		"fedprox": state_prox,
		"scaffold": state_scaf
	}

	print(f"[FL] Enviando modelos ao servidor (silo={silo_id})...")
	for attempt in range(10):
		try:
			response = requests.post(f"{SERVER_URL}/aggregate", json=payload, timeout=10000)
			response.raise_for_status()
			data = response.json()
			if data.get("status") == "dropped":
				print(f"[FL] Pacote descartado pelo servidor. Reenviando em 1s...")
				time.sleep(1)
				continue
			print(f"[FL] Modelo global recebido do servidor.")
			return data["fedavg"], data["fedprox"], data["scaffold"]
		except requests.exceptions.ConnectionError:
			print(f"[FL] Servidor indisponível, tentativa {attempt+1}/10. Aguardando...")
			time.sleep(5)
		except requests.exceptions.ReadTimeout:
			print(f"[FL] Timeout na leitura, tentativa {attempt+1}/10. Aguardando...")
			time.sleep(5)

	raise RuntimeError("[FL] Não foi possível conectar ao servidor após 10 tentativas.")

def get_data(data, frame_nos, dataset, topic, usernum, fps, offset, milisec, width, height, view_width, view_height):
	VIEW_PATH = '/app/client_foco_aleatorio/Viewport/'
	OBJ_PATH = '/app/client_foco_aleatorio/shared_data/Obj_traj/'

	obj_info = np.load(OBJ_PATH + 'ds{}/ds{}_topic{}.npy'.format(dataset, dataset, topic), allow_pickle=True,  encoding='latin1').item()
	view_info = pickle.load(open(VIEW_PATH + 'ds{}/viewport_ds{}_topic{}_user{}'.format(dataset, dataset, topic, usernum), 'rb'), encoding='latin1')

	n_objects = []
	for i in obj_info.keys():
		try:
			n_objects.append(max(obj_info[i].keys()))
		except:
			n_objects.append(0)
	total_objects=max(n_objects)

	if dataset == 1:
		max_frame = int(view_info[-1][0]*1.0*fps/milisec)

		for i in range(len(view_info)-1):
			frame = int(view_info[i][0]*1.0*fps/milisec)
			frame += int(offset*1.0*fps/milisec)

			frame_nos.append(frame)
			if(frame > max_frame):
				break
			X={}
			X['VIEWPORT_x']=int(view_info[i][1][1]*width/view_width)
			X['VIEWPORT_y']=int(view_info[i][1][0]*height/view_height)
			for j in range(total_objects):
				try:
					centroid = obj_info[frame][j]

					if obj_info[frame][j] == None:
						X['OBJ_'+str(j)+'_x']=np.random.normal(0,1)
						X['OBJ_'+str(j)+'_y']=np.random.normal(0,1)
					else:
						X['OBJ_'+str(j)+'_x']=centroid[0]
						X['OBJ_'+str(j)+'_y']=centroid[1]

				except:
					X['OBJ_'+str(j)+'_x']=np.random.normal(0,1)
					X['OBJ_'+str(j)+'_y']=np.random.normal(0,1)


			data.append((X, int(view_info[i+1][1][1]*width/view_width),int(view_info[i+1][1][0]*height/view_height)))

	elif dataset == 2:
		for k in range(len(view_info)-1):
			if view_info[k][0]<=offset+60 and view_info[k+1][0]>offset+60:
				max_frame = int(view_info[k][0]*1.0*fps/milisec)
				break
		
		for k in range(len(view_info)-1):
			if view_info[k][0]<=offset and view_info[k+1][0]>offset:
				min_index = k+1
				break
				

		prev_frame = 0
		for i in range(min_index,len(view_info)-1):
			frame = int((view_info[i][0])*1.0*fps/milisec)

			if frame == prev_frame:
				continue
			
			if(frame > max_frame):
				break

			frame_nos.append(frame)
			
			X={}
			X['VIEWPORT_x']=int(view_info[i][1][1]*width/view_width)
			X['VIEWPORT_y']=int(view_info[i][1][0]*height/view_height)
			for j in range(total_objects):
				try:
					centroid = obj_info[frame][j]

					if obj_info[frame][j] == None:
						X['OBJ_'+str(j)+'_x']=np.random.normal(0,1)
						X['OBJ_'+str(j)+'_y']=np.random.normal(0,1)
					else:
						X['OBJ_'+str(j)+'_x']=centroid[0]
						X['OBJ_'+str(j)+'_y']=centroid[1]

				except:
					X['OBJ_'+str(j)+'_x']=np.random.normal(0,1)
					X['OBJ_'+str(j)+'_y']=np.random.normal(0,1)


			data.append((X, int(view_info[i+1][1][1]*width/view_width),int(view_info[i+1][1][0]*height/view_height)))
			prev_frame = frame
			
	return data, frame_nos, max_frame,total_objects

def rangeID(silo_id, numUser):
	if silo_id == 1:
		return range(1, 1+numUser)
	else:
		return range(9, 9+numUser)

def main():

	parser = argparse.ArgumentParser(description='Run PARIMA algorithm and calculate QoE of a video for a single user')

	parser.add_argument('-D', '--dataset', type=int, required=True, help='Dataset ID (1 or 2)')
	parser.add_argument('-T', '--topic', required=True, help='Topic in the particular Dataset (video name)')
	parser.add_argument('--fps', type=int, required=True, help='fps of the video')
	parser.add_argument('-O', '--offset', type=int, default=0, help='Offset for the start of the video in seconds (when the data was logged in the dataset) [default: 0]')
	# parser.add_argument('-U', '--user', type=int, default=0, help='User ID on which the algorithm will be run [default: 0]')
	parser.add_argument('-Q', '--quality', required=True, help='Preferred bitrate quality of the video (360p, 480p, 720p, 1080p, 1440p)')
	parser.add_argument('-I', '--silo_id', type=int, required=True, help='ID do silo (1 ou 2)')
	parser.add_argument('--syncOrAssync', required=True, help='Se mantém conexão síncrona com o servidor ou não (default: sync)')
	parser.add_argument('--numUsers', type=int, required=True, help='Número total de usuários')
	parser.add_argument('--chuncksToWait', type=int, required=True, help='Número de chunks antes de enviar para o servidor (default: 1)')

	args = parser.parse_args()

	if args.dataset != 1 and args.dataset != 2:
		print("Incorrect value of the Dataset ID provided!!...")
		print("======= EXIT ===========")
		exit()

	if args.silo_id != 1 and args.silo_id != 2:
		print("Incorrect value of the Silo ID provided!!...")
		print("======= EXIT ===========")
		exit()

	verify_path = Path(f"/app/client_foco_aleatorio/models/{args.topic}/")
	verify_path.mkdir(parents=True, exist_ok=True)

	# Get the necessary information regarding the dimensions of the video
	print("Reading JSON...")
	file = open('/app/client_foco_aleatorio/Prediction/meta.json', )
	jsonRead = json.load(file)

	width = jsonRead["dataset"][args.dataset-1]["width"]
	height = jsonRead["dataset"][args.dataset-1]["height"]
	view_width = jsonRead["dataset"][args.dataset-1]["view_width"]
	view_height = jsonRead["dataset"][args.dataset-1]["view_height"]
	milisec = jsonRead["dataset"][args.dataset-1]["milisec"]

	pref_bitrate = jsonRead["bitrates"][args.quality]
	ncol_tiles = jsonRead["ncol_tiles"]
	nrow_tiles = jsonRead["nrow_tiles"]
	player_width = jsonRead["player_width"]
	player_height = jsonRead["player_height"]

	player_tiles_x = math.ceil(player_width*ncol_tiles*1.0/width)
	player_tiles_y = math.ceil(player_height*nrow_tiles*1.0/height)

	manhattan_error_fedavg_list = []
	manhattan_error_fedprox_list = []
	manhattan_error_scaffold_list = []
	
	numUsers = args.numUsers

	users_data = {}
	user_history = {}

	silo = args.silo_id

	chunkToWait = args.chuncksToWait
	countChunk = 0

	connection_type = args.syncOrAssync

	max_frame = 0

	for user in rangeID(silo, numUsers):
		data, frame_nos = [], []

		data, frame_nos, nmax_frame, tot_objects = get_data(
	        data, frame_nos, args.dataset, args.topic, user,
	        args.fps, args.offset, milisec, width, height,
	        view_width, view_height
	    )
		if nmax_frame > max_frame:
			max_frame = nmax_frame

		users_data[user] = (data, frame_nos)

	for user in rangeID(silo, numUsers):

		user_history[user] = {
        	"act_avg": [], "pred_avg": [], "frames_avg": [], "bitrate_avg": [],
        	"act_prox": [], "pred_prox": [], "frames_prox": [], "bitrate_prox": [],
        	"act_scaf": [], "pred_scaf": [], "frames_scaf": [], "bitrate_scaf": []
    }

	global_state_fedavg = None
	global_state_fedprox = None
	global_state_scaffold = None

	try:
				with open(f"/app/client_foco_aleatorio/models/{args.topic}/{args.topic}_global_fedavg.pkl", "rb") as f:
					global_state_fedavg = pickle.load(f)				
				with open(f"/app/client_foco_aleatorio/models/{args.topic}/{args.topic}_global_fedprox.pkl", "rb") as f:
					global_state_fedprox = pickle.load(f)
				with open(f"/app/client_foco_aleatorio/models/{args.topic}/{args.topic}_global_scaffold.pkl", "rb") as f:
					global_state_scaffold = pickle.load(f)
	except:
		pass

	state_fedavg = global_state_fedavg
	state_fedprox = global_state_fedprox
	state_scaffold = global_state_scaffold
		
	chunk_size = args.fps
	num_chunks = int(max_frame / chunk_size)
	print(f"LOG\nCHUNKS: {num_chunks}\nCHUNK SIZE: {chunk_size}\nMAX FRAME: {max_frame}")

	c_local = {"weights_x": {}, "weights_y": {}}
	last_sync_thread = None

	for chunk in range(num_chunks):
		print(f"\n=== ROUND {chunk} ===")

		if connection_type == 'assync':
			with pending_global_lock:
				if pending_global["fedavg"] is not None:
					state_fedavg = pending_global["fedavg"]
					state_fedprox = pending_global["fedprox"]
					state_scaffold = pending_global["scaffold"]
					pending_global["fedavg"] = None
					pending_global["fedprox"] = None
					pending_global["scaffold"] = None
					print(f"[FL] Modelo global aplicado no chunk {chunk}.")

		start_frame = chunk * chunk_size
		end_frame = start_frame + chunk_size

		path_fedavg = f"/app/client_foco_aleatorio/models/{args.topic}/{args.topic}_aleatorio_fedavg.pkl"
		path_fedprox = f"/app/client_foco_aleatorio/models/{args.topic}/{args.topic}_aleatorio_fedprox.pkl"
		path_scaffold = f"/app/client_foco_aleatorio/models/{args.topic}/{args.topic}_aleatorio_scaffold.pkl"

		for user in rangeID(silo, numUsers):

			data, frame_nos = users_data[user]

			act_tiles_fedavg, pred_tiles_fedavg, chunk_frames_fedavg, manhattan_error_fedavg, x_mae, y_mae = build_model(data, frame_nos, max_frame, tot_objects, width, height, nrow_tiles, ncol_tiles, args.fps, chunk_size, global_state=state_fedavg, save_path=path_fedavg,start_frame_id=start_frame,end_frame_id=end_frame,warmup=(chunk<5),method="fedavg")
				
			act_tiles_fedprox, pred_tiles_fedprox, chunk_frames_fedprox, manhattan_error_fedprox, x_mae, y_mae = build_model(data, frame_nos, max_frame, tot_objects, width, height, nrow_tiles, ncol_tiles, args.fps, chunk_size, global_state=state_fedprox, save_path=path_fedprox,start_frame_id=start_frame,end_frame_id=end_frame,warmup=(chunk<5),method="fedprox",mu=0.01)

			act_tiles_scaffold, pred_tiles_scaffold, chunk_frames_scaffold, manhattan_error_scaffold, x_mae, y_mae = build_model(data, frame_nos, max_frame, tot_objects, width, height, nrow_tiles, ncol_tiles, args.fps, chunk_size, global_state=state_scaffold, save_path=path_scaffold,start_frame_id=start_frame,end_frame_id=end_frame,warmup=(chunk<5),method="scaffold",c_global=state_scaffold,c_local=c_local)
			
			vid_bitrate_avg = alloc_bitrate(pred_tiles_fedavg, chunk_frames_fedavg, nrow_tiles, ncol_tiles, pref_bitrate, player_tiles_x, player_tiles_y)
			vid_bitrate_fedprox = alloc_bitrate(pred_tiles_fedprox, chunk_frames_fedprox, nrow_tiles, ncol_tiles, pref_bitrate, player_tiles_x, player_tiles_y)
			vid_bitrate_scaffold = alloc_bitrate(pred_tiles_scaffold, chunk_frames_scaffold, nrow_tiles, ncol_tiles, pref_bitrate, player_tiles_x, player_tiles_y)

			if chunk >= 5:
				adj_frames_avg = [[f + start_frame for f in chunk] for chunk in chunk_frames_fedavg]
				adj_frames_prox = [[f + start_frame for f in chunk] for chunk in chunk_frames_fedprox]
				adj_frames_scaf = [[f + start_frame for f in chunk] for chunk in chunk_frames_scaffold]

				user_history[user]["act_avg"].extend(act_tiles_fedavg)
				user_history[user]["pred_avg"].extend(pred_tiles_fedavg)
				user_history[user]["frames_avg"].extend(adj_frames_avg)
				user_history[user]["bitrate_avg"].extend(vid_bitrate_avg)

				user_history[user]["act_prox"].extend(act_tiles_fedprox)
				user_history[user]["pred_prox"].extend(pred_tiles_fedprox)
				user_history[user]["frames_prox"].extend(adj_frames_prox)
				user_history[user]["bitrate_prox"].extend(vid_bitrate_fedprox)

				user_history[user]["act_scaf"].extend(act_tiles_scaffold)
				user_history[user]["pred_scaf"].extend(pred_tiles_scaffold)
				user_history[user]["frames_scaf"].extend(adj_frames_scaf)
				user_history[user]["bitrate_scaf"].extend(vid_bitrate_scaffold)

				manhattan_error_fedavg_list.extend(manhattan_error_fedavg)
				manhattan_error_fedprox_list.extend(manhattan_error_fedprox)
				manhattan_error_scaffold_list.extend(manhattan_error_scaffold)

			with open(path_fedavg, "rb") as f:
				state_fedavg = pickle.load(f)
			with open(path_fedprox, "rb") as f:
				state_fedprox = pickle.load(f)
			with open(path_scaffold, "rb") as f:
				state_scaffold = pickle.load(f)
			c_local = state_scaffold

		if chunk >= 5:
			countChunk += 1
			if countChunk == chunkToWait:
				countChunk = 0
				if connection_type == 'sync':
					state_fedavg, state_fedprox, state_scaffold = sync_with_server(
						path_fedavg, path_fedprox, path_scaffold, SILO_ID
					)
				else:
					t = threading.Thread(
						target=async_sync_with_server,
						args=(path_fedavg, path_fedprox, path_scaffold, SILO_ID),
						daemon=True
					)
					t.start()
					last_sync_thread = t

			print('Erro Manhattan (FedAvg) do Round {}: {}'.format(chunk, np.mean(manhattan_error_fedavg_list)))
			print('Erro Manhattan (FedProx) do Round {}: {}'.format(chunk, np.mean(manhattan_error_fedprox_list)))
			print('Erro Manhattan (Scaffold) do Round {}: {}'.format(chunk, np.mean(manhattan_error_scaffold_list)))

	print("Calculate QoE...")

	if connection_type == 'assync' and last_sync_thread is not None:
		print("[FL] Aguardando última sincronização com o servidor...")
		last_sync_thread.join()
		with pending_global_lock:
			if pending_global["fedavg"] is not None:
				state_fedavg = pending_global["fedavg"]
				state_fedprox = pending_global["fedprox"]
				state_scaffold = pending_global["scaffold"]

	qoe_avg_all = []
	qoe_prox_all = []
	qoe_scaf_all = []

	for user in rangeID(silo, numUsers):

		h = user_history[user]

		qoe_avg = calc_qoe(
			h["bitrate_avg"],
			h["act_avg"],
			h["frames_avg"],
			width, height,
			nrow_tiles, ncol_tiles,
			player_width, player_height
		)

		qoe_prox = calc_qoe(
			h["bitrate_prox"],
			h["act_prox"],
			h["frames_prox"],
			width, height,
			nrow_tiles, ncol_tiles,
			player_width, player_height
		)

		qoe_scaf = calc_qoe(
			h["bitrate_scaf"],
			h["act_scaf"],
			h["frames_scaf"],
			width, height,
			nrow_tiles, ncol_tiles,
			player_width, player_height
		)

		qoe_avg_all.append(qoe_avg)
		qoe_prox_all.append(qoe_prox)
		qoe_scaf_all.append(qoe_scaf)

	mean_qoe_avg = np.mean(qoe_avg_all)
	mean_qoe_fedprox = np.mean(qoe_prox_all)
	mean_qoe_scaffold = np.mean(qoe_scaf_all)

	n_active_chunks = num_chunks - 5

	#print(qoe)
	#Print averaged results
	print("\n======= RESULTS ============")
	print('Dataset: {}'.format(args.dataset))
	print('Topic: {}'.format(args.topic))
	# print('User ID: {}'.format(args.user))
	print('QoE Média (FedAvg): {}'.format(mean_qoe_avg/n_active_chunks))
	print('QoE Média (FedProx): {}'.format(mean_qoe_fedprox/n_active_chunks))
	print('QoE Média (Scaffold): {}'.format(mean_qoe_scaffold/n_active_chunks))
	print('Erro Manhattan Médio (FedAvg): {}'.format(np.mean(manhattan_error_fedavg_list)))
	print('Erro Manhattan Médio (FedProx): {}'.format(np.mean(manhattan_error_fedprox_list)))
	print('Erro Manhattan Médio (Scaffold): {}'.format(np.mean(manhattan_error_scaffold_list)))

	print('\n\n')

if __name__ == '__main__':
	main()

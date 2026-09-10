from flask import Flask, request, jsonify
from collections import defaultdict
import threading
import time
import pickle
import os
from fedavg import fedavg_from_states
from fedprox import fedprox_from_states
from scaffold import scaffold_from_states
import random

app = Flask(__name__)

TOTAL_CLIENTS = int(os.environ.get("TOTAL_CLIENTS", 4))
SERVER_MODE = os.environ.get("SERVER_MODE", "sync")
TIMER_INTERVAL = float(os.environ.get("TIMER_INTERVAL", "1.5"))
PACKET_LOSS_RATE = float(os.environ.get("PACKET_LOSS_RATE", "0.0"))

received = {}
received_lock = threading.Lock()
aggregation_done = threading.Event()
global_result = {}
response_count = 0
response_count_lock = threading.Lock()
last_aggregated = set()


def do_aggregate():
    global global_result
    print("[Server] Todos os clientes enviaram. Agregando...", flush=True)
    global_result = {
        "fedavg": fedavg_from_states([v["fedavg"] for v in received.values()]),
        "fedprox": fedprox_from_states([v["fedprox"] for v in received.values()]),
        "scaffold": scaffold_from_states([v["scaffold"] for v in received.values()])
    }
    print("[Server] Agregação completa.", flush=True)


def timer_aggregate():
    while True:
        time.sleep(TIMER_INTERVAL)
        with received_lock:
            new_clients = set(received.keys()) - last_aggregated
            if not new_clients:
                continue

            print(f"[Server] Timer disparou. Modelos novos: {len(new_clients)}. Agregando...", flush=True)
            do_aggregate()
            last_aggregated.update(received.keys())
            last_aggregated.clear()
            aggregation_done.set()


@app.route("/aggregate", methods=["POST"])
def aggregate():
    global response_count

    if random.random() < PACKET_LOSS_RATE:
        print(f"[Server] Pacote descartado (simulação de perda).", flush=True)
        return jsonify({"status": "dropped"}), 200

    data = request.json
    client_key = f"{data['silo_id']}"

    with received_lock:
        received[client_key] = {
            "fedavg": data["fedavg"],
            "fedprox": data["fedprox"],
            "scaffold": data["scaffold"]
        }
        count = len(received)
        print(f"[Server] Recebido de {client_key}. Total: {count}/{TOTAL_CLIENTS}", flush=True)

        if SERVER_MODE == "sync":
            if count == TOTAL_CLIENTS:
                do_aggregate()
                received.clear()
                aggregation_done.set()

    aggregation_done.wait()

    with received_lock:
        response_count += 1
        if response_count == TOTAL_CLIENTS:
            response_count = 0
            aggregation_done.clear()

    return jsonify(global_result), 200


if __name__ == "__main__":
    if SERVER_MODE == "timer":
        t = threading.Thread(target=timer_aggregate, daemon=True)
        t.start()
        print(f"[Server] Modo timer iniciado. Intervalo: {TIMER_INTERVAL}s", flush=True)
    else:
        print("[Server] Modo síncrono iniciado.", flush=True)
    app.run(host="0.0.0.0", port=5000, threaded=True)   
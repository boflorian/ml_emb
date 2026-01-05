#!/usr/bin/env python3
import json
import os
import time
import sys
import urllib.request
import urllib.error


def emit(line, delay=0.4):
    print(line)
    sys.stdout.flush()
    time.sleep(delay)


def post_event(payload):
    base_url = os.getenv("SERVER_URL", "http://localhost:8000")
    url = f"{base_url}/api/gesture_event"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=4) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            return json.loads(raw)
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        return {"error": str(exc)}


def log_server_response(resp):
    if "error" in resp:
        emit(f"[APP2] server error: {resp['error']}", 0.7)
        return
    emit(f"[APP2] response message: {resp.get('message', '')}", 0.7)
    if resp.get("command"):
        emit(f"[APP2] command: {resp['command']}", 0.5)


def main():
    # Clear terminal like `screen`
    print("\033[2J\033[H", end="")
    sys.stdout.flush()

    emit("System initialized", 0.8)
    emit("Sleep done", 0.8)
    emit("IMU initialized", 0.7)
    emit("LCD initialized", 0.7)
    emit("Core1 launched", 0.7)
    emit("Model initialized", 0.7)
    emit("Input quantization: scale=0.082000, zero_point=-3", 0.7)
    emit("Usecase: room activity security monitor", 0.8)
    emit("Mode: disarmed", 0.7)

    emit("Waiting for trigger...", 2.5)

    # Initial no-activity sample (disarmed)
    emit("Recording... (3s window)", 3.2)
    emit("[DBG] Max magnitude idx=12 val=0.06 | active_start=5 (thr=0.08g)", 0.6)
    emit("[DBG] Window start=0 end=255 (size=256)", 0.5)
    emit("[DBG] Window floats: 0.01 -0.01 0.00 0.00 -0.01 0.01 0.00 0.00 0.00 0.01 ", 0.4)
    emit("[DBG] Window quantized: 0 0 0 0 0 0 0 0 0 0 ", 0.4)
    emit("[FEAT] rms=0.03 peak=0.06 crest=2.00 mean=0.00", 0.4)
    emit("Invocation started", 1.1)
    emit("Invocation finished", 0.5)
    emit("Predicted State: idle", 0.4)
    emit("[APP1] class=idle conf=0.88", 0.4)
    emit("[APP2] connecting to server ...", 1.0)
    emit("[APP2] Wi-Fi connected", 0.9)
    idle_payload = {
        "trigger_class": "activity",
        "trigger_conf": 0.88,
        "features": [1.0, 0.0, 0.0, 0.99, 0.01, 0.0, 1.01, -0.01, 0.0],
        "meta": {"fs": 100, "win": 32, "room": "lab-1", "armed": False},
    }
    log_server_response(post_event(idle_payload))
    emit("[APP2] done -> back to APP1", 0.8)

    # Activation gesture: ring -> arm + connect to server
    emit("Recording... (3s window)", 3.2)
    emit("[DBG] Max magnitude idx=66 val=1.10 | active_start=11 (thr=0.08g)", 0.6)
    emit("[DBG] Window start=0 end=255 (size=256)", 0.5)
    emit("[DBG] Window floats: 0.05 -0.02 0.06 0.01 -0.01 0.04 0.07 -0.02 0.00 0.05 ", 0.4)
    emit("[DBG] Window quantized: 0 0 1 -1 0 1 1 -1 0 1 ", 0.4)
    emit("Invocation started", 1.2)
    emit("Invocation finished", 0.5)
    emit("Predicted Gesture: ring", 0.4)
    emit("[APP1] class=ring conf=0.90", 0.4)
    emit("[TRIGGER] ring -> arm", 0.3)
    emit("[APP2] connecting to server ...", 1.1)
    emit("[APP2] Wi-Fi connected", 0.9)
    arm_payload = {
        "trigger_class": "ring",
        "trigger_conf": 0.9,
        "features": [1.0, 0.0, 0.0, 1.02, -0.02, 0.0, 0.98, 0.02, 0.0],
        "meta": {"fs": 100, "win": 32, "room": "lab-1"},
    }
    log_server_response(post_event(arm_payload))
    emit("[APP2] armed, streaming mode", 0.8)
    emit("Mode: armed", 0.8)

    # Wait 2 seconds, then light movement
    emit("Monitoring...", 2.0)
    emit("Recording... (3s window)", 3.2)
    emit("[DBG] Max magnitude idx=78 val=0.48 | active_start=14 (thr=0.08g)", 0.6)
    emit("[DBG] Window start=0 end=255 (size=256)", 0.5)
    emit("[DBG] Window floats: 0.02 0.04 0.06 -0.01 0.07 0.04 -0.03 0.02 0.05 0.02 ", 0.4)
    emit("[DBG] Window quantized: 0 0 1 0 1 1 0 0 1 0 ", 0.4)
    emit("[FEAT] rms=0.16 peak=0.48 crest=3.00 mean=0.01", 0.4)
    emit("Invocation started", 1.2)
    emit("Invocation finished", 0.5)
    emit("Predicted State: weak", 0.4)
    emit("[APP1] class=weak conf=0.72", 0.4)
    emit("[TRIGGER] activity", 0.3)
    emit("[APP2] connecting to server ...", 1.1)
    emit("[APP2] Wi-Fi connected", 0.9)
    light_payload = {
        "trigger_class": "activity",
        "trigger_conf": 0.72,
        "features": [1.2, 0.0, 0.0, 1.15, 0.0, 0.0, 1.1, 0.0, 0.0],
        "meta": {"fs": 100, "win": 32, "room": "lab-1", "armed": True},
    }
    log_server_response(post_event(light_payload))
    emit("[APP2] continue monitoring", 0.8)

    # Quiet window to return to secure while armed
    emit("Monitoring...", 2.0)
    emit("Recording... (3s window)", 3.2)
    emit("[DBG] Max magnitude idx=16 val=0.06 | active_start=5 (thr=0.08g)", 0.6)
    emit("[DBG] Window start=0 end=255 (size=256)", 0.5)
    emit("[DBG] Window floats: 0.01 -0.01 0.00 0.00 -0.01 0.01 0.00 0.00 0.00 0.01 ", 0.4)
    emit("[DBG] Window quantized: 0 0 0 0 0 0 0 0 0 0 ", 0.4)
    emit("[FEAT] rms=0.03 peak=0.06 crest=2.00 mean=0.00", 0.4)
    emit("Invocation started", 1.1)
    emit("Invocation finished", 0.5)
    emit("Predicted State: idle", 0.4)
    emit("[APP1] class=idle conf=0.90", 0.4)
    emit("[TRIGGER] activity", 0.3)
    emit("[APP2] connecting to server ...", 1.0)
    emit("[APP2] Wi-Fi connected", 0.9)
    idle_between_payload = {
        "trigger_class": "activity",
        "trigger_conf": 0.9,
        "features": [1.0, 0.0, 0.0, 0.99, 0.01, 0.0, 1.01, -0.01, 0.0],
        "meta": {"fs": 100, "win": 32, "room": "lab-1", "armed": True},
    }
    log_server_response(post_event(idle_between_payload))
    emit("[APP2] continue monitoring", 0.8)

    # Wait 2 seconds, then strong movement
    emit("Monitoring...", 2.0)
    emit("Recording... (3s window)", 3.2)
    emit("[DBG] Max magnitude idx=142 val=3.40 | active_start=9 (thr=0.08g)", 0.6)
    emit("[DBG] Window start=0 end=255 (size=256)", 0.5)
    emit("[DBG] Window floats: 0.12 -0.03 0.08 0.01 -0.02 0.05 0.09 -0.01 0.00 0.04 ", 0.4)
    emit("[DBG] Window quantized: -1 0 2 -3 1 0 1 -2 0 2 ", 0.4)
    emit("[FEAT] rms=0.46 peak=3.40 crest=7.39 mean=0.01", 0.4)
    emit("Invocation started", 1.3)
    emit("Invocation finished", 0.5)
    emit("Predicted State: strong", 0.4)
    emit("[APP1] class=strong conf=0.90", 0.4)
    emit("[TRIGGER] activity", 0.3)
    emit("[APP2] connecting to server ...", 1.1)
    emit("[APP2] Wi-Fi connected", 0.9)
    slam_payload = {
        "trigger_class": "activity",
        "trigger_conf": 0.9,
        "features": [1.8, 0.0, 0.0, 1.6, 0.0, 0.0, 1.7, 0.0, 0.0],
        "meta": {"fs": 100, "win": 32, "room": "lab-1", "armed": True},
    }
    log_server_response(post_event(slam_payload))
    emit("[APP2] continue monitoring", 0.8)


if __name__ == "__main__":
    main()

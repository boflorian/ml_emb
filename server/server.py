import json
import os
import time
import urllib.request
import urllib.error

from flask import Flask, request, jsonify

app = Flask(__name__)

def clamp01(x):
    return max(0.0, min(1.0, x))

def ml_infer(features):
    if not features:
        return {"quality": 0.5, "flags": ["no_features"]}

    energy = 0.0
    for v in features:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = 0.0
        energy += fv * fv

    flags = []
    if energy < 1.0:
        flags.append("low_energy")
        quality = 0.2
    elif energy > 200.0:
        flags.append("noisy")
        quality = 0.3
    else:
        flags.append("clean")
        quality = 0.85

    return {"quality": clamp01(quality), "flags": flags}

def llm_infer(trigger_class, trigger_conf, ml_out, meta):
    url = os.getenv("LLM_GATEWAY_URL")
    api_key = os.getenv("LLM_GATEWAY_KEY")
    flags = ml_out.get("flags", [])

    if not url:
        return fallback_message(flags)

    prompt = (
        "You are assisting a Pico W gesture demo. "
        "Return a single short instruction (max 160 chars), no lists, no markdown. "
        f"Trigger={trigger_class} conf={trigger_conf:.2f}. "
        f"Quality={ml_out.get('quality', 0):.2f} flags={flags}. "
        f"Meta={meta}."
    )

    payload = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(raw)
            for key in ("message", "text", "response"):
                if key in data and isinstance(data[key], str):
                    return data[key][:160]
    except (urllib.error.URLError, ValueError, json.JSONDecodeError):
        return fallback_message(flags)

    return fallback_message(flags)

def fallback_message(flags):
    if "noisy" in flags:
        return "Try smaller motion and keep steady."
    if "low_energy" in flags:
        return "Make the gesture larger and slower."
    return "Gesture looks OK. Repeat once."

@app.get("/api/v1/healthcheck")
def healthcheck():
    return jsonify(data="healthy")

@app.post("/api/v1/greet")
def greet():
    name = request.json.get("name", "World")
    return jsonify(data=f"Hello, {name}!")

@app.post("/api/v1/predict_a")
def predict_a():
    x = float(request.json.get("x", 0.0))
    return jsonify(data=str(2 * x + 1))

@app.post("/api/gesture_event")
def gesture_event():
    payload = request.get_json(silent=True) or {}
    trigger_class = payload.get("trigger_class", "unknown")
    trigger_conf = float(payload.get("trigger_conf", 0.0) or 0.0)
    features = payload.get("features", [])
    meta = payload.get("meta", {})

    ml_out = ml_infer(features)
    message = llm_infer(trigger_class, trigger_conf, ml_out, meta)

    print(
        f"[SERVER] t={int(time.time())} class={trigger_class} "
        f"conf={trigger_conf:.2f} ml={ml_out} msg={message}"
    )

    return jsonify(
        result="ok",
        ml=ml_out,
        message=message,
    )

if __name__ == "__main__":
    print("Server listening on 0.0.0.0:8000")
    print('Example: curl -X POST http://localhost:8000/api/gesture_event -H "Content-Type: application/json" -d \'{"trigger_class":"ring","trigger_conf":0.9,"features":[0.1,0.2]}\'' )
    app.run(host="0.0.0.0", port=8000)

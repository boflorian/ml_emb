import json
import os
import time
import urllib.request
import urllib.error

from flask import Flask, request, jsonify

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARMED = False
LAST_EVENT = {
    "ts": 0,
    "label": "idle",
    "flags": ["no_data"],
    "quality": 0.0,
    "message": "Waiting for activity.",
    "meta": {},
}

def clamp01(x):
    return max(0.0, min(1.0, x))

def ml_infer(features):
    if not features:
        return {"quality": 0.4, "flags": ["no_features"], "label": "unknown"}

    vals = []
    for v in features:
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals.append(0.0)

    n = len(vals)
    energy = sum(v * v for v in vals)
    rms = (energy / n) ** 0.5 if n else 0.0
    peak = max((abs(v) for v in vals), default=0.0)
    mean = sum(vals) / n if n else 0.0
    abs_mean = sum(abs(v) for v in vals) / n if n else 0.0
    crest = (peak / rms) if rms > 1e-6 else 0.0
    impulse = (peak / abs_mean) if abs_mean > 1e-6 else 0.0

    mags = []
    for i in range(0, n - 2, 3):
        x, y, z = vals[i], vals[i + 1], vals[i + 2]
        mags.append((x * x + y * y + z * z) ** 0.5)
    mag_dev = [abs(m - 1.0) for m in mags]
    rms_mag = (sum(d * d for d in mag_dev) / len(mag_dev)) ** 0.5 if mag_dev else 0.0
    peak_mag = max(mag_dev, default=0.0)

    flags = []
    if n < 8:
        flags.append("short_window")
        quality = 0.3
        label = "unknown"
    elif rms_mag < 0.03:
        flags.append("quiet")
        quality = 0.8
        label = "idle"
    elif peak_mag > 0.6 or rms_mag > 0.3:
        flags.append("impact")
        quality = 0.85
        label = "strong"
    #elif peak_mag > 0.35 or rms_mag > 0.18:
    #    flags.append("moving")
    #    quality = 0.8
    #    label = "weak movement"
    #elif peak_mag > 0.2 or rms_mag > 0.1:
    #    flags.append("shift")
    #    quality = 0.75
    #    label = "weak movement"
    else:
        flags.append("light")
        quality = 0.7
        label = "weak"

    return {
        "quality": clamp01(quality),
        "flags": flags,
        "label": label,
        "rms": round(rms, 3),
        "peak": round(peak, 3),
        "crest": round(crest, 2),
        "mean": round(mean, 3),
        "impulse": round(impulse, 2),
        "rms_mag": round(rms_mag, 3),
        "peak_mag": round(peak_mag, 3),
    }

def llm_infer(trigger_class, trigger_conf, ml_out, meta):
    url = os.getenv("LLM_GATEWAY_URL")
    api_key = os.getenv("LLM_GATEWAY_KEY")
    flags = ml_out.get("flags", [])
    label = ml_out.get("label", "unknown")

    if not url:
        return fallback_message(label, flags)

    prompt = (
        "You are assisting a Pico W room activity security demo. "
        "Return a single short instruction (max 160 chars), no lists, no markdown. "
        f"Trigger={trigger_class} conf={trigger_conf:.2f}. "
        f"Quality={ml_out.get('quality', 0):.2f} flags={flags} label={label}. "
        f"Stats=rms:{ml_out.get('rms')} peak:{ml_out.get('peak')} crest:{ml_out.get('crest')}. "
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
        return fallback_message(label, flags)

    return fallback_message(label, flags)

def fallback_message(label, flags):
    if "short_window" in flags or "no_features" in flags:
        return "Not enough data. Hold steady for one more window."
    if label == "idle":
        return "Monitoring. Room is quiet."
    if label == "weak":
        return "Watch. Light movement detected."
    if label == "strong":
        return "Alert. Strong movement detected."
    return "Activity detected. Please check."

@app.get("/api/v1/healthcheck")
def healthcheck():
    return jsonify(data="healthy")

@app.get("/api/activity_status")
def activity_status():
    global ARMED
    payload = dict(LAST_EVENT)
    now = int(time.time())
    connected = (now - payload.get("ts", 0)) <= 5
    if not connected:
        ARMED = False
    payload["connected"] = connected
    payload["armed"] = ARMED
    return jsonify(payload)

@app.get("/dashboard")
def dashboard():
    path = os.path.join(BASE_DIR, "dashboard.html")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()

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
    global ARMED
    payload = request.get_json(silent=True) or {}
    trigger_class = payload.get("trigger_class", "unknown")
    trigger_conf = float(payload.get("trigger_conf", 0.0) or 0.0)
    features = payload.get("features", [])
    if not features:
        features = payload.get("samples", [])
    meta = payload.get("meta", {})

    if trigger_class == "ring":
        ARMED = True
    elif trigger_class == "disarm" or meta.get("armed") is False:
        ARMED = False

    ml_out = ml_infer(features)
    message = llm_infer(trigger_class, trigger_conf, ml_out, meta)

    command = None
    if ARMED and ml_out.get("label") == "strong":
        command = {"type": "display", "text": "Alert"}
    elif ARMED and ml_out.get("label") == "weak":
        command = {"type": "display", "text": "Watch"}

    LAST_EVENT.update(
        {
            "ts": int(time.time()),
            "label": ml_out.get("label", "unknown"),
            "flags": ml_out.get("flags", []),
            "quality": ml_out.get("quality", 0.0),
            "message": message,
            "meta": meta,
        }
    )

    print(
        f"[SERVER] t={int(time.time())} class={trigger_class} "
        f"conf={trigger_conf:.2f} ml={ml_out} msg={message}"
    )

    return jsonify(
        result="ok",
        ml=ml_out,
        message=message,
        command=command,
    )

if __name__ == "__main__":
    print("Server listening on 0.0.0.0:8000")
    print('Example: curl -X POST http://localhost:8000/api/gesture_event -H "Content-Type: application/json" -d \'{"trigger_class":"activity","trigger_conf":0.9,"features":[0.3,0.7,1.1,0.9,0.4,0.6,0.8,0.2],"meta":{"fs":400,"win":256,"room":"lab-1"}}\'' )
    app.run(host="0.0.0.0", port=8000)

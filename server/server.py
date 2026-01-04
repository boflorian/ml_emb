import json
import os
import time
import urllib.request
import urllib.error

from flask import Flask, request, jsonify

app = Flask(__name__)

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

    flags = []
    if n < 8:
        flags.append("short_window")
        quality = 0.3
        label = "unknown"
    elif rms < 0.06:
        flags.append("quiet")
        quality = 0.75
        label = "idle"
    elif peak > 2.8 and crest > 6.0:
        flags.append("impact")
        quality = 0.7
        label = "door_slam"
    elif rms > 0.35 and 3.0 <= crest <= 5.0:
        flags.append("moving")
        quality = 0.75
        label = "footsteps"
    elif rms > 0.18 and crest > 5.0:
        flags.append("shift")
        quality = 0.7
        label = "object_move"
    else:
        flags.append("light")
        quality = 0.7
        label = "light_activity"

    return {
        "quality": clamp01(quality),
        "flags": flags,
        "label": label,
        "rms": round(rms, 3),
        "peak": round(peak, 3),
        "crest": round(crest, 2),
        "mean": round(mean, 3),
        "impulse": round(impulse, 2),
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
        return "Room is quiet. Monitoring continues."
    if label == "footsteps":
        return "Movement detected. Check the area."
    if label == "object_move":
        return "Object shift detected. Verify the room."
    if label == "door_slam":
        return "Strong impact detected. Possible entry."
    if label == "light_activity":
        return "Light activity detected. Stay alert."
    return "Activity detected. Please check."

@app.get("/api/v1/healthcheck")
def healthcheck():
    return jsonify(data="healthy")

@app.get("/api/activity_status")
def activity_status():
    return jsonify(LAST_EVENT)

@app.get("/dashboard")
def dashboard():
    html = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Room Activity Monitor</title>
  <style>
    :root {
      --bg1: #0b1320;
      --bg2: #12243a;
      --accent: #f4c430;
      --alert: #f05454;
      --warn: #f2a340;
      --ok: #4dd599;
      --text: #e9eef7;
      --muted: #a9b4c7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Alegreya Sans", "Segoe UI", Arial, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at 20% 10%, #1c2f4a, var(--bg1));
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .panel {
      width: min(900px, 96vw);
      background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(0,0,0,0.2));
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 18px;
      padding: 28px;
      box-shadow: 0 18px 40px rgba(0,0,0,0.45);
    }
    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 20px;
    }
    .title {
      font-size: 28px;
      letter-spacing: 0.6px;
    }
    .badge {
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
      color: var(--muted);
      font-size: 13px;
    }
    .status {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 18px;
    }
    .card {
      background: rgba(12, 18, 28, 0.7);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 20px;
      min-height: 140px;
    }
    .label {
      font-size: 14px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 1.2px;
    }
    .value {
      font-size: 30px;
      margin-top: 10px;
      font-weight: 600;
    }
    .message {
      font-size: 18px;
      margin-top: 12px;
      color: var(--text);
      line-height: 1.4;
    }
    .alert {
      color: var(--alert);
    }
    .warn {
      color: var(--warn);
    }
    .ok {
      color: var(--ok);
    }
    .pulse {
      position: relative;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: var(--ok);
      box-shadow: 0 0 0 0 rgba(77, 213, 153, 0.7);
      animation: pulse 2s infinite;
    }
    .pulse.alert {
      background: var(--alert);
      box-shadow: 0 0 0 0 rgba(240, 84, 84, 0.7);
    }
    .pulse.warn {
      background: var(--warn);
      box-shadow: 0 0 0 0 rgba(242, 163, 64, 0.7);
    }
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(77, 213, 153, 0.7); }
      70% { box-shadow: 0 0 0 18px rgba(77, 213, 153, 0); }
      100% { box-shadow: 0 0 0 0 rgba(77, 213, 153, 0); }
    }
    @media (max-width: 720px) {
      .status { grid-template-columns: 1fr; }
      .value { font-size: 26px; }
    }
  </style>
  <link href="https://fonts.googleapis.com/css2?family=Alegreya+Sans:wght@400;600&display=swap" rel="stylesheet">
</head>
<body>
  <div class="panel">
    <div class="header">
      <div class="title">Room Activity Monitor</div>
      <div class="badge">Pico IMU + Server Analysis</div>
    </div>
    <div class="status">
      <div class="card">
        <div class="label">Latest Activity</div>
        <div class="value" id="label">Waiting...</div>
        <div class="message" id="message">No data yet.</div>
      </div>
      <div class="card">
        <div class="label">Status</div>
        <div class="value">
          <span id="statusText" class="ok">Secure</span>
        </div>
        <div class="message">
          <span class="pulse" id="pulse"></span>
          <span id="timestamp">--</span>
        </div>
      </div>
    </div>
  </div>
  <script>
    const labelEl = document.getElementById("label");
    const messageEl = document.getElementById("message");
    const statusText = document.getElementById("statusText");
    const pulse = document.getElementById("pulse");
    const tsEl = document.getElementById("timestamp");

    const alertLabels = new Set(["door_slam", "object_move"]);
    const warnLabels = new Set(["footsteps", "light_activity"]);

    function render(data) {
      labelEl.textContent = data.label || "unknown";
      messageEl.textContent = data.message || "No message.";
      const ts = data.ts ? new Date(data.ts * 1000) : null;
      tsEl.textContent = ts ? ts.toLocaleTimeString() : "--";

      if (alertLabels.has(data.label)) {
        statusText.textContent = "Alert";
        statusText.className = "alert";
        pulse.className = "pulse alert";
      } else if (warnLabels.has(data.label)) {
        statusText.textContent = "Watch";
        statusText.className = "warn";
        pulse.className = "pulse warn";
      } else {
        statusText.textContent = "Secure";
        statusText.className = "ok";
        pulse.className = "pulse";
      }
    }

    async function poll() {
      try {
        const resp = await fetch("/api/activity_status");
        const data = await resp.json();
        render(data);
      } catch (err) {
        messageEl.textContent = "Unable to reach server.";
      }
    }

    poll();
    setInterval(poll, 2000);
  </script>
</body>
</html>
"""
    return html

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

    command = None
    if ml_out.get("label") == "door_slam":
        command = {"type": "display", "text": "Possible entry"}
    elif ml_out.get("label") == "object_move":
        command = {"type": "display", "text": "Check room"}
    elif ml_out.get("label") == "footsteps":
        command = {"type": "display", "text": "Movement"}

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

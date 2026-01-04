#!/usr/bin/env python3
import time
import sys

def emit(line, delay=0.4):
    print(line)
    sys.stdout.flush()
    time.sleep(delay)


def main():
    # Clear terminal like `screen`
    print("\033[2J\033[H", end="")
    sys.stdout.flush()

    emit("System initialized", 0.6)
    emit("Sleep done", 0.6)
    emit("IMU initialized", 0.5)
    emit("LCD initialized", 0.5)
    emit("Core1 launched", 0.5)
    emit("Model initialized", 0.5)
    emit("Input quantization: scale=0.082000, zero_point=-3", 0.5)
    emit("Usecase: room activity security monitor", 0.6)

    # First classification: idle
    emit("Recording...", 0.7)
    emit("[DBG] Max magnitude idx=35 val=0.09 | active_start=6 (thr=0.08g)", 0.5)
    emit("[DBG] Window start=0 end=255 (size=256)", 0.4)
    emit("[DBG] Window floats: 0.03 -0.02 0.01 0.00 -0.01 0.02 0.01 0.00 0.00 0.01 ", 0.3)
    emit("[DBG] Window quantized: 0 0 1 -1 0 1 0 0 0 0 ", 0.3)
    emit("[FEAT] rms=0.04 peak=0.09 crest=2.25 mean=0.00", 0.3)
    emit("Invocation started", 0.2)
    emit("Invocation finished", 0.2)
    emit("Output type=1 dims: 1 3", 0.2)
    emit("Output scores (int8): 84 -12 -30 ", 0.2)
    emit("Bias-corrected scores: 124 28 -50 ", 0.2)
    emit("Top: class 0 (124), Second: class 1 (28), Margin: 96", 0.2)
    emit("Predicted State: 0 (idle)", 0.3)
    emit("[APP1] class=idle conf=0.86", 0.3)

    # Second classification: footsteps
    emit("Recording...", 0.7)
    emit("[DBG] Max magnitude idx=78 val=0.95 | active_start=14 (thr=0.08g)", 0.5)
    emit("[DBG] Window start=0 end=255 (size=256)", 0.4)
    emit("[DBG] Window floats: 0.08 0.10 0.15 -0.03 0.20 0.12 -0.08 0.05 0.11 0.04 ", 0.3)
    emit("[DBG] Window quantized: 1 1 2 -1 3 2 -1 1 1 0 ", 0.3)
    emit("[FEAT] rms=0.42 peak=1.10 crest=2.62 mean=0.02", 0.3)
    emit("Invocation started", 0.2)
    emit("Invocation finished", 0.2)
    emit("Output type=1 dims: 1 3", 0.2)
    emit("Output scores (int8): -12 96 -30 ", 0.2)
    emit("Bias-corrected scores: 28 136 -70 ", 0.2)
    emit("Top: class 1 (136), Second: class 0 (28), Margin: 108", 0.2)
    emit("Predicted State: 1 (footsteps)", 0.3)
    emit("[APP1] class=footsteps conf=0.78", 0.3)

    # Third classification: door slam trigger
    emit("Recording...", 0.7)
    emit("[DBG] Max magnitude idx=142 val=3.40 | active_start=9 (thr=0.08g)", 0.5)
    emit("[DBG] Window start=0 end=255 (size=256)", 0.4)
    emit("[DBG] Window floats: 0.12 -0.03 0.08 0.01 -0.02 0.05 0.09 -0.01 0.00 0.04 ", 0.3)
    emit("[DBG] Window quantized: -1 0 2 -3 1 0 1 -2 0 2 ", 0.3)
    emit("[FEAT] rms=0.46 peak=3.40 crest=7.39 mean=0.01", 0.3)
    emit("Invocation started", 0.2)
    emit("Invocation finished", 0.2)
    emit("Output type=1 dims: 1 3", 0.2)
    emit("Output scores (int8): -28 -4 114 ", 0.2)
    emit("Bias-corrected scores: 12 36 154 ", 0.2)
    emit("Top: class 2 (154), Second: class 1 (36), Margin: 118", 0.2)
    emit("Predicted State: 2 (door_slam)", 0.3)
    emit("[APP1] class=door_slam conf=0.90", 0.3)
    emit("[TRIGGER] activity", 0.2)
    emit("[APP2] connecting to server ...", 0.4)
    emit("[APP2] Wi-Fi connected", 0.4)
    emit("[APP2] response message: Strong impact detected. Possible entry.", 0.5)
    emit("[APP2] done -> back to APP1", 0.4)

    emit("Recording...", 0.6)

    emit("[SERVER] t=1730000000 class=activity conf=0.90 ml={'quality': 0.7, 'flags': ['impact'], 'label': 'door_slam', 'rms': 0.46, 'peak': 3.4, 'crest': 7.39, 'mean': 0.01, 'impulse': 5.82} msg=Strong impact detected. Possible entry.", 0.2)
    emit("[SERVER] command={'type': 'display', 'text': 'Possible entry'}", 0.2)


if __name__ == "__main__":
    main()

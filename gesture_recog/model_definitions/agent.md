# Notes for CNN (IMU gestures)

## Current CNN (as of this note)
- LayerNorm input → Conv1D residual blocks (64 k9, 64 k5 w/ residual, 96 k3 residual, 128 k3 dilated).
- Pool after first two blocks; optional SpatialDropout1D tied to `dropout`; Dense 256 + BN + Dropout; softmax.
- Optimizer in `build_cnn`: AdamW (lr from cfg, wd=1e-4, clipnorm=1.0), loss: sparse CE.

## Two-phase optimizer idea (manual run, no main.py edits)
1) Warm-up with AdamW: lr=1e-3, wd=1e-4, clipnorm=1.0 for ~10–30 epochs.
2) Switch to SGD+momentum: lr=5e-4 (try 2e-4 if jumpy), momentum=0.9, nesterov=True, no weight decay; continue training for ~10–20 epochs.
3) Keep loss/metrics same; callbacks (ES/CKPT) work across the switch. Recompile with SGD before phase 2; weights stay.

Skeleton:
```python
model = build_cnn(win=64, num_classes=4, lr=1e-3, l2=1e-4, dropout=0.25)
model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3, weight_decay=1e-4, clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])
model.fit(train_ds, val_ds, epochs=E1, ...)

sgd = tf.keras.optimizers.SGD(learning_rate=5e-4, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])
model.fit(train_ds, val_ds, initial_epoch=E1, epochs=E1+E2, ...)
```

## Quick tuning knobs
- Dropout: 0.25–0.35; if overfitting, bump; if underfitting, lower.
- LR scheduling: cosine decay/restarts on AdamW can add 1–2% if plateaus early.
- Label smoothing: 0.05 for sparse CE to reduce overconfidence.
- Window: try `win=80–96` (hop 64) if longer gestures; watch latency. 
- Class balance: use class weights or resampling if some gestures are rare.

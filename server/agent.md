# Agent Guide: server.py and lol.py

Scope: Only modify `server.py` and `lol.py` for the simulated server use case and terminal output demo.

Goals:
- Make `server/server.py` simulate a realistic local use case 
- Keep API contract stable for Pico W: always return top-level `message`.
- Keep outputs short and Pico-friendly (<=160 chars), no lists/markdown.
- Ensure `lol.py` reflects the same server behavior and logs.

Key behavior to preserve:
- Endpoint `POST /api/gesture_event` on `0.0.0.0:8000`
- Response JSON includes `result`, `ml`, `message`
- Optional `command` is OK if used by the firmware

Do not change:
- Firmware-related files outside these two targets
- API path names or required top-level keys

Notes:
- LLM gateway call should remain optional (fallback to deterministic messages).
- Keep logic simple and deterministic if no LLM gateway configured.

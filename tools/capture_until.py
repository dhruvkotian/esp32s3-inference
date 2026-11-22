#!/usr/bin/env python3
"""Capture UART0 from the S3-CAM until a marker appears or a timeout expires.

Same reset sequence as capture.py: the CH343 bridge has no auto-reset wiring
that matches esptool's inverted DTR/RTS sequence, so EN is pulsed explicitly.
"""
import sys
import time

import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
TIMEOUT = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0
OUT = sys.argv[3] if len(sys.argv) > 3 else "serial.log"
MARKER = (sys.argv[4] if len(sys.argv) > 4 else "BENCH_DONE").encode()

ser = serial.Serial(PORT, 115200, timeout=0.2)
ser.setDTR(False)
ser.setRTS(True)
time.sleep(0.15)
ser.setRTS(False)
time.sleep(0.1)

t0 = time.time()
buf = b""
with open(OUT, "wb") as f:
    while time.time() - t0 < TIMEOUT:
        chunk = ser.read(4096)
        if chunk:
            f.write(chunk)
            f.flush()
            sys.stdout.write(chunk.decode("utf-8", errors="replace"))
            sys.stdout.flush()
            buf += chunk
            if MARKER in buf:
                break
ser.close()
print(f"\n--- captured to {OUT} ---")

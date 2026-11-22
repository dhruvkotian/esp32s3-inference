#!/usr/bin/env python3
"""Capture UART0 output from the S3-CAM through its CH343 USB-UART bridge.

The board has no auto-reset wiring on DTR/RTS that matches esptool's default
inverted sequence for native USB, so reset is done explicitly:
  RTS asserted (EN low) -> DTR cleared (IO0 high) -> RTS released (EN high)
"""
import sys
import time

import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
OUT = sys.argv[3] if len(sys.argv) > 3 else "serial.log"

ser = serial.Serial(PORT, 115200, timeout=0.2)
ser.setDTR(False)
ser.setRTS(True)
time.sleep(0.15)
ser.setRTS(False)
time.sleep(0.1)

t0 = time.time()
with open(OUT, "wb") as f:
    while time.time() - t0 < SECS:
        chunk = ser.read(4096)
        if chunk:
            f.write(chunk)
            f.flush()
            sys.stdout.write(chunk.decode("utf-8", errors="replace"))
            sys.stdout.flush()
ser.close()
print(f"\n--- captured {SECS:.0f}s to {OUT} ---")

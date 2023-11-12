from collections import deque
import numpy as np
import pyaudio as pa

def play_note(freq: float=440.0) -> None:
    SR = 44100  # sampling ratio
    omega = 2 * np.pi * freq
    dur = 12
    buflen = int(SR * dur)
    buf = np.zeros(buflen)
    dlen = int(SR / freq)  # Ligne retardement initial
    dq = deque(np.random.random(dlen) * 2 - 1, dlen)
    x = dq.pop()
    for i in range(buflen):
        y = dq.pop()
        buf[i] = val = 0.500 * (x + y)
        dq.appendleft(val)
        x = y
    chunk = 2048
    p = pa.PyAudio()
    stream = p.open(format=pa.paFloat32, channels=1, rate=SR, output=True, frames_per_buffer=chunk)
    stream.write(buf.astype(np.float32))
    stream.close()
    p.terminate()


play_note(164.81)
play_note(196.00)
play_note(246.94)
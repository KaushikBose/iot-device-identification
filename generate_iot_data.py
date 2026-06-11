import os
import numpy as np
from scipy import signal as sp
from scipy.signal import butter, lfilter

np.random.seed(42)

fs = 10e6
N = 500_000
t = np.arange(N, dtype=np.float64) / fs

def save(cls, scenario, sig):
    os.makedirs(f"data/recordings/{cls}", exist_ok=True)

    path = f"data/recordings/{cls}/{scenario}_capture01.npy"
    np.save(path, sig.astype(np.float32))

    f, _, Sxx = sp.spectrogram(
        sig[:4096], fs=fs, nperseg=256, noverlap=128
    )

    f_peak = f[np.argmax(np.abs(Sxx).mean(axis=1))] / 1e3
    bw = (
        np.sum(
            np.abs(Sxx).mean(axis=1)
            > np.abs(Sxx).mean(axis=1).max() * 0.1
        )
        * (fs / 256 / 1e3)
    )

    pw = 10 * np.log10(np.mean(sig**2) + 1e-12)

    print(
        f"{cls:12s} {scenario:10s} "
        f"peak={f_peak:+6.1f}kHz  "
        f"bw={bw:5.1f}kHz  "
        f"pwr={pw:.1f}dBW"
    )

def add_noise(sig, snr_db):
    spwr = np.mean(sig**2) + 1e-12
    npwr = spwr / (10 ** (snr_db / 10))
    return sig + np.sqrt(npwr) * np.random.randn(len(sig))

scenarios = {
    "baseline": 25,
    "background": 10,
    "upstairs": 5,
}

for scenario, snr in scenarios.items():

    # DOORALARM
    fo = -100e3
    syms = np.repeat(
        np.random.randint(0, 2, N // 20000 + 1),
        20000
    )[:N].astype(np.float64)

    sig = 0.0002 * syms * np.cos(2 * np.pi * fo * t)
    save("dooralarm", scenario, add_noise(sig, snr))

    # LORA
    bw = 125e3
    f0 = 50e3
    T_sym = 4096 / fs

    phase = 2 * np.pi * (
        f0 * t +
        (bw / (2 * T_sym)) * np.mod(t, T_sym) ** 2
    )

    sig = 0.0015 * np.cos(phase)
    save("lora", scenario, add_noise(sig, snr))

    # MICROPHONE
    white = np.random.randn(N)

    b, a = butter(4, [0.02, 0.18], btype="band")
    sig = 0.0008 * lfilter(b, a, white)

    env = np.abs(np.sin(2 * np.pi * 800 * t)) ** 0.5
    sig *= (0.3 + 0.7 * env)

    save("microphone", scenario, add_noise(sig, snr))

    # MBUS
    sym_len = int(fs / 4800)

    bits = np.repeat(
        np.random.randint(0, 2, N // sym_len + 1),
        sym_len
    )[:N]

    freq = 150e3 + bits * 50e3
    phase = 2 * np.pi * np.cumsum(freq) / fs

    sig = 0.0003 * np.cos(phase)
    save("mbus", scenario, add_noise(sig, snr))

    # SIGFOX
    fo = 250e3
    sym_len = 2000

    syms = np.repeat(
        np.random.choice([-1, 1], N // sym_len + 1),
        sym_len
    )[:N].astype(np.float64)

    sig = 0.0020 * syms * np.cos(2 * np.pi * fo * t)
    save("sigfox", scenario, add_noise(sig, snr))

    # MIWI
    fo = -200e3

    barker = np.array(
        [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
        dtype=np.float64
    )

    chips = np.tile(barker, N // 11 + 1)[:N]

    data = np.repeat(
        np.random.choice([-1, 1], N // 110 + 1),
        110
    )[:N].astype(np.float64)

    sig = 0.0005 * chips * data * np.cos(2 * np.pi * fo * t)

    save("miwi", scenario, add_noise(sig, snr))

print("\nAll 18 files regenerated — each class now occupies a unique frequency region.")
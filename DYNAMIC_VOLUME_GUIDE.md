# Dynamic Volume Control Guide

## Overview
The script now allows you to define separate volume envelopes for left and right channels that change at each time step.

---

## Basic Concept

```python
def left_volume_envelope(t):
    """t is time in seconds, return volume 0.0 to 1.0"""
    return 0.5  # Example: constant 0.5

def right_volume_envelope(t):
    """t is time in seconds, return volume 0.0 to 1.0"""
    return 0.3  # Example: constant 0.3
```

The script applies these to every sample:
```python
time_vector = np.arange(len(long_signal)) / fs
left_volumes = np.array([left_volume_envelope(t) for t in time_vector])
right_volumes = np.array([right_volume_envelope(t) for t in time_vector])

stereo_signal = np.column_stack((
    long_signal * left_volumes,
    long_signal * right_volumes
))
```

---

## Volume Envelope Examples

### 1. **Constant Volume** (Simplest)
```python
def left_volume_envelope(t):
    return 0.5

def right_volume_envelope(t):
    return 0.2
```

---

### 2. **Linear Fade In / Fade Out**
```python
def left_volume_envelope(t):
    # Fade in: 0 → 1.0 over 10 seconds
    return np.clip(t / 10, 0.0, 1.0)

def right_volume_envelope(t):
    # Fade out: 1.0 → 0 over last 5 seconds
    fade_start = target_duration - 5
    if t < fade_start:
        return 1.0
    else:
        return np.clip(1.0 - (t - fade_start) / 5, 0.0, 1.0)
```

---

### 3. **Step Changes at Specific Times**
```python
def left_volume_envelope(t):
    if t < 10:
        return 0.3
    elif t < 20:
        return 0.6
    elif t < 30:
        return 0.9
    else:
        return 0.3

def right_volume_envelope(t):
    return 0.2  # Right stays constant
```

---

### 4. **Sine Wave Modulation**
```python
def left_volume_envelope(t):
    # Oscillates between 0.4 and 0.8 at 1 Hz
    center = 0.6
    amplitude = 0.2
    return center + amplitude * np.sin(2 * np.pi * 1.0 * t)

def right_volume_envelope(t):
    # Oscillates at 0.5 Hz (half the left frequency)
    center = 0.3
    amplitude = 0.1
    return center + amplitude * np.sin(2 * np.pi * 0.5 * t)
```

---

### 5. **Exponential Decay**
```python
def left_volume_envelope(t):
    # Decays exponentially with time constant 10 seconds
    return np.exp(-t / 10)

def right_volume_envelope(t):
    # Decays slower with time constant 20 seconds
    return np.exp(-t / 20)
```

---

### 6. **Cosine Taper (Smooth)**
```python
def left_volume_envelope(t):
    # Smooth fade in/out using cosine
    return 0.5 * (1 - np.cos(np.pi * t / target_duration))

def right_volume_envelope(t):
    # Opposite: starts high, fades down
    return 0.5 * (1 + np.cos(np.pi * t / target_duration))
```

---

### 7. **Piecewise Linear Ramp**
```python
def left_volume_envelope(t):
    if t < 5:
        # Ramp up from 0 to 0.5
        return 0.5 * (t / 5)
    elif t < 15:
        # Hold at 0.5
        return 0.5
    elif t < 20:
        # Ramp down from 0.5 to 0
        return 0.5 * (1 - (t - 15) / 5)
    else:
        # Off
        return 0.0

def right_volume_envelope(t):
    # Inverted: off when left is on
    left_val = left_volume_envelope(t)
    return 0.3 if left_val < 0.1 else 0.0
```

---

### 8. **Chirp-like Modulation (Frequency Sweep)**
```python
def left_volume_envelope(t):
    # Volume "frequency" increases over time
    freq_sweep = 0.1 + 2.0 * (t / target_duration)  # 0.1 to 2.1 Hz
    return 0.5 + 0.3 * np.sin(2 * np.pi * freq_sweep * t)

def right_volume_envelope(t):
    return 0.2  # Constant
```

---

### 9. **Square Wave (On/Off Pattern)**
```python
def left_volume_envelope(t):
    # Toggle every 2 seconds
    cycle_time = t % 4  # Period = 4 seconds
    return 0.8 if cycle_time < 2 else 0.1

def right_volume_envelope(t):
    # Opposite phase: off when left is on
    cycle_time = t % 4
    return 0.1 if cycle_time < 2 else 0.8
```

---

### 10. **Envelope with Noise-like Modulation**
```python
def left_volume_envelope(t):
    # Deterministic "noise-like" using sine harmonics
    v = 0.5
    v += 0.15 * np.sin(2 * np.pi * 0.3 * t)
    v += 0.10 * np.sin(2 * np.pi * 0.7 * t)
    v += 0.05 * np.sin(2 * np.pi * 1.9 * t)
    return np.clip(v, 0.0, 1.0)

def right_volume_envelope(t):
    # Similar but different phase
    v = 0.3
    v += 0.10 * np.sin(2 * np.pi * 0.5 * t + np.pi/4)
    v += 0.08 * np.sin(2 * np.pi * 1.1 * t + np.pi/3)
    return np.clip(v, 0.0, 1.0)
```

---

## CSV Output

The CSV now includes two additional columns:

```
timestamp_iso,sample_index,time_seconds,left_level,right_level,left_volume,right_volume
2024-05-08T14:30:45.123456,0,0.000000,0.000000,0.000000,0.200000,0.200000
2024-05-08T14:30:45.123456,1920,0.010000,0.234567,0.046913,0.203333,0.199975
2024-05-08T14:30:45.123456,3840,0.020000,0.512345,0.102469,0.206667,0.199999
```

- **left_volume**: The envelope multiplier applied to left channel at that time
- **right_volume**: The envelope multiplier applied to right channel at that time
- **left_level**: The actual amplitude = `left_signal * left_volume`
- **right_level**: The actual amplitude = `right_signal * right_volume`

---

## How to Customize

1. **Open the script**
2. **Find these functions:**
   ```python
   def left_volume_envelope(t):
       ...
   
   def right_volume_envelope(t):
       ...
   ```

3. **Replace with your desired envelope** (pick from examples above or create your own)

4. **Run:**
   ```bash
   python sound_generator_dynamic.py
   ```

5. **Check the CSV** to see what volumes were applied at each timestep

---

## Tips

- **Always return values between 0.0 and 1.0** (or use `np.clip()`)
- **Use `target_duration`** variable to make envelopes responsive to signal length
- **Common issue**: If you reference a time >= `target_duration`, handle it gracefully
- **Performance**: The envelope is calculated once per sample (192,000 times per second), so keep it simple
- **Combine functions**: You can add multiple effects together and clip the result

```python
def left_volume_envelope(t):
    # Base linear ramp
    base = 0.5 * (t / target_duration)
    # Add sine modulation
    mod = 0.2 * np.sin(2 * np.pi * 2 * t)
    # Combine and clip
    return np.clip(base + mod, 0.0, 1.0)
```

---

## Debugging

Print the volume ranges to verify:
```python
print(f"Left volume range: {left_volumes.min():.4f} to {left_volumes.max():.4f}")
print(f"Right volume range: {right_volumes.min():.4f} to {right_volumes.max():.4f}")
```

This helps confirm your envelope functions are working as expected!

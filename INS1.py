import numpy as np
import scipy.io as sio
from scipy.signal import iirnotch, butter, filtfilt
from scipy.fft import fft
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# =============================================================================
# Task 1.1 - Load and Convert to mV
# =============================================================================

# Load both .mat files
data1 = sio.loadmat('Data_1.mat')
data2 = sio.loadmat('Data_2.mat')

# Extract variables from HD-sEMG data
EMG1 = data1['EMG']  # HD-sEMG (64 channels)
fsamp1 = float(data1['fsamp'].flat[0])
force = data1['performed_data'].flatten()

# Transpose so shape is (samples × channels)
EMG1 = EMG1.T

# Extract variables from iEMG data
EMG2 = data2['EMG']
fsamp2 = float(data2['SamplingFrequency'].flat[0])

# Convert HD-sEMG from ADC counts to mV
conv_factor_sEMG = (5 / (2 ** 16) / 150) * 1000
EMG1_mV = EMG1 * conv_factor_sEMG

# Transpose iEMG to (samples × channels)
EMG2 = EMG2.T

# =============================================================================
# Task 1.2 - Visualize Raw EMG Signals
# =============================================================================

t1 = np.arange(EMG1_mV.shape[0]) / fsamp1
t2 = np.arange(EMG2.shape[0]) / fsamp2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# HD-sEMG: first 10 channels
ax = axes[0]
offset1 = 4.0
for ch in range(10):
    ax.plot(t1, EMG1_mV[:, ch] + ch * offset1, label=f'Ch{ch + 1}')
ax.set_title('Task 1.2: HD-sEMG - First 10 Channels')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (mV)')
ax.legend(fontsize=6, ncol=2)
ax.grid(True)

# iEMG: 3 channels
ax = axes[1]
offset2 = 0.15
channel_labels = ['FDS', 'ED distal', 'ED proximal']
for ch in range(3):
    ax.plot(t2, EMG2[:, ch] + ch * offset2, label=channel_labels[ch])
ax.set_title('Task 1.2: iEMG - 3 Channels')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (mV)')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

# =============================================================================
# Task 1.3 - Filtering (Notch + Bandpass)
# =============================================================================

# Notch filter at 50 Hz
wo = 50 / (fsamp1 / 2)
bw = wo / 35
b_notch, a_notch = iirnotch(wo, Q=35)  # Q = 1/bw equivalent

# Bandpass filter 20–500 Hz, 4th-order Butterworth
b_band, a_band = butter(4, [20 / (fsamp1 / 2), 500 / (fsamp1 / 2)], btype='bandpass')

# Apply filters to HD-sEMG
EMG1_filt = filtfilt(b_notch, a_notch, EMG1_mV, axis=0)
EMG1_filt = filtfilt(b_band, a_band, EMG1_filt, axis=0)

# Apply filters to iEMG
EMG2_filt = filtfilt(b_notch, a_notch, EMG2, axis=0)
EMG2_filt = filtfilt(b_band, a_band, EMG2_filt, axis=0)

# =============================================================================
# Task 1.4 - FFT Before and After Filtering
# =============================================================================

mean_raw1 = EMG1_mV.mean(axis=1)
mean_filt1 = EMG1_filt.mean(axis=1)
mean_raw2 = EMG2.mean(axis=1)
mean_filt2 = EMG2_filt.mean(axis=1)

N1 = len(mean_raw1)
f1 = np.arange(N1) * (fsamp1 / N1)

N2 = len(mean_raw2)
f2 = np.arange(N2) * (fsamp2 / N2)

fft_raw1 = np.abs(fft(mean_raw1))
fft_filt1 = np.abs(fft(mean_filt1))
fft_raw2 = np.abs(fft(mean_raw2))
fft_filt2 = np.abs(fft(mean_filt2))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(f1, fft_raw1, 'r', label='Raw')
ax.plot(f1, fft_filt1, 'b', label='Filtered')
ax.set_xlim([0, 1000])
ax.set_title('Task 1.4: FFT HD-sEMG')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude')
ax.legend()

ax = axes[1]
ax.plot(f2, fft_raw2, 'r', label='Raw')
ax.plot(f2, fft_filt2, 'b', label='Filtered')
ax.set_xlim([0, 1000])
ax.set_title('Task 1.4: FFT iEMG')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude')
ax.legend()

plt.tight_layout()
plt.show()

# =============================================================================
# Task 1.5 - Overlay Raw vs Filtered
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# HD-sEMG: first 10 channels
ax = axes[0]
offset1 = 3.0
for ch in range(10):
    ax.plot(t1, EMG1_mV[:, ch] + ch * offset1, 'b--', linewidth=0.8)
    ax.plot(t1, EMG1_filt[:, ch] + ch * offset1, 'r', linewidth=0.8)
ax.set_title('Task 1.5: HD-sEMG - Raw (Blue) vs Filtered (Red)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude + Offset')
ax.set_ylim([-0.5, 11 * offset1])
ax.set_xlim([t1[0], t1[-1]])
ax.legend(['Raw', 'Filtered'])
ax.grid(True)

# iEMG: all 3 channels
ax = axes[1]
offset2 = 0.15
for ch in range(3):
    ax.plot(t2, EMG2[:, ch] + ch * offset2, 'b--', linewidth=0.8)
    ax.plot(t2, EMG2_filt[:, ch] + ch * offset2, 'r', linewidth=0.8)
ax.set_title('Task 1.5: iEMG - Raw (Blue) vs Filtered (Red)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude + Offset')
ax.set_ylim([-0.1, 4 * offset2])
ax.set_xlim([t2[0], t2[-1]])
ax.legend(['Raw', 'Filtered'])
ax.grid(True)

plt.tight_layout()
plt.show()

# =============================================================================
# Task 1.6.1 - Average Every 8 Filtered HD-sEMG Channels (→ 8 groups)
# =============================================================================

avg_HD = np.zeros((EMG1_filt.shape[0], 8))
for i in range(8):
    ch_range = slice(i * 8, (i + 1) * 8)
    avg_HD[:, i] = EMG1_filt[:, ch_range].mean(axis=1)


# =============================================================================
# Task 1.6.2 - RMS with Different Window Sizes
# =============================================================================

def moving_rms(signal, window):
    """Compute moving RMS using a uniform sliding window."""
    kernel = np.ones(window) / window
    # Apply along axis=0 (samples) for each channel
    if signal.ndim == 1:
        return np.sqrt(np.convolve(signal ** 2, kernel, mode='same'))
    return np.sqrt(np.array(
        [np.convolve(signal[:, c] ** 2, kernel, mode='same')
         for c in range(signal.shape[1])]).T)


win_ms_list = [50, 100, 200, 500]
offset_HD = 0.5
offset_iEMG = 0.05

fig, axes = plt.subplots(4, 2, figsize=(16, 22))

for i, win_ms in enumerate(win_ms_list):
    win_HD = round(fsamp1 * win_ms / 1000)
    win_iEMG = round(fsamp2 * win_ms / 1000)

    RMS_HD = moving_rms(avg_HD, win_HD)
    RMS_iEMG = moving_rms(EMG2_filt, win_iEMG)

    # HD-sEMG RMS
    ax = axes[i, 0]
    for ch in range(8):
        ax.plot(t1, RMS_HD[:, ch] + ch * offset_HD)
    ax.set_title(f'RMS HD-sEMG – Window = {win_ms} ms')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RMS (mV)')
    ax.grid(True)

    # iEMG RMS
    ax = axes[i, 1]
    for ch in range(3):
        ax.plot(t2, RMS_iEMG[:, ch] + ch * offset_iEMG)
    ax.set_title(f'RMS iEMG – Window = {win_ms} ms')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RMS (mV)')
    ax.grid(True)

plt.tight_layout(pad=3.0, h_pad=4.0)
plt.show()

# =============================================================================
# Task 2.1 - RMS (200 ms) and Force on Dual Y-Axis
# =============================================================================

win_len_200 = round(fsamp1 * 0.2)
RMS_HD_200 = moving_rms(avg_HD, win_len_200)
t = np.arange(RMS_HD_200.shape[0]) / fsamp1
offset = 0.7

fig, ax1 = plt.subplots(figsize=(14, 6))

color_blue = (0, 0.447, 0.741)
for ch in range(4):
    ax1.plot(t, RMS_HD_200[:, ch] + ch * offset,
             color=color_blue, linewidth=1.2)
for ch in range(4, 7):
    ax1.plot(t, RMS_HD_200[:, ch] + ch * offset,
             color=color_blue, linewidth=0.5)
ax1.plot(t, RMS_HD_200[:, 7] + 7 * offset,
         color=color_blue, linewidth=1.2)

ax1.set_ylabel('RMS Amplitude + Offset (mV)')
ax1.set_ylim([-0.2, 9 * offset])
ax1.set_yticks([i * offset for i in range(8)])
ax1.set_yticklabels([f'Ch {i + 1}' for i in range(8)])

ax2 = ax1.twinx()
ax2.plot(t, force, 'k', linewidth=1.5, label='Force')
ax2.set_ylabel('Force (N or AU)')
ax2.set_ylim([force.min(), force.max() * 1.1])

ax1.set_xlabel('Time (s)')
ax1.set_title('Task 2.1: RMS of Averaged HD-sEMG (200ms) and Measured Force')
ax1.grid(True)

lines = [plt.Line2D([0], [0], color=color_blue, lw=1.2)] + \
        [plt.Line2D([0], [0], color='k', lw=1.5)]
labels = [f'Ch{i + 1}' for i in range(8)] + ['Force']
ax1.legend(lines, labels, loc='upper right', fontsize=7)

plt.tight_layout()
plt.show()

# =============================================================================
# Task 2.2 - Correlation Between RMS and Force
# =============================================================================

RMS_ch1 = moving_rms(avg_HD[:, 0], win_len_200)
RMS_ch1_norm = RMS_ch1 / RMS_ch1.max()
force_norm = (force - force.min()) / force.max()

# Pearson correlation
R = np.corrcoef(RMS_ch1_norm, force_norm)[0, 1]

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(RMS_ch1_norm, force_norm, s=10)
ax.set_xlabel('Normalized RMS (Channel 1)')
ax.set_ylabel('Normalized Force')
ax.set_title(f'Task 2.2: Correlation between RMS and Force – R = {R:.3f}')
ax.grid(True)
plt.tight_layout()
plt.show()

# =============================================================================
# Task 2.3 - Force Prediction Using Linear Regression
# =============================================================================

win_ms = 200
win_len = round(fsamp1 * win_ms / 1000)

RMS_HD = moving_rms(avg_HD, win_len)  # shape: (samples, 8)

min_len = min(len(force), RMS_HD.shape[0])
RMS_HD = RMS_HD[:min_len, :]
force_trimmed = force[:min_len]

# Fit multiple linear regression
reg = LinearRegression()
reg.fit(RMS_HD, force_trimmed)
predicted_force = reg.predict(RMS_HD)

RMSE = np.sqrt(mean_squared_error(force_trimmed, predicted_force))

t_pred = np.arange(min_len) / fsamp1

fig, ax1 = plt.subplots(figsize=(14, 5))

ax1.plot(t_pred, predicted_force, 'r', linewidth=1.2, label='Predicted Force')
ax1.set_ylabel('Predicted Force')

ax2 = ax1.twinx()
ax2.plot(t_pred, force_trimmed, 'b', linewidth=1.2, label='Measured Force')
ax2.set_ylabel('Measured Force')

ax1.set_xlabel('Time (s)')
ax1.set_title(f'Task 2.3: Force Prediction (RMS {win_ms}ms) – RMSE = {RMSE:.3f}')

lines = [plt.Line2D([0], [0], color='r', lw=1.2),
         plt.Line2D([0], [0], color='b', lw=1.2)]
ax1.legend(lines, ['Predicted Force', 'Measured Force'])
ax1.grid(True)

plt.tight_layout()
plt.show()
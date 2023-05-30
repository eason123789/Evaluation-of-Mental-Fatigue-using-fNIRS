import mne
import numpy as np
from mne.preprocessing import ICA

# 假设 raw 是你的 EEG 数据对象
raw = mne.io.read_raw_eeglab('rawdata/s01_060227n.set', preload=True)

# 应用 1 Hz 到 50 Hz 的带通滤波器
raw.filter(1., 50., fir_design='firwin')


# 用 ICA 来修正眼部和肌肉伪迹
# 首先，你需要选择要用于 ICA 的通道
ica_channels = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                               stim=False, exclude='bads')

# 然后，你可以设置 ICA 的参数并应用它
ica = ICA(n_components=15, random_state=97)
ica.fit(raw, picks=ica_channels)

# 你可以使用 EOG 通道作为参考来自动检测伪迹组件
eog_indices, eog_scores = ica.find_bads_eog(raw)

# 然后你可以删除这些组件
ica.exclude.extend(eog_indices)

# 最后，应用 ICA，得到修正后的数据
raw_corrected = ica.apply(raw.copy())

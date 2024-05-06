import librosa
import numpy as np
import pywt
from pydub import AudioSegment

def analyze_audio_file(file_path, frame_duration):
    # Загрузка аудиофайла
    audio = AudioSegment.from_file(file_path)
    signal = np.array(audio.get_array_of_samples())
    sr = audio.frame_rate
    num_channels = audio.channels

    # Разбиение аудиосигнала на кадры
    frame_length = int(sr * frame_duration)
    hop_length = frame_length
    frames = [librosa.util.frame(channel, frame_length=frame_length, hop_length=hop_length).T for channel in signal.reshape(-1, num_channels).T]

    # Вычисление количества полутонов
    num_semitones = 120

    # Создание массивов для результатов анализа
    channel_angles = np.zeros((num_channels, 1, num_semitones))
    channel_volumes = np.zeros((num_channels, 1, num_semitones))

    # Применение вейвлет-преобразования Хаара к каждому кадру для каждого канала
    for channel_idx, channel_frames in enumerate(frames):
        for i, frame in enumerate(channel_frames):
            cA, cD = pywt.dwt(frame, 'haar')
            cA_mean = np.mean(np.abs(cA))

            # Расчет средней громкости для каждой частоты с шагом в 1 полутон
            for j in range(num_semitones):
                start_freq = 20 * 2**(j/12)
                end_freq = 20 * 2**((j+1)/12)
                freq_indices = librosa.core.fft_frequencies(sr=sr, n_fft=frame_length)
                indices = np.where((freq_indices >= start_freq) & (freq_indices < end_freq))[0]
                if len(indices) > 0:
                    channel_volumes[channel_idx, 0, j] = np.mean(np.abs(cD[indices]))

            # Расчет углового соотношения звучания
            if num_channels == 2:
                left_volume = channel_volumes[0, 0, :]
                right_volume = channel_volumes[1, 0, :]
                channel_angles[0, 0, :] = -180 * (left_volume - right_volume) / (left_volume + right_volume)
                channel_angles[1, 0, :] = 180 * (left_volume - right_volume) / (left_volume + right_volume)

    return channel_angles, channel_volumes, num_semitones

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def plot_audio_analysis(channel_angles, channel_volumes, num_semitones):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(10, 7))

    vertices = [
        [(-180, 20, 0), (180, 20, 0)], 
        [(-180, 20, 140), (180, 20, 140)], 
        [(-180, 20, 0), (-180, 20, 140)], 
        [(180, 20, 140), (180, 20, 0)],
        [(0, 20000, 0), (0, 20000, 140)],
        [(-180, 20, 140), (0, 20000, 140)],
        [(180, 20, 140), (0, 20000, 140)],
        [(-180, 20, 0), (0, 20000, 0)],
        [(180, 20, 0), (0, 20000, 0)]
    ]

    ax.add_collection3d(Poly3DCollection(vertices, facecolors='white', linewidths=1, edgecolors='r', alpha=.25))

    for v in vertices:
        for point in v:
            ax.text(*point, f'({point[0]}, {point[1]}, {point[2]})')

    # Отображение данных анализа аудио
    x = np.linspace(-180, 180, num_semitones)
    y = np.linspace(20, 20000, num_semitones)
    X, Y = np.meshgrid(x, y)

    for channel_idx in range(channel_angles.shape[0]):
        ax.plot_surface(X, Y, channel_volumes[channel_idx, 0, :], rstride=1, cstride=1, cmap='viridis')

    ax.set(xlabel='Ширина (градусы)', ylabel='Глубина (Hz)', zlabel='Высота (dB)',
           xlim=(-180, 180), ylim=(20, 20000), zlim=(0, 140))

    plt.show()

# Пример использования функции для отображения результатов анализа
channel_angles, channel_volumes, num_semitones = analyze_audio_file("Liquid_Code.wav", 1/20)
plot_audio_analysis(channel_angles, channel_volumes, num_semitones)

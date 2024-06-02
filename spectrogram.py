import numpy as np
import matplotlib.pyplot as plt
# plotly offline
import plotly.offline as pyo
from plotly.offline import init_notebook_mode #to plot in jupyter notebook
import plotly.graph_objs as go
# init_notebook_mode() # init plotly in jupyter notebook

from scipy.io import wavfile # scipy library to read wav files

src_name = "IMG_2255"

import subprocess
# -ab 160k -ac 2 -ar 44100 -vn
command = f"ffmpeg -i data/{src_name}.MOV data/{src_name}.wav -y"
subprocess.call(command, shell=True)

fs, Audiodata = wavfile.read(f"data/{src_name}.wav")
print(f"{fs = }")
Audiodata = Audiodata[:, 0] / (2.**15) # Normalized between [-1,1]
print(f"{Audiodata = }")
print(f"{Audiodata.shape = }")

#Spectrogram
from scipy import signal
plt.figure()
N = 16384 #Number of point in the fft
w = signal.blackman(N)
# print(f"{w = }")
print(f"{w.shape = }")
freqs, bins, Pxx = signal.spectrogram(Audiodata, fs, window = w, nfft=N)

data = 10*np.log10(Pxx)
data[data < -40] = -40
print(data.shape)

# Plot with plotly
trace = [go.Heatmap(
    x= bins,
    y= freqs,
    z= data,
    colorscale='Inferno_r',
    )]
layout = go.Layout(
    title = 'Spectrogram with plotly',
    yaxis = dict(title = 'Frequency'), # x-axis label
    xaxis = dict(title = 'Time'), # y-axis label
    )
fig = go.Figure(data=trace, layout=layout)
fig.update_yaxes(type="log")

pyo.iplot(fig, filename='Spectrogram')

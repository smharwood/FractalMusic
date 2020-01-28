"""
27 November 2019
SM Harwood
    
Analyze sound to help come up with a random seed for Chaos Game
"""
import sys
import numpy as np
import sounddevice as sd
from scipy.fftpack import fft, fftshift
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


def get_peak(target=440, within=10, duration_seconds=2, rate=6000, test=False):
    """ 
    Record audio and analyze to find peak frequency in some region 

    Args:
    target : (float) The frequency in Hertz to look around
    within : (float) The window size around target to search 
    duration_seconds : (float) Duration of recording (in seconds)
    rate : (int) Sample rate in Hertz (samples/second)
    test : (Boolean) Whether to show some plots of whats going on

    Returns:
    peak : (float) Identified peak frequency in Hz in the specified window around target
    data : (array) Raw recorded data to fiddle with if desired
    """

    # Grab some audio as numpy array
    # Pad desired duration by a couple seconds 
    # (b/c there are weird artifacts at beginning) 
    pad_duration = duration_seconds + 0.6
    print("Recording sound...")
    data = sd.rec(int(pad_duration * rate), samplerate=rate, channels=1)
    sd.wait()  
    # flatten data
    data = data.ravel()[-int(duration_seconds*rate):]
    if test:
        print("Sound data shape: {}".format(data.shape))
        plt.plot(np.arange(len(data))/rate,data)
        plt.title('Waveform')
        plt.xlabel('Time, seconds')
        plt.show()
        # save
        #wavfile.write('waveform.wav', rate, data)
        # Spectrogram
        f, t, Sxx = signal.spectrogram(data,rate)
        Sxx_mod = np.log(Sxx)
        plt.pcolormesh(t, f, Sxx_mod)
        plt.show()
        print("Spectrogram resolution: {} Hz".format(max(f)/len(f)))

    # Discrete Fourier Transform
    # Take second half because 
    # those are the negative frequencies which don't matter for real signal
    fft_out = fft(data)
    fft_mod = np.log(np.abs(fft_out[0:len(fft_out)//2]))
    
    # Find maximum frequency of spectrum
    # Focus in on a 10 Hz window around some target
    # Note: resolution (how accurately we can determine/resolve frequencies)
    # is essentially the inverse of how long we record
    resolution = rate/len(data)
    index_windowL = int((target-within)/resolution)
    index_windowU = int((target+within)/resolution)
#    print(resolution)
#    print(len(myrecording))
#    print(index_windowL)
#    print(index_windowU)
    peak_index = index_windowL + np.argmax(fft_mod[index_windowL:index_windowU])
    peak = peak_index*resolution
    if test:
        print("Peak: {} (Hz)".format(peak))
        x = resolution*np.arange(len(fft_mod))
        plt.plot(x,fft_mod)
        plt.axvline(x=peak,color='k')
        plt.xlabel('Frequency, Hz')
        plt.title('Spectrum')
        plt.show()
    return peak, data


def dummy():
    """ If you are a dummy and your data is not shaped properly 
        this is what happens 
    """
    # artificial data: sin wave at 1 Hz
    duration = 3
    domain = np.arange(0,duration,1/100)
    data = np.sin(domain*2*np.pi)
    fft_out = fft(data)
    fft_mod = np.abs(fft_out[0:len(fft_out)//2])
    # plot
    plt.plot(domain,data)
    plt.title('Time series')
    plt.show()
    w = np.arange(len(fft_mod))/duration
    plt.plot(w,fft_mod)
    plt.title('Spectrum')
    plt.show()
    # improperly shaped data
    dataT = data.reshape(-1,1)
    print(dataT.shape)
    fft_modT = np.abs(fft(dataT))
    plt.plot(fft_modT)
    plt.title('WRONG spectrum')
    plt.show()
    

if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        duration = int(args[0])
        get_peak(duration_seconds=duration,test=True)
    else:
        get_peak(test=True)

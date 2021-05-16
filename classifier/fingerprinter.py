"""
27 April 2021
SM Harwood

More shazam-inspired music fingerprinter/identifier
see http://coding-geek.com/how-shazam-works/
"""
import argparse, os, re
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft
from scipy.signal import get_window
from classifier import _ALL_CLASSES, get_audio

def get_features(
        data,
        rate,
        target_length,
        fft_length,
        fps=24,
        smoothing=0.8
    ):
    """ Analyze audio as similarly as the JS code does,
    This will kind of copy classifier.get_augmented_features

    Parameters:
    data: Array, the audio data
    rate: Integer, the sampling rate of the audio data
    target_length: Length of analysis in seconds;
        along with fps this determines the length of fingerprint
    fps: Intger, target/average frames per second (each frame does a FFT analysis)
        this gives number of samples between frames: rate//fps
    fft_length: Integer, number of samples for the windowed FFT
    smoothing: Float \in [0,1], Weight for previous values in a smoothing operation
    """
    # Baseline number of frames, data points/steps between frames
    d_dtype = data.dtype
    assert d_dtype.name == 'int16', "Expecting a different data type"
    num_frames = int(target_length*fps)
    f_step = rate//fps
    steps = range(0, len(data), f_step)

    # Spectrographic data gets a fixed length - 
    # do analysis until we reach desired length
    window = get_window('blackman', fft_length)
    # Follow analysernode documentation:
    # https://webaudio.github.io/web-audio-api/#current-time-domain-data
    # Scan through time domain data and do windowed FFT
    # Apply Blackman window when doing FFT
    # Smooth with previous time step (and take abs. value)
    # Convert frequency domain to decibels
    spec = np.zeros((fft_length//2, num_frames))
    smoothed = np.zeros(fft_length//2)
    for j,s in enumerate(steps):
        if j >= num_frames: break
        freq = rfft(data[s:s+fft_length]*window)
        smoothed = smoothing*smoothed + \
                (1 - smoothing)*np.abs(freq[:-1])
        spec[:, j] = np.log10(smoothed)
    # As in the classifier, these features/fingerprint must be used in a scale-
    # and shift- invariant way (and indeed, the max frequencies in various bands
    # will be used)
    # So, we skip any final scaling steps and return features as is
    return spec

def get_fingerprints(
        path,
        target_length,
        fft_length,
        plotit=False
    ):
    """ Make the fingerprints for all songs  """

    # Goal of fingerprint is to pick out top top frequencies in various bins:
    #   the very low sound band (0 - 107 Hz)
    #   the low sound band      (107 - 214 Hz)
    #   the low-mid sound band  (214 - 428 Hz)
    #   the mid sound band      (428 - 856 Hz)
    #   the mid-high sound band (856 - 1712 Hz)
    #   the high sound band     (1712 - 5468 Hz)
    # Bin width = sample rate / fft window length
    # Think: bin width * length of spectrogram = half of sampling rate = Nyquist
    target_rate = 44100
    bin_width = target_rate / fft_length
    freq_limits = [0, 107, 214, 428, 856, 1712, 5468]
    band_indices = [int(np.round(f / bin_width)) for f in freq_limits]
    print("Freq Bin width: {}".format(bin_width))
    print("Band indices: {}".format(band_indices))
    # Band indices: [0, 10, 20, 40, 80, 159, 508] for fft_length=4096

    classes = _ALL_CLASSES
    pattern = '({}).*(\.mp3)'.format('|'.join(classes))
    reg = re.compile(pattern, re.IGNORECASE)

    labels = []
    fingerprints = []
    print("Making fingerprints for classes ", classes)
    # Find the mp3 files (filename contains the class name)
    # For each, match to the class, extract the features
    for f in os.listdir(path):
        m = reg.search(f)
        if not m: continue
        c = m.groups()[0].lower()
        print("{} = {}".format(f,c))
        data, rate = get_audio(os.path.join(path,f), 2*target_length)
        spec = get_features(data, rate, target_length, fft_length)
        assert rate == target_rate, "Sample rates do not agree"
        # Fingerprints are max frequency indices in various bands
        max_freq_indices = np.zeros((6, spec.shape[1]))
        for j in range(spec.shape[1]):
            max_freq_indices[:,j] = [s + np.argmax(spec[s:t,j])
                                  for s,t in zip(band_indices,band_indices[1:])]
        labels.append(c)
        fingerprints.append(max_freq_indices)
        if plotit:
            plt.plot(max_freq_indices.transpose())
            plt.savefig("{}_fp.png".format(c))
            plt.close()
    return fingerprints, labels

def export_fingerprints(
        path,
        target_length=30,
        fft_length=4096
    ):
    """ Get and export fingerprints """
    fingerprints, labels = get_fingerprints(path, target_length, fft_length)
    # Sort based on labels to make it easier on the javascript
    # sorted() will do so lexicographically on pairs
    zipped_lists = zip(labels, fingerprints)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    labels, fingerprints = [list(tup) for tup in tuples]
    flattened_fp = np.zeros((len(fingerprints), fingerprints[0].size))
    for i,fp in enumerate(fingerprints):
        flattened_fp[i,:] = np.ravel(fp, order='F')
    np.savetxt('fingerprints_{}.csv'.format(fft_length), flattened_fp,
        delimiter=',', fmt='%d')
    np.savetxt('fp_classes.txt', np.array(labels), delimiter='\n', fmt='%s')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to fingerprint songs")
    parser.add_argument('-f', '--fingerprint', action='store_true',
        help="Fingerprint, using audio in PATH")
    parser.add_argument('-p', '--path', type=str, default='audio',
        help="Path to folder of audio clips")
    parser.add_argument('-w', '--fft_window', type=int,
        help="Length of FFT window")
    args = parser.parse_args()

    noaction = True
    if args.fingerprint:
        export_fingerprints(args.path)
        noaction = False
    if noaction:
        parser.print_help()

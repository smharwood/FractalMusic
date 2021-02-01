"""
21 Jan 2020
SM Harwood

Train a classifier of audio data
Sort of following
https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
"""
import os, sys, re, subprocess, argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft
#from scipy.fftpack import fft
from scipy.io import wavfile as wav
from scipy.signal import resample, spectrogram, get_window
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# TODO consider a more convolutional approach;
# maybe the classifier works on short chunks and you get a series of predictions
# over time and then take the most frequent

_ALL_CLASSES = [
        'broccoli', 'canyon', 'daisy', 'dna', 'feathers', 'florida', 'leaves',
        'lightening', 'nautilus', 'pineapple', 'snowflake', 'tree', 'turtle'
    ]

def get_audio(
        filename,
        target_length,
        target_rate=44100):
    """ Get audio data and possibly resample it """
    # Get sound data as wav
    name = filename.split('.')[0]
    try:
        rate, data = wav.read(name+'.wav')
    except Exception:
        subprocess.call("ffmpeg -i {} {}.wav".format(filename, name).split())
        rate, data = wav.read(name+'.wav')
    # Take average of the channels, (ravel/flatten for good measure)
    data = np.mean(data, axis=1).ravel()
    print("Audio: {}".format(filename))
    #print("Data shape: {}".format(data.shape))
    #print("Rate: {}".format(rate))

    if rate != target_rate:
        # Resample to get the same target sample rate
        # First truncate (with some padding to avoid artifacts?)
        data = data[:2*target_length*rate]
        target_num = 2*target_length*target_rate
        data = resample(data, target_num)
    # Clip to target length
    data = data[:target_length*target_rate]
    return data, target_rate

def get_augmented_features(
        data,
        rate,
        fft_length=2048,
        fps=20,
        smoothing=0.8,
        num_augments=0,
        min_db=-100,
        max_db=-30
    ):
    """ Analyze audio as similarly as the JS code does,
    so that we can train a classifier that can use that data directly.
    However, tweak and augment base dataset for a more robust classifier 

    Parameters:
    data: Array, the audio data
    rate: Integer, the sampling rate of the audio data
    fft_length: Integer, number of samples for the windowed FFT
    fps: Intger, target/average frames per second (each frame does a FFT analysis)
    smoothing: Float \in [0,1], Weight for previous values in a smoothing operation
    num_augments: Integer, number of augmented samples to create
    """
    # Shift start, add noise, randomize FFT analysis step length
    num_steps = int(len(data)/rate*fps)
    augmenteds = [data]
    fft_steps = [rate//fps]
    scale = 0.05*0.5*(np.max(data) - np.min(data)) # about 5% noise?
    for i in range(num_augments):
        start = np.random.exponential(1.5)
        rand_scale = scale*np.random.rand()
        noise = np.random.normal(0, rand_scale, len(data))
        offset = int(start*rate)
        noise[offset:] += data[:-offset]
        augmenteds.append(noise)
        fft_steps.append(rate//(fps + np.random.randint(-5,5)))

    # Spectrographic data gets a fixed length - 
    # randomized fft_step means that it either gets clipped or padded with zeros
    #num_steps = len(range(0, len(data) - fft_length, fft_step))
    window = get_window('blackman', fft_length)
    specs = []
    for dat, f_step in zip(augmenteds, fft_steps):
        # Follow analysernode documentation:
        # https://webaudio.github.io/web-audio-api/#current-time-domain-data
        # Scan through time domain data and do windowed FFT
        # Apply Blackman window when doing FFT
        # Note that the frequency domain gets scaled
        # Convert frequency domain to decibels
        # Smooth with previous time step
        # Normalize to [0,1]
        # This last step is not part of standard, but its a simple way to make
        # this consistent with what happens in javascript.
        # Also, we can skip some things (scaling by 1/N, scaling by 20 in dB)
        steps = range(0, len(dat) - fft_length, f_step)
        spec = np.zeros((fft_length//2, num_steps))
        smoothed = np.zeros(fft_length//2)
        for j,s in enumerate(steps):
            if j >= num_steps: break
            freq = rfft(dat[s:s+fft_length]*window)
            smoothed = smoothing*smoothed + \
                    (1 - smoothing)*np.abs(freq[:-1])
            spec[:, j] = np.log10(smoothed)
        spec -= np.min(spec)
        spec /= np.max(spec)
        # Fortran (column-major) index order:
        # will concatenate the windowed FFTs together
        specs.append(np.ravel(spec, order='F'))
    return specs, augmenteds

def get_dataset(path, num_augments, target_length):
    """ Make the dataset for a classifier  """
    classes = _ALL_CLASSES
    pattern = '({}).*(\.mp3)'.format('|'.join(classes))
    reg = re.compile(pattern, re.IGNORECASE)

    print("Making dataset for classes ", classes)

    # Construct dataset as dictionary
    dataset = dict()
    dataset['description'] = \
        "music clip spectrograms labeled as their natural fractal inspiration"
    dataset['labels'] = []
    dataset['data'] = []
    dataset['filenames'] = []
    # Find the mp3 files (filename contains the class name)
    # For each, match to the class, extract the features, and augment
    for f in os.listdir(path):
        m = reg.search(f)
        if not m: continue
        c = m.groups()[0].lower()
        print("{} = {}".format(f,c))
        data, rate = get_audio(os.path.join(path,f), target_length)
        specs, _ = get_augmented_features(data, rate,
            num_augments=num_augments)
        dataset['labels'] += [c]*len(specs)
        dataset['data'] += specs
        dataset['filenames'] += [f]*len(specs)
    return dataset

def rename_audio(path='audio'):
    """ Rename audio files to '<class>N.mp3' """
    # Assuming the audio files have the class name in the file name
    # A bit of a hack so that my shell call to ffmpeg can convert them to WAV
    classes = _ALL_CLASSES
    counts = dict()
    for c in classes:
        counts[c] = 0
    pattern = '({}).*(\.mp3)'.format('|'.join(classes))
    reg = re.compile(pattern, re.IGNORECASE)
    for f in os.listdir(path):
        m = reg.search(f)
        if not m: continue
        c = m.groups()[0].lower()
        counts[c] += 1
        os.rename(os.path.join(path, f), 
                  os.path.join(path, '{}{}.mp3'.format(c,counts[c])) )
    return

def train_clf(path='audio', num_augments=60, target_length=10):
    """ Train linear classifier """
    dataset = get_dataset(path, num_augments, target_length)
    joblib.dump(dataset, "dataset.pkl")

    X = np.array(dataset['data'])
    y = np.array(dataset['labels'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        test_size=0.2, 
        shuffle=True,
        random_state=42)
    # Scale dataset - this is a step we will have to implement in JS
    scaler = StandardScaler()
    X_train_prep = scaler.fit_transform(X_train)

    print('Training...')
    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, verbose=1, random_state=42)
    sgd_clf.fit(X_train_prep, y_train)
    joblib.dump(sgd_clf, "trained_sgd.pkl")
    joblib.dump(scaler, "trained_scaler.pkl")

    # Test
    test_clf(sgd_clf, scaler, X_test, y_test)
    return

def test_clf(clf=None, scaler=None, X_test=None, y_test=None):
    print('Testing')
    # Re-load anything from pickle
    if clf is None:
        clf = joblib.load("trained_sgd.pkl")
    if scaler is None:
        scaler = joblib.load("trained_scaler.pkl")
    if X_test is None or y_test is None:
        dataset = joblib.load("dataset.pkl")
        X = np.array(dataset['data'])
        y = np.array(dataset['labels'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
            test_size=0.2, 
            shuffle=True,
            random_state=42)

    # Scale dataset
    X_test_prep = scaler.fit_transform(X_test)
    # Predict
    y_pred = clf.predict(X_test_prep)
    prcnt_correct = 100*np.sum(y_pred == y_test)/len(y_test)
    prcnt_random = 100*(1/len(clf.classes_))**len(y_test)
    print('Correct: {:5.2f}%'.format(prcnt_correct))
    print('(Random guessing: {:7.1e}%)'.format(prcnt_random))

    # Manual prediction:
    test_manual(clf, scaler, X_test, y_test, y_pred)
    return

def test_manual(clf, scaler, X_test, y_test, y_pred):
    """ Implement the predictions manually;
    export weights and copy what will happen in JS
    """
    # Manual predictions:
    # Rip out the raw weights and intercepts, and compute class scores
    # The multiclass classification is one-vs-all; 
    # assign class that gets highest score
    # Classifier was trained on scaled and shifted data,
    # so overall, for "raw" data x
    # scores = A*(x - mu)/s + b
    # Equivalently:
    # scores = (A/s)*x + (b - (A/s)*mu)
    # i^th column of (A/s) = i^th column of A scaled by s_i
    # A.shape = (n_classes, n_features), s.shape = (n_features,)
    # so A/s broadcasts over the rows
    A = clf.coef_
    b = clf.intercept_
    mu = scaler.mean_
    scale = scaler.scale_
    A_mod = A/scale
    b_mod = b - A_mod.dot(mu)
    # Export weights; play with precision, what about JSON?
    np.savetxt('weights.csv', A_mod, delimiter=',', fmt='%.3f')
    np.savetxt('intercepts.csv', b_mod, delimiter=',')
    np.savetxt('classes.txt', clf.classes_, delimiter='\n', fmt='%s')
    A_mod = np.genfromtxt('weights.csv', delimiter=',')
    # scores.shape = (n_classes, n_examples)
    scores = A_mod.dot( X_test.transpose() ) + b_mod.reshape((-1,1))
    y_index = np.argmax(scores, axis=0)
    y_pred_manual = [clf.classes_[yi] for yi in y_index]
    prcnt_correct = 100*np.sum(y_pred_manual == y_test)/len(y_test)
    prcnt_agree   = 100*np.sum(y_pred_manual == y_pred)/len(y_pred)
    print('Manual, correct: {:5.2f}%'.format(prcnt_correct))
    print('Manual, agreement: {:5.2f}%'.format(prcnt_agree))
    return

def test_features(fname='test.mp3'):
    """ Verify things sort of make sense """
    fft_length=2048
    data, rate = get_audio(
        fname,
        target_length=10,
        target_rate=44100)
    specs, augmenteds = get_augmented_features(
        data,
        rate, 
        fft_length=fft_length,
        num_augments=1)

    bname = os.path.basename(fname).split('.')[0]
    index = 0
    spec = specs[index].reshape((fft_length//2, -1), order='F')
    data = augmenteds[index]
    print("Feature size: {}".format(spec.shape))
    plt.imshow(spec, origin='lower', aspect='auto')
    plt.savefig('jank_{}.png'.format(bname), bbox_inches='tight')
    plt.close()
    # Compare against a "real" spectrogram
    f, t, Sxx = spectrogram(data, rate, 
        window=get_window('blackman', fft_length),
        scaling='spectrum')
    Sxx_db = 20*np.log10(Sxx / fft_length)
    plt.pcolormesh(t, f, Sxx_db)
    plt.savefig('auto_{}.png'.format(bname), bbox_inches='tight')
    print("Min jank/auto value: {}/{}".format(np.min(spec), np.min(Sxx_db)))
    print("Max jank/auto value: {}/{}".format(np.max(spec), np.max(Sxx_db)))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to train and test music classifier")
    parser.add_argument('-t', '--train', action='store_true',
        help="Train classifier, using audio in PATH")
    parser.add_argument('-p', '--path', type=str, default='audio',
        help="Path to folder of audio clips")
    parser.add_argument('-s', '--test', action='store_true',
        help="Test trained and saved classifier")
    parser.add_argument('-v', '--vis', action='store_true',
        help="Visualize feature creation")
    parser.add_argument('-f', '--file', type=str, default='test.mp3',
        help="Audio file to visualize")
    parser.add_argument('-r', '--rename', action='store_true',
        help="Rename audio in PATH")
    args = parser.parse_args()

    noaction = True
    if args.train:
        train_clf(args.path)
        noaction = False
    if args.test:
        test_clf()
        noaction = False
    if args.vis:
        test_features(args.file)
        noaction = False
    if args.rename:
        rename_audio(args.path)
        noaction = False
    if noaction:
        parser.print_help()

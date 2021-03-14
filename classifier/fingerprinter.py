"""
21 Jan 2021
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
from global_scaler import GlobalMinMaxScaler

# 5 March 2021
# Current settings are the most defensible from a theory and practice viewpoint
# and seem to yield a reasonable classifier.
# Still not perfect

_ALL_CLASSES = [
        'broccoli', 'canyon', 'daisy', 'dna', 'feathers', 'florida', 'leaves',
        'lightening', 'nautilus', 'pineapple', 'snowflake', 'tree', 'turtle'
    ]

def get_audio(
        filename,
        target_length,
        target_rate=44100
    ):
    """ Get audio data and possibly resample it """
    # Get sound data as wav
    name = filename.split('.')[0]
    try:
        rate, data = wav.read(name+'.wav')
    except Exception:
        subprocess.call("ffmpeg -i {} -ar {} -ac 1 {}.wav".format(filename, target_rate, name).split())
        rate, data = wav.read(name+'.wav')
    # ffmpeg does a much better job converting to mono and resampling
    num_channels = len(data.shape)
    if rate != target_rate or num_channels != 1:
        assert False, "Delete your old wavfiles, please"
    print("Audio: {}".format(filename))
    print("Data shape: {}".format(data.shape))
    #print("Rate: {}".format(rate))

    # Trim out silence at beginning;
    # compare to some fraction of average volume
    # (looking to get rid of absolute silence at start of some clips)
    frac = 0.00
    avg = np.mean(np.fabs(data))
    first_i = np.argmax(np.fabs(data) > frac*avg)
    data = data[first_i:]
    print("Trimmed {} samples".format(first_i))

    # Clip to target length (or to end if target length too long)
    target_num = int(target_length*target_rate)
    data = data[:target_num]
    return data, target_rate

def get_augmented_features(
        data,
        rate,
        target_length,
        fps=24,
        fft_length=2048,
        smoothing=0.8,
        num_augments=0,
        max_drop=0.0
    ):
    """ Analyze audio as similarly as the JS code does,
    so that we can train a classifier that can use that data directly.
    However, tweak and augment base dataset for a more robust classifier 

    Parameters:
    data: Array, the audio data
    rate: Integer, the sampling rate of the audio data
    target_length: Length of analysis in seconds;
        along with fps this determines the size of the classifier
    fps: Intger, target/average frames per second (each frame does a FFT analysis)
        this gives number of samples between frames: rate//fps
    fft_length: Integer, number of samples for the windowed FFT
    smoothing: Float \in [0,1], Weight for previous values in a smoothing operation
    num_augments: Integer, number of augmented samples to create
    max_drop: Float \in [0,1], maximum frame drop rate
    """
    # Baseline number of frames, data points/steps between frames
    d_dtype = data.dtype
    assert d_dtype.name == 'int16', "Expecting a different data type"
    num_frames = int(target_length*fps)
    f_step = rate//fps
    steps = range(0, len(data), f_step)
    # Set up some reasonable noise
    # Signal-to-noise ratio (SNR) = 20 log( RMS(signal) / RMS(noise) )
    #snr = 10*np.log10(np.mean(data**2)/np.mean(noise**2))

    # Randomize:
    # noise, start sample, scaling, frame drop rate
    # Noise is sort of required as a regularizer (avoids taking log of zero)
    # Note that data is int16 valued and white noise should be approximated as a binomial
    aug_data = data + (np.random.binomial(2, 0.5, len(data)) - 1)
    augmenteds = [aug_data]
    vol_scales = [1.0]
    drop_rates = [0]
    for i in range(num_augments):
        # keep start offset fairly narrow?
        frame_offset = np.random.uniform(-1.5, 0)
        offset = int(frame_offset * f_step)
        if offset < 0:
            aug_dat = np.concatenate((data[-offset:], np.zeros(-offset, dtype=d_dtype)))
        elif offset > 0:
            aug_dat = np.concatenate((np.zeros(offset, dtype=d_dtype), data[:-offset]))
        else:
            aug_dat = np.copy(data)
        # Rescale to get desired SNR?
        n = np.random.randint(1,90)
        noise = np.random.binomial(2*(100*n), 0.5, len(data)) - (100*n)
        aug_dat += noise
        augmenteds.append(aug_dat)
        vol_scales.append(np.random.uniform(0.8, 1.2))
        if max_drop > 0:
            drop_rates.append(np.random.triangular(0, 0, max_drop))
        else:
            drop_rates.append(0)

    # Spectrographic data gets a fixed length - 
    # do analysis until we reach desired length
    window = get_window('blackman', fft_length)
    specs = []
    for dat, vs, drate in zip(augmenteds, vol_scales, drop_rates):
        # Follow analysernode documentation:
        # https://webaudio.github.io/web-audio-api/#current-time-domain-data
        # Scan through time domain data and do windowed FFT
        # Apply Blackman window when doing FFT
        # Smooth with previous time step (and take abs. value)
        # Convert frequency domain to decibels
        drops = np.random.binomial(1, drate, len(steps))
        spec = np.zeros((fft_length//2, num_frames))
        smoothed = np.zeros(fft_length//2)
        j = 0
        for s, d in zip(steps, drops):
            if j >= num_frames: break
            if d: continue
            freq = rfft(dat[s:s+fft_length]*window)
            smoothed = smoothing*smoothed + \
                    (1 - smoothing)*np.abs(freq[:-1])
            spec[:, j] = np.log10(smoothed)
            j += 1
        # Scaling must look at entire dataset, and "raw" dB features
        # do not map directly to what is required for prediction
        # (too hardware - i.e. microphone - dependent)
        # So, scaling and shifting will happen on entire dataset.
        # Thus, we can skip e.g. a factor of 20 in the conversion to dB above
        # Just randomly scale (in logspace this is just a random shift)
        spec += np.log10(vs)

        # Fortran (column-major) index order:
        # will concatenate the windowed FFTs together
        specs.append(np.ravel(spec, order='F'))
    return specs, augmenteds

def get_dataset(
        path,
        target_length,
        num_augments
    ):
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
    max_val = 0
    max_class = -1
    for f in os.listdir(path):
        m = reg.search(f)
        if not m: continue
        c = m.groups()[0].lower()
        print("{} = {}".format(f,c))
        data, rate = get_audio(os.path.join(path,f), 2*target_length)
        specs, _ = get_augmented_features(
            data, rate, target_length,
            num_augments=num_augments)
        dataset['labels'] += [c]*len(specs)
        dataset['data'] += specs
        dataset['filenames'] += [f]*len(specs)
        print("Class {}, # of examples: {}".format(c, len(specs)))
        if np.max(specs) > max_val:
            max_class = c
    print("Loudest class: {}".format(c))
    return dataset

def train_clf(
        path='audio',
        target_length=4.0,
        num_augments=100,
        dB_range=80,
        use_scaling=True
    ):
    """ Train linear classifier """
    dataset = get_dataset(
        path,
        target_length,
        num_augments)
    joblib.dump(dataset, "dataset.pkl")

    X = np.array(dataset['data'])
    y = np.array(dataset['labels'])
    # CLIP AND SCALE DATA - 
    # Final step of feature engineering is to scale everything to [0,1]
    # which are essentially the features we have in the Javascript code.
    # Record original dynamic range of the dataset
    # but then clip outliers beyond some dB range and shift and scale.
    # Recall that features are log10-valued, dB is 10 (or 20?) times that
    x_max_original = np.max(X)
    x_min_original = np.min(X)
    x_mean = np.mean(X)
    clip_max = x_mean + 0.5*dB_range/10
    clip_min = x_mean - 0.5*dB_range/10
    X = np.clip(X, clip_min, clip_max)
    X -= clip_min
    X /= (clip_max - clip_min)
    print("final dataset min/max: {}/{}".format(np.min(X), np.max(X)))

    # test/train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        test_size=0.2, 
        shuffle=True,
        random_state=42)
    # Scale dataset
    # Standard normal scaling does seem to help;
    # it must promote finding "robust" solutions during training
    if use_scaling:
        scaler = StandardScaler()
        X_train_prep = scaler.fit_transform(X_train)
    else:
        scaler = None
        X_train_prep = X_train

    print('Training...')
    sgd_clf = SGDClassifier(
        max_iter=1000, tol=1e-3, penalty='l2',
        verbose=1,
        random_state=42)
    sgd_clf.fit(X_train_prep, y_train)
    joblib.dump(sgd_clf, "trained_sgd.pkl")
    joblib.dump(scaler, "trained_scaler.pkl")

    # Test
    test_clf(sgd_clf, scaler, X_test, y_test)

    print()
    print("Original data, min/max: {}/{}".format(x_min_original, x_max_original))
    print("Clipped data, min/max: {}/{}".format(clip_min, clip_max))
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
    if scaler is not None:
        X_test_prep = scaler.fit_transform(X_test)
    else:
        # (pickled scaler was None)
        X_test_prep = X_test
    # Predict
    y_pred = clf.predict(X_test_prep)
    prcnt_correct = 100*np.sum(y_pred == y_test)/len(y_test)
    prcnt_random = 100*(1/len(clf.classes_))**len(y_test)
    print('Correct: {:5.2f}%'.format(prcnt_correct))
    print('(Random guessing: {:7.1e}%)'.format(prcnt_random))

    # Manual prediction
    test_manual(clf, scaler, X_test_prep, y_test, y_pred)
    return

def test_manual(clf, scaler, X_test, y_test, y_pred):
    """ Implement the predictions manually;
    export weights and copy what will happen in JS
    """
    # Manual predictions:
    # Rip out the raw weights and intercepts, and compute class scores
    # The multiclass classification is one-vs-all; 
    # assign class that gets highest score.
    # Data may be scaled; if x is "raw" data
    # scores = A(x - mu)/s + b
    # scores = (A/s)x + (b - (A/s)mu)
    # where column i of A/s equals column i of A divided by s_i
    # A.shape = (n_classes, n_features), s.shape = (n_features,)
    # so A/s broadcasts over the rows
    if scaler is not None:
        mu = scaler.mean_
        s = scaler.scale_
    else:
        mu = np.zeros(X_test.shape[1])
        s = np.ones(X_test.shape[1])
    A = clf.coef_ / s
    b = clf.intercept_ - A.dot(mu)
    #for bval in b: print(bval)
    # Export weights; play with precision, what about JSON?
    np.savetxt('weights.csv', A, delimiter=',', fmt='%.3f')
    np.savetxt('intercepts.csv', b, delimiter=',')
    np.savetxt('classes.txt', clf.classes_, delimiter='\n', fmt='%s')
    A_mod = np.genfromtxt('weights.csv', delimiter=',')
    # scores.shape = (n_classes, n_examples)
    scores = A_mod.dot( X_test.transpose() ) + b.reshape((-1,1))
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
    target_length=4
    data, rate = get_audio(
        fname,
        2*target_length,
        target_rate=44100)
    specs, augmenteds = get_augmented_features(
        data, rate, target_length,
        fft_length=fft_length,
        num_augments=1)

    bname = os.path.basename(fname).split('.')[0]
    index = 1
    spec = specs[index].reshape((fft_length//2, -1), order='F')
    data = augmenteds[index]
    wav.write('{}.wav'.format(bname), rate, data)
    print("Feature size: {}".format(spec.shape))
    plt.imshow(spec, origin='lower', aspect='auto')
    plt.savefig('jank_{}.png'.format(bname), bbox_inches='tight')
    plt.close()
    # Compare against a "real" spectrogram
    f, t, Sxx = spectrogram(data[:target_length*rate], rate, 
        window=get_window('blackman', fft_length),
        scaling='spectrum')
    Sxx_db = 20*np.log10(Sxx / fft_length)
    plt.pcolormesh(t, f, Sxx_db)
    plt.savefig('auto_{}.png'.format(bname), bbox_inches='tight')
    # What is decibel range of this?
    # spec is log10-valued
    print("Decibel range: {}".format(10*(np.max(spec)-np.min(spec))))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to train and test music classifier")
    parser.add_argument('-t', '--train', action='store_true',
        help="Train classifier, using audio in PATH")
    parser.add_argument('-s', '--test', action='store_true',
        help="Test trained and saved classifier")
    parser.add_argument('-v', '--vis', action='store_true',
        help="Visualize feature creation")
    parser.add_argument('-p', '--path', type=str, default='audio',
        help="Path to folder of audio clips")
    parser.add_argument('-f', '--file', type=str, default='test.mp3',
        help="Audio file to visualize")
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
    if noaction:
        parser.print_help()

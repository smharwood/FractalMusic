"""
5 March 2021
SM Harwood

Some utils to support classifier/fingerprinter
In particular, Frankenstein a good classifier out of pieces
"""
import argparse, os, re
import numpy as np
from fingerprinter import _ALL_CLASSES, get_audio

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

def determine_max_silence(path, max_length=60, min_length=1.0, silence_th=10.0):
    """ How long is silence in any of these initial clips? 

    Params:
    path (string): Directory name of clips to analyze
    max_length (number): Analyze first max_length seconds of the clips
    min_length (number): Discard any silences less than min_length seconds long
    silence_th (number): Threshhold for determining silence;
        Clips are loaded as WAV files with int16 values; max absolute value of
        the signal is ~32,000. So threshhold is some small integer.
    """
    """
    Representative results:
    Start: 37.86, Length: 1.25
    Start: 34.11, Length: 1.37
    Start: 31.15, Length: 1.59
    Start: 30.59, Length: 1.25
    ...
    Start: 14.80, Length: 3.41
    """
    # Find the mp3 files (filename contains the class name)
    # and get data
    classes = _ALL_CLASSES
    pattern = '({}).*(\.mp3)'.format('|'.join(classes))
    reg = re.compile(pattern, re.IGNORECASE)
    silences_start = []
    silences_length = []
    for f in os.listdir(path):
        m = reg.search(f)
        if not m: continue
        c = m.groups()[0].lower()
        data, rate = get_audio(os.path.join(path,f), max_length)
        # where and how long are the silences?
        is_silent = False
        for i, d in enumerate(data):
            if np.fabs(d) <= silence_th:
                if not is_silent:
                    silence_start_index = i
                is_silent = True
            else:
                if is_silent:
                    length_samples = i - silence_start_index
                    if length_samples > min_length * rate:
                        silences_start.append(silence_start_index / rate)
                        silences_length.append(length_samples / rate)
                is_silent = False
    arg_sorting = np.argsort(silences_start)
    for i in np.flip(arg_sorting):
        print("Start: {:.2f}, Length: {:.2f}".format(silences_start[i], silences_length[i]))
    return

# Take classifier from ./good_for_all_but_snowflake
# and replace its row for class "snowflake"
# with the corresponding row from ./good_for_most, which does pretty well with snowflake
# SORT OF WORKS
def make_frankenstein(
        main="good_for_all_but_snowflake/",
        second="good_for_most/",
        target_classes=['snowflake']
    ):
    """
    Replace 'target_classes' in 'main' classifier with 'second' classifier
    """
    A_main = np.genfromtxt(main + 'weights.csv', delimiter=',')
    b_main = np.genfromtxt(main + 'intercepts.csv', delimiter=',')
    classes_main = np.genfromtxt(main + 'classes.txt', delimiter=',', dtype=str)

    A_second = np.genfromtxt(second+'weights.csv', delimiter=',')
    b_second = np.genfromtxt(second+'intercepts.csv', delimiter=',')
    classes_second = np.genfromtxt(second+'classes.txt', delimiter=',', dtype=str)

    # We may need to re-scale rows taken from 'second' classifier
    # Might be other options;
    # for now match the norms of the intercepts
    b_main_norm = np.linalg.norm(b_main)
    b_second_norm = np.linalg.norm(b_second)
    scaling = b_main_norm / b_second_norm

    for c in target_classes:
        main_index = np.argmax(classes_main == c)
        second_index = np.argmax(classes_second == c)
        print("Class {}, main   weights mean value: {}".format(c,
            np.mean(A_main[main_index])))
        print("Class {}, second weights mean value: {}".format(c,
            np.mean(A_second[second_index])))
        # Update weights/intercepts - copy in the class row (with scaling)
        A_main[main_index] = scaling*A_second[second_index]
        b_main[main_index] = scaling*b_second[second_index]

    # re-export
    np.savetxt('frankenstein/weights.csv', A_main, delimiter=',', fmt='%.3f')
    np.savetxt('frankenstein/intercepts.csv', b_main, delimiter=',')
    np.savetxt('frankenstein/classes.txt', classes_main, delimiter='\n', fmt='%s')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utils for music classifier")
    parser.add_argument('-r', '--rename', action='store_true',
        help="Rename audio in PATH")
    parser.add_argument('-f', '--frankenstein', action='store_true',
        help="Make frankenstein classifier")
    parser.add_argument('-s', '--silence', action='store_true',
        help="Determine max silence length of clips in PATH")
    parser.add_argument('-p', '--path', type=str, default='audio',
        help="Path to folder of audio clips")
    args = parser.parse_args()

    noaction = True
    if args.rename:
        rename_audio(args.path)
        noaction = False
    if args.frankenstein:
        make_frankenstein()
        noaction = False
    if args.silence:
        determine_max_silence(args.path)
        noaction = False
    if noaction:
        parser.print_help()

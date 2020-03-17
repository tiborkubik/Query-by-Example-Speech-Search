import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk
from scipy.stats import pearsonr
from scipy.ndimage.interpolation import rotate
import statistics
import math

# Task 3: reading a sentence + spectogram creation
def task3_getSegments(toRead):
    fs, data = wavfile.read(toRead)
    data = data / 2**15             # normalisation

    # 3a
    data = data - statistics.mean(data);

    N = 512

    # length of segment
    wlen = 25e-3 * fs       # 400
    wlen = int(wlen)

    # shift between segments
    wshift = 10e-3 * fs     # 160
    wshift = int(wshift)

    woverlap = wlen - wshift;

    # 3b
    # creation of an array of segments
    segments = []

    i = 0

    while i < len(data):
        segments.append(data[i:i+wlen])
        i += wshift

    # 3c
    for i in range(0, len(segments)):
        segments[i] = segments[i] * np.hamming(len(segments[i]))
        # thanks buddy for this line, I was lost how to fill with 0s :(((
        segments[i] = np.pad(segments[i], (0, 512 - len(segments[i])), 'constant')

    # 3d + 3e
    for i in range(0, len(segments)):
        N = len(segments[i])/2

        segments[i] = np.fft.fft(segments[i], 256)
        #f, t, sgr = spectrogram(afterFft, fs, noverlap=woverlap, nfft=512)

        for j in range(0, len(segments[i])):
            segments[i][j] = 10 * np.log10(np.abs(segments[i][j] + 1e-20))

    return segments

# Task 3: plotting spectogram
def task3_PlotGraph(toRead):
        fs, data = wavfile.read(toRead)
        data = data / 2**15             # normalisation

        # 3a
        data = data - statistics.mean(data);

        # length of segment
        wlen = 25e-3 * fs       # 400
        wlen = int(wlen)
        # shift between segments
        wshift = 10e-3 * fs     # 160
        wshift = int(wshift);

        woverlap = wlen - wshift;

        f, t, sgr = spectrogram(data, fs, noverlap=woverlap, nfft=512)

        # prevod na PSD
        # (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)
        sgr_log = 10 * np.log10(np.abs(sgr + 1e-20))

        plt.figure(figsize=(9,3))

        plt.pcolormesh(t,f,sgr_log)

        plt.gca().set_title('sx84.wav')
        plt.gca().set_xlabel('Time [s]')
        plt.gca().set_ylabel('Frequency [Hz]')

        plt.tight_layout()

        plt.plot()
        plt.show()

# task 4
# Function calculates parameters of a query/sentence
# param: matrix of a segments of a query/sentence
# return: matrix of features
def task4(toRead):

    # segs -> matrix of segments of given sentence/query
    segs = task3_getSegments(toRead)

    features = []

    # len(segs) -> number of segments -> for each segment,
    for i in range(0, len(segs)):
        B = 0
        temp = []
        for j in range(0, 16):
            bnd = 0

            for x in range(0,16):
                index = B+x
                bnd = bnd + segs[i][index]

            temp.append(bnd)
            B = B + 16

        features.append(temp)

    # deleting the imaginary part
    for i in range(0, len(features)):
        for j in range(0, len(features[i])):
            features[i][j] = features[i][j].real

    return features

# Task 5: Calculating score by pearson.
def task5(sentence, query):
    paramsSentence = task4(sentence)
    paramsQuery = task4(query)

    resultScore = []
    nOfSteps = len(paramsSentence) - len(paramsQuery)

    if(nOfSteps < 0):
        return None

    for x in range(0, nOfSteps):
        toCompare = len(paramsQuery)

        temp = []
        sum = 0
        for y in range(0, toCompare):
            index = x+y

            a = pearsonr(paramsSentence[index],paramsQuery[y])

            temp.append(a)

            # getting just first part of pearson result
            temp[y] = temp[y][0]

        for y in range(0, toCompare):
            sum = sum + temp[y]

        resultForOneCmp = sum/toCompare
        resultScore.append(resultForOneCmp)

    return resultScore

# Task 6: plotting all the graphs..
def task6(sentence, query1, query2):
    fs, data = wavfile.read(sentence)
    t = np.arange(data.size)/fs

    # To check values of threshold, UNCOMMENT THIS
    checkThresh(query1, query2, t)

    # features for spectogram
    features = task4(sentence)
    features = rotate(features, angle=90)


    # function changes values so that they are the same as for signal
    featx, x1, x2 = editXaxis(features, query1, query2)

    xInt = range(0, math.ceil(len(data)/fs)+1)
    maxXval = max(xInt)

    fig, ax = plt.subplots(3)
    fig.suptitle('\"Householder\" and \"Outstanding\" vs sx84.wav')

    # Signal
    ax[0].plot(t,data)
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('signal')
    ax[0].set_xlim(0, maxXval)

    # Features
    ax[1].pcolormesh(features)
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('features')
    #ax[1].set_xlim(0, maxXval)

    # Score
    ax[2].plot(x1, query1, label = 'Householder')
    ax[2].plot(x2, query2, label = 'Outstanding')
    ax[2].set_ylabel('scores')
    ax[2].set_xlabel('t')
    ax[2].set_xlim(0, maxXval)
    ax[2].grid(alpha=1, linestyle='--')
    ax[2].legend()
    fig.set_size_inches(14.5, 8.5)
    plt.subplots_adjust(hspace = 0.5)
    plt.show()


def checkThresh(query1, query2, t):
    # checking if we found hit
    print('Hits for query1')
    for x in range(0, len(query1)):
        if(query1[x] > 0.91):

            res = (x / len(query1)) * len(t)

            print(res)
            print('\n')

    print('Hits for query2')
    for x in range(0, len(query2)):
        if(query2[x] > 0.875):

            res = (x / len(query2)) * len(t)

            print(res)
            print('\n')

def editXaxis(features, query1, query2):
    newXf = []
    newXq1 = []
    newXq2 = []

    rangeF = len(features)
    rangeQ1 = len(query1)
    rangeQ2 = len(query2)


    for i in range(0, rangeF):
        newXf.append(i/100)

    for i in range(0, rangeQ1):
        newXq1.append(i/100)

    for i in range(0, rangeQ2):
        newXq2.append(i/100)

    return newXf, newXq1, newXq2


# MAIN #

# Note: For now, you have to edit these lines separately for every sentence. I stucked in file reading in python
#       I would rework it, but there are so many projects.... sorry for that
a = task5('../sentences/sx174.wav', '../queries/q1.wav')
b = task5('../sentences/sx174.wav', '../queries/q2.wav')

task6('../sentences/sx174.wav', a, b)

"""
porting useful stuff from vktrs: https://github.com/dmarx/video-killed-the-radio-star/blob/main/Video_Killed_The_Radio_Star_Defusion.ipynb
"""

##############################################################

# audio processing


def analyze_audio_structure(
    audio_fpath,
    BINS_PER_OCTAVE = 12 * 3, # should be a multiple of twelve: https://github.com/MTG/essentia/blob/master/src/examples/python/tutorial_spectral_constantq-nsg.ipynb
    N_OCTAVES = 7,
):
    """
    via librosa docs
    https://librosa.org/doc/latest/auto_examples/plot_segmentation.html#sphx-glr-auto-examples-plot-segmentation-py
    cites: McFee and Ellis, 2014 - https://brianmcfee.net/papers/ismir2014_spectral.pdf
    """
    y, sr = librosa.load(audio_fpath)

    C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr,
                                            bins_per_octave=BINS_PER_OCTAVE,
                                            n_bins=N_OCTAVES * BINS_PER_OCTAVE)),
                                ref=np.max)

    # reduce dimensionality via beat-synchronization
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    Csync = librosa.util.sync(C, beats, aggregate=np.median)

    # I have concerns about this frame fixing operation
    beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats, x_min=0), sr=sr)

    # width=3 prevents links within the same bar
    # mode=’affinity’ here implements S_rep (after Eq. 8)
    R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity', sym=True)
    # Enhance diagonals with a median filter (Equation 2)
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))
    # build the sequence matrix (S_loc) using mfcc-similarity
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    Msync = librosa.util.sync(mfcc, beats)
    path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)
    # compute the balanced combination
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)
    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)
    A = mu * Rf + (1 - mu) * R_path

    # compute normalized laplacian and its spectrum
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    evals, evecs = scipy.linalg.eigh(L)
    # clean this up with a median filter. can help smooth over discontinuities
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))
    return dict(
        y=y,
        sr=np.array(sr).astype(np.uint32),
        tempo=tempo,
        beats=beats,
        beat_times=beat_times,
        evecs=evecs,
    )


def laplacian_segmentation(
    audio_fpath=None,
    evecs=None,
    n_clusters = 5,
    n_spectral_features = None,
):
    """
    segment audio by clustering a self-similarity matrix.
    via librosa docs
    https://librosa.org/doc/latest/auto_examples/plot_segmentation.html#sphx-glr-auto-examples-plot-segmentation-py
    cites: McFee and Ellis, 2014 - https://brianmcfee.net/papers/ismir2014_spectral.pdf
    """
    if evecs is None:
        if audio_fpath is None:
            raise Exception("One of `audio_fpath` or `evecs` must be provided")
        features = analyze_audio_structure(audio_fpath)
        evecs = features['evecs']

    if n_clusters < 2:
        seg_ids = np.zeros(evecs.shape[0], dtype=int)
        return seg_ids

    if n_spectral_features is None:
        n_spectral_features = n_clusters

    # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
    Cnorm = np.cumsum(evecs**2, axis=1)**0.5
    k = n_spectral_features
    X = evecs[:, :k] / Cnorm[:, k-1:k]


    # use these k components to cluster beats into segments
    KM = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init="auto")
    seg_ids = KM.fit_predict(X)

    return seg_ids #, beat_times, tempo


# for video duration
def get_audio_duration_seconds(audio_fpath):
    outv = subprocess.run([
        'ffprobe'
        ,'-i',audio_fpath
        ,'-show_entries', 'format=duration'
        ,'-v','quiet'
        ,'-of','csv=p=0'
        ],
        stdout=subprocess.PIPE
        ).stdout.decode('utf-8')
    return float(outv.strip())

##########################################################

# audioreactivity stuff


def full_width_plot():
    ax = plt.gca()
    ax.figure.set_figwidth(20)
    plt.show()

def display_signal(y, sr, show_spec=True, title=None, start_time=0, end_time=9999):

#     if show_spec:
#         frame_time = librosa.samples_to_time(np.arange(len(normalized_signal)), sr=sr)
#     else:
#         frame_time = librosa.frames_to_time(np.arange(len(normalized_signal)), sr=sr)

    if show_spec:
        #librosa.display.waveshow(y, sr=sr)
        times = librosa.samples_to_time(np.arange(len(y)), sr=sr)
    else:
        #times = librosa.times_like(y, sr=sr).ravel()
        times = librosa.frames_to_time(np.arange(len(y)), sr=sr).ravel()

    start_idx = np.argmax(start_time <= times)
    #end_idx = len(times) - np.argmax([end_time <= times][::-1])
    end_idx = np.argmax(end_time <= times)
    if start_idx >= end_idx:
        end_idx = -1

    times = times[start_idx:end_idx]
    y = y[start_idx:end_idx]

    plt.plot(times, y)
    if title:
        plt.title(title)
    full_width_plot()

    if show_spec:
        try:
            M = librosa.feature.melspectrogram(y=y, sr=sr)
            librosa.display.specshow(librosa.power_to_db(M, ref=np.max),
                             y_axis='mel', x_axis='time')
            full_width_plot()

        except:
            pass

    # plt.plot(frame_time, y)
    # if title:
    #     plt.title(title)
    # full_width_plot()


# https://github.com/pytti-tools/pytti-core/blob/9e8568365cfdc123d2d2fbc20d676ca0f8715341/src/pytti/AudioParse.py#L95
from scipy.signal import butter, sosfilt, sosfreqz

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='bandpass', output='sos')
    return sos

def butter_bandpass_filter(y, sr, lowcut, highcut, order=10):
    sos = butter_bandpass(lowcut, highcut, sr, order=order)
    y = sosfilt(sos, y)
    return y

########################################################################



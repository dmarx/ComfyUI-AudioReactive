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
#########################################

def get_path_to_stems():
    workspace, storyboard = load_storyboard()
    assets_root = Path(workspace.application_root) / 'shared_assets'
    #stems_path = root / "stems"
    stems_path = assets_root / "stems"
    stems_outpath = stems_path / 'htdemucs_ft' / Path(storyboard.params.audio_fpath).stem
    return stems_outpath

def ensure_stems_separated():
    stems_outpath = get_path_to_stems()
    stems_path = str(stems_outpath.parent.parent)
    if not stems_outpath.exists():
        !demucs -n htdemucs_ft -o "{stems_path}" "{storyboard.params.audio_fpath}"

def get_stem(instrument_name):
    ensure_stems_separated()
    stems_outpath = get_path_to_stems()
    stem_fpaths  = list(stems_outpath.glob('*.wav'))

    for stem_fpath in stem_fpaths:
        if instrument_name in str(stem_fpath):
            y, sr = librosa.load(stem_fpath)
            return y, sr
    raise ValueError(
        f"Unable to locate stem for instrument: {instrument_name}\n"
        f"in folder: {stems_outpath}"
    )

##########################################################################################################

# deforum compatibility sprint


import math
import numpy as np

def build_eval_scope(storyboard):
    # preload eval scope with math stuff
    math_env = {
        "abs": abs,
        "max": max,
        "min": min,
        "pow": pow,
        "round": round,
        "np": np,
        "__builtins__": None,
    }
    math_env.update(
        {key: getattr(math, key) for key in dir(math) if "_" not in key}
    )

    # add signals to scope
    for signal_name, sig_curve in storyboard.signals.items():
        sig_curve = OmegaConf.to_container(sig_curve) # zomg...
        curve = load_curve(sig_curve)
        math_env[signal_name] = curve
    return math_env


#eval(signal_mappings['noise_curve'], math_env, t=0)
#math_env['t']=0
#eval(signal_mappings['noise_curve'], math_env)


#################

custom_signal_fpath = '' # @param {'type':'string'}

def get_user_specified_signal():
    y, sr = librosa.load(custom_signal_fpath)
    return y, sr

###################

import numpy as np
from scipy import signal
from inspect import signature
from functools import partial
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
#sklearn_extra.cluster.KMedoids

# all operations must either have signature (y, sr) or return a function which does

# def rms(y, sr):
#     return librosa.feature.rms(y=y)

# def novelty(y, sr):
#     return librosa.onset.onset_strength(y, sr)

# def predominant_pulse(y, sr):
#     return librosa.beat.plp(y, sr)

def pow2(y, sr):
    return y**2

def sqrt(y, sr):
    return y**-2


# so... apparently `pow` is a python builtin. whoops. Meh, fuck it.
def _pow(k):
    def pow_(y, sr):
        return y**k
    return pow_

# def stretch(k=2):
#     return _pow(k)

# def smoosh(k=2):
#     return _pow(-k)

def stretch(y, sr):
    y = normalize(y, sr)
    return normalize(y**2, sr)

def smoosh(y, sr):
    y = normalize(y, sr)
    return normalize(y**.5, sr)

def normalize(y, sr):
    normalized_signal = np.abs(y).ravel()
    normalized_signal /= max(normalized_signal)
    return normalized_signal

######################################

def smooth(k=150):
    k=int(k)
    def smooth_(y, sr=None):
        win_smooth = signal.windows.hann(k)
        filtered = signal.convolve(y, win_smooth, mode='same') / sum(win_smooth)
        return filtered
    return smooth_

def sustain(k=500):
    k=int(k)
    def sustain_(y, sr=None):
        win_sustain = signal.windows.hann(2*k)
        win_sustain[:k]=0
        filtered = signal.convolve(y.ravel(), win_sustain, mode='same') / sum(win_sustain)
        return filtered
    return sustain_


# TODO: decay() - sustain with an exponential window

#####################333

def bandpass(low: float, high:float):
    return partial(butter_bandpass_filter, lowcut=low, highcut=high)

def threshold(low):
    def f(y, sr):
        y[y<low] = 0
        return y
    return f

def clamp(high):
    def f(y, sr):
        y[y>high] = high
        return y
    return f


###############################3


# def peak_detection(y, sr):
#     peaks, _ = find_peaks(y)
#     return peaks

# TODO: support offset, so user could e.g. take either every even or every odd peak.
def modulo(k=2, offset=0):
    #k=int(k)
    def modulo_(y, sr=None):
        #peaks = peak_detection(y, sr)
        peaks, _ = find_peaks(y)
        #selected_peaks = peaks[::k]  # Select every kth peak
        selected_peaks=[]
        for peak_index, peak in enumerate(peaks.ravel()):
            if (peak_index + offset) % k == 0:
                selected_peaks.append(peak)
        print(selected_peaks)
        selected_peaks = np.array(selected_peaks)
        new_signal = np.zeros_like(y)
        new_signal[selected_peaks] = y[selected_peaks]  # Build a new signal with only the selected peaks
        return new_signal
    return modulo_

#################################

# chatgpt wrote this, needs to be tested. also, i might want to use medoids rather than means



def quantize(k=1):
    k=int(k)
    # why doesn't it respect `k` in the closure scope? Works fine for modulo(). weird.
    #def quantize_(y, sr=None):
    def quantize_(y, sr=None, K=k):
        k=K
        # Remove zero values
        nonzero_values = y[y > 0].reshape(-1, 1)

        # If the number of nonzero values is less than k, reduce k
        if nonzero_values.shape[0] < k:
            k = nonzero_values.shape[0]

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(nonzero_values)

        # Replace each value with the centroid of its cluster
        print(f"cluster centers: {np.unique(kmeans.cluster_centers_)}")
        quantized_values = kmeans.cluster_centers_[kmeans.labels_].flatten()

        # Create a new signal
        quantized_signal = np.zeros_like(y)
        quantized_signal[y > 0] = quantized_values
        return quantized_signal
    return quantize_

#####################################3

# TODO: add ability for user to do stuff via idiomatic `keyframed` rather than convolving signals

# # TODO: add operations: slice/isolate, mute, shift_y/truncate/drop (subtract some value from amplitude)
simple_signal_operations = {
    'raw': lambda y, sr: y,
    ##########
    'rms': lambda y, sr: normalize(librosa.feature.rms(y=y).ravel(), sr),
    'novelty': librosa.onset.onset_strength,
    'predominant_pulse': librosa.beat.plp,
    'bandpass': bandpass,
    'harmonic': lambda y, sr: librosa.effects.harmonic(y=y),
    'percussive': lambda y, sr: librosa.effects.percussive(y=y),
    ##########
    'pow2': lambda y, sr: y**2,
    'stretch': stretch,
    'sqrt': sqrt,
    'smoosh': smoosh,
    'pow':_pow,
    #################
    'smooth': smooth,
    'sustain': sustain,
    # #########
    'normalize': normalize,
    'abs': lambda y, sr: np.abs(np.abs(y)),
    'threshold': threshold,
    'clamp': clamp,
    'modulo': modulo,
    'quantize':quantize,
}

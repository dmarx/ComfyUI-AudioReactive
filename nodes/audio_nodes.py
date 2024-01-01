from .audio_utils import simple_signal_operations
from types import Callable

CATEGORY = "keyframed/audioreactivity"


class SimpleSignalOperation:
    CATEGORY=CATEGORY
    RETURN_TYPES=("SIGNAL",)
    FUNCTION = "main"

    _f: Callable = None # lazy abstract method
  
    @classmethod
    def INPUT_TYPES(cls):
        outv = {
            "required": {
                "signal": ("SIGNAL",{}),
            }
        }
        return outv

    def main(self, signal, **kargs):
        y, sr = signal.get("y"), signal.get("sr")
        y, sr = f(y, sr)
        return {"y":y, "sr":sr)


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

# can i just metaclass this?
class OpRms:
    f = simple_signal_operations['rms']

# TODO: all the other ops

# TODO: export nodes boilerplate

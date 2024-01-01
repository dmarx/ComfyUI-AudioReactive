from .audio_utils import simple_signal_operations
from typing import Callable

import numpy as np
import librosa

CATEGORY = "AudioReactive"


class SimpleSignalOperation:
    CATEGORY=CATEGORY
    RETURN_TYPES=("SIGNAL",)
    FUNCTION = "main"

    f: Callable = None # lazy abstract method
  
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
        y, sr = self.f(y, sr)
        return {"y":y, "sr":sr}


# simple_signal_operations = {
#     'raw': lambda y, sr: y,
#     ##########



#     'novelty': librosa.onset.onset_strength,
class OpNovelty:
    f = simple_signal_operations['rms']

  

#     'predominant_pulse': librosa.beat.plp,
class OpNovelty:
    f = simple_signal_operations['rms']

#     'bandpass': bandpass,
class OpNovelty:
    f = simple_signal_operations['rms']

#     'harmonic': lambda y, sr: librosa.effects.harmonic(y=y),
class OpNovelty:
    f = simple_signal_operations['rms']

#     'percussive': lambda y, sr: librosa.effects.percussive(y=y),
class OpNovelty:
    f = simple_signal_operations['rms']
#     ##########
#     'pow2': lambda y, sr: y**2,

class OpNovelty:
    f = simple_signal_operations['rms']

#     'stretch': stretch,
class OpNovelty:
    f = simple_signal_operations['rms']

#     'sqrt': sqrt,
class OpNovelty:
    f = simple_signal_operations['rms']

#     'smoosh': smoosh,
class OpNovelty:
    f = simple_signal_operations['rms']

#     'pow':_pow,
class OpNovelty:
    f = simple_signal_operations['rms']
#     #################

#     'smooth': smooth,
class OpNovelty:
    f = simple_signal_operations['rms']

#     'sustain': sustain,
class OpNovelty:
    f = simple_signal_operations['rms']
#     # #########

#     'normalize': normalize,
class OpNovelty:
    f = simple_signal_operations['rms']

#     'abs': lambda y, sr: np.abs(np.abs(y)),
class OpNovelty:
    f = simple_signal_operations['rms']

#     'threshold': threshold,
class OpNovelty:
    f = simple_signal_operations['rms']
  
#     'clamp': clamp,
class OpNovelty:
    f = simple_signal_operations['rms']
  
#     'modulo': modulo,
class OpNovelty:
    f = simple_signal_operations['rms']
  
#     'quantize':quantize,
class OpNovelty:
    f = simple_signal_operations['rms']
# }


# can i just metaclass this?
class OpRms(SimpleSignalOperation):
    f = simple_signal_operations['rms']
    
#     'rms': lambda y, sr: normalize(librosa.feature.rms(y=y).ravel(), sr),
#class OpRms:
# #class SimpleSignalOperation:
#     CATEGORY=CATEGORY
#     RETURN_TYPES=("SIGNAL",)
#     FUNCTION = "main"

#     #f: Callable = None # lazy abstract method
  
#     @classmethod
#     def INPUT_TYPES(cls):
#         outv = {
#             "required": {
#                 "signal": ("SIGNAL",{}),
#             }
#         }
#         return outv

#     def main(self, signal:dict, **kargs):
#         y, sr = signal.get("y"), signal.get("sr")
#         y, sr = self.f(y, sr)
#         outv = {"y":y, "sr":sr}
#         return (outv,)

# TODO: all the other ops

NODE_CLASS_MAPPINGS = {
    "OpRms": OpRms,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpRms": "OpRms"
}

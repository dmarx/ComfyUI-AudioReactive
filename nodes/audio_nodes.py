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
class OpNovelty(SimpleSignalOperation):
    f = simple_signal_operations['novelty']

  

#     'predominant_pulse': librosa.beat.plp,
class Oppredominant_pulse(SimpleSignalOperation):
    f = simple_signal_operations['predominant_pulse']

#     'bandpass': bandpass,
class Opbandpass(SimpleSignalOperation):
    f = simple_signal_operations['bandpass']

#     'harmonic': lambda y, sr: librosa.effects.harmonic(y=y),
class Opharmonic(SimpleSignalOperation):
    f = simple_signal_operations['harmonic']

#     'percussive': lambda y, sr: librosa.effects.percussive(y=y),
class Oppercussive(SimpleSignalOperation):
    f = simple_signal_operations['percussive']
#     ##########

#     'pow2': lambda y, sr: y**2,
class Oppow2(SimpleSignalOperation):
    f = simple_signal_operations['pow2']

#     'stretch': stretch,
class Opstretch(SimpleSignalOperation):
    f = simple_signal_operations['stretch']

#     'sqrt': sqrt,
class Opsqrt(SimpleSignalOperation):
    f = simple_signal_operations['sqrt']

#     'smoosh': smoosh,
class Opsmoosh(SimpleSignalOperation):
    f = simple_signal_operations['smoosh']

#     'pow':_pow,
class Oppow(SimpleSignalOperation):
    f = simple_signal_operations['pow']
#     #################

#     'smooth': smooth,
class Opsmooth(SimpleSignalOperation):
    f = simple_signal_operations['smooth']

#     'sustain': sustain,
class Opsustain(SimpleSignalOperation):
    f = simple_signal_operations['sustain']
#     # #########

#     'normalize': normalize,
class OpNormalize(SimpleSignalOperation):
    f = simple_signal_operations['normalize']

#     'abs': lambda y, sr: np.abs(np.abs(y)),
class Opabs(SimpleSignalOperation):
    f = simple_signal_operations['abs']

#     'threshold': threshold,
class Opthreshold(SimpleSignalOperation):
    f = simple_signal_operations['threshold']
  
#     'clamp': clamp,
class Opclamp(SimpleSignalOperation):
    f = simple_signal_operations['clamp']
  
#     'modulo': modulo,
class Opmodulo(SimpleSignalOperation):
    f = simple_signal_operations['modulo']
  
#     'quantize':quantize,
class OpQuantize(SimpleSignalOperation):
    f = simple_signal_operations['quantize']
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

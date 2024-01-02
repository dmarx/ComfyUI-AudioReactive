from .audio_utils import simple_signal_operations
from typing import Callable

CATEGORY = "AudioReactive/Operators"


class SimpleSignalOperation:
    CATEGORY=CATEGORY
    RETURN_TYPES=("SIGNAL",)
    FUNCTION = "main"

    #f: Callable = None # lazy abstract method
    _f: str = ''

    def f(self, *args, **kwargs):
        return simple_signal_operations[self._f](*args, **kwargs)

    @classmethod
    def alias(cls):
        return f"{cls._f.title()} (Audio Op)"
  
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
        #y, sr = self.f(y, sr)
        #return {"y":y, "sr":sr}
        #signal = self.f(y, sr)
        signal = self.f(y=y, sr=sr)
        return (signal,)


class OpNovelty(SimpleSignalOperation):
    _f = 'novelty'


class OpPredominant_pulse(SimpleSignalOperation):
    _f = 'predominant_pulse'


class OpBandpass(SimpleSignalOperation):
    _f = 'bandpass'


class OpHarmonic(SimpleSignalOperation):
    _f = 'harmonic'


class OpPercussive(SimpleSignalOperation):
    _f = 'percussive'


class OpPow2(SimpleSignalOperation):
    _f = 'pow2'


class OpStretch(SimpleSignalOperation):
    _f = 'stretch'


class OpSqrt(SimpleSignalOperation):
    _f = 'sqrt'


class OpSmoosh(SimpleSignalOperation):
    _f = 'smoosh'


class OpPow(SimpleSignalOperation):
    _f = 'pow'


class OpSmooth(SimpleSignalOperation):
    _f = 'smooth'


class OpSustain(SimpleSignalOperation):
    _f = 'sustain'
    

class OpNormalize(SimpleSignalOperation):
    _f = 'normalize'


class OpAbs(SimpleSignalOperation):
    _f = 'abs'


class OpThreshold(SimpleSignalOperation):
    _f = 'threshold'
  

class OpClamp(SimpleSignalOperation):
    _f = 'clamp'
  

class OpModulo(SimpleSignalOperation):
    _f = 'modulo'
  

class OpQuantize(SimpleSignalOperation):
    _f = 'quantize'


class OpRms(SimpleSignalOperation):
    _f = 'rms'
    


NODE_CLASS_MAPPINGS = {
"OpNovelty":OpNovelty,
"OpPredominant_pulse":OpPredominant_pulse,
"OpBandpass":OpBandpass,
"OpHarmonic":OpHarmonic, 
"OpPercussive":OpPercussive,
"OpPow2":OpPow2,
"OpStretch":OpStretch,
"OpSqrt":OpSqrt,
"OpSmoosh":OpSmoosh,
"OpPow":OpPow,
"OpSmooth":OpSmooth,
"OpSustain":OpSustain,
"OpNormalize":OpNormalize,
"OpAbs":OpAbs,
"OpThreshold":OpThreshold,
"OpClamp":OpClamp,
"OpModulo":OpModulo,
"OpQuantize":OpQuantize,
"OpRms":OpRms,
}

NODE_DISPLAY_NAME_MAPPINGS = {k:v.alias() for k,v in NODE_CLASS_MAPPINGS.items()}

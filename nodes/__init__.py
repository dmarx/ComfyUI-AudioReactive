from .audio_operator_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

import librosa

CATEGORY="AudioReactive"

def read_audio_file(signal_fpath) -> dict:
    y, sr = librosa.load(signal_fpath)
    return {"y":y, "sr":sr} # i.e. a `SIGNAL`


class ARReadAudio:
    CATEGORY=CATEGORY
    RETURN_TYPES = ("SIGNAL",)
    FUNCTION = "main"

    @classmethod
    def INPUT_TYPES(cls):
        outv = {
            "required": {
                "fpath": ("STRING",{"default":"audio.wav"}),
            }
        }
        return outv

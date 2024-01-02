from .audio_operator_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

import librosa
from copy import deepcopy
#import warnings
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import torchvision.transforms as TT
import keyframed as kf

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

    def main(self, fpath):
        signal = read_audio_file(fpath)
        return (signal,)


NODE_CLASS_MAPPINGS["ARReadAudio"] = ARReadAudio
NODE_DISPLAY_NAME_MAPPINGS["ARReadAudio"] = "Read Audio Fpath"

############################################

# via https://github.com/dmarx/video-killed-the-radio-star/blob/main/Video_Killed_The_Radio_Star_Defusion.ipynb

def full_width_plot():
    ax = plt.gca()
    ax.figure.set_figwidth(20)
    #plt.show()

#def display_signal(
def draw_signal(
    y, sr, raw=True, #show_spec=True, title=None, 
                   start_time=0, end_time=9999):

#     if show_spec:
#         frame_time = librosa.samples_to_time(np.arange(len(normalized_signal)), sr=sr)
#     else:
#         frame_time = librosa.frames_to_time(np.arange(len(normalized_signal)), sr=sr)

    if raw: # show_spec:
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
    #if title:
    #    plt.title(title)
    full_width_plot()

    # if show_spec:
    #     try:
    #         M = librosa.feature.melspectrogram(y=y, sr=sr)
    #         librosa.display.specshow(librosa.power_to_db(M, ref=np.max),
    #                          y_axis='mel', x_axis='time')
    #         full_width_plot()

    #     except:
    #         pass

    # plt.plot(frame_time, y)
    # if title:
    #     plt.title(title)
    # full_width_plot()


    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close() # no idea if this makes a difference
    buf.seek(0)

    # Read the image into a numpy array, converting it to RGB mode
    pil_image = Image.open(buf).convert('RGB')
    #plot_array = np.array(pil_image) #.astype(np.uint8)

    # Convert the array to the desired shape [batch, channels, width, height]
    #plot_array = np.transpose(plot_array, (2, 0, 1))  # Reorder to [channels, width, height]
    #plot_array = np.expand_dims(plot_array, axis=0)   # Add the batch dimension
    #plot_array = torch.tensor(plot_array) #.float()
    #plot_array = torch.from_numpy(plot_array)

    img_tensor = TT.ToTensor()(pil_image)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.permute([0, 2, 3, 1])
    return img_tensor

############################

# via https://github.com/dmarx/ComfyUI-Keyframed/blob/main/nodes/core.py#L309

def plot_curve(curve, n, show_legend, is_pgroup=False):
        """
        
        """
        # Create a figure and axes object
        fig, ax = plt.subplots()

        # Build the plot using the provided function
        #build_plot(ax)
        #curve.plot(ax=ax)
        #curve.plot(n=n)

        eps:float=1e-9
        # value to be subtracted from keyframe to produce additional points for plotting.
        # Plotting these additional values is important for e.g. visualizing step function behavior.

        m=3
        if n < m:
            n = curve.duration + 1
            n = max(m, n)
        
        
        xs_base = list(range(int(n))) + list(curve.keyframes)
        logger.debug(f"xs_base:{xs_base}")
        xs = set()
        for x in xs_base:
            xs.add(x)
            xs.add(x-eps)

        width, height = 12,8 #inches
        plt.figure(figsize=(width, height))        

        xs = [x for x in list(set(xs)) if (x >= 0)]
        xs.sort()

        def draw_curve(curve):
            ys = [curve[x] for x in xs]
            #line = plt.plot(xs, ys, *args, **kargs)
            line = plt.plot(xs, ys, label=curve.label)
            kfx = curve.keyframes
            kfy = [curve[x] for x in kfx]
            plt.scatter(kfx, kfy, color=line[0].get_color())

        #if isinstance(curve, kf.ParameterGroup): # type collision with kf.Composition
        if is_pgroup:
            for c in curve.parameters.values():
                draw_curve(c)
        else:
            draw_curve(curve)
        if show_legend:
            plt.legend()


        #width, height = 10, 5 #inches
        #plt.figure(figsize=(width, height))

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close() # no idea if this makes a difference
        buf.seek(0)

        # Read the image into a numpy array, converting it to RGB mode
        pil_image = Image.open(buf).convert('RGB')
        #plot_array = np.array(pil_image) #.astype(np.uint8)

        # Convert the array to the desired shape [batch, channels, width, height]
        #plot_array = np.transpose(plot_array, (2, 0, 1))  # Reorder to [channels, width, height]
        #plot_array = np.expand_dims(plot_array, axis=0)   # Add the batch dimension
        #plot_array = torch.tensor(plot_array) #.float()
        #plot_array = torch.from_numpy(plot_array)

        img_tensor = TT.ToTensor()(pil_image)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.permute([0, 2, 3, 1])
        return img_tensor

class ARDrawSignal:
    CATEGORY = f"{CATEGORY}"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": ("SIGNAL", {"forceInput": True,}),
            }
        }

    def main(self, signal):
        img_tensor = draw_signal(
            y=signal['y'], 
            sr=signal['sr'], 
            raw=signal.get('is_raw',True),
        )
        return (img_tensor,)


class SignalToCurve:
    CATEGORY = f"{CATEGORY}"
    FUNCTION = "main"
    RETURN_TYPES = ("CURVE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": ("SIGNAL", {"forceInput": True,}),
            }
        }

    def main(self, signal, signal_name=None):
        driving_signal = signal["y"]
        sr = signal["sr"]
        if signal.get('is_raw',True):
            frame_time = librosa.samples_to_time(np.arange(len(driving_signal)), sr=sr)
        else:
            frame_time = librosa.frames_to_time(np.arange(len(driving_signal)), sr=sr)
        
        driving_signal_kf = kf.Curve({t:v for t,v in zip(frame_time, driving_signal)}, label=signal_name)
        return (driving_signal_kf,)

NODE_CLASS_MAPPINGS["ARDrawSignal"] = ARDrawSignal
NODE_DISPLAY_NAME_MAPPINGS["ARDrawSignal"] = "Draw Audio Signal"

############################################

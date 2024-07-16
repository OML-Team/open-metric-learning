import base64
from io import BytesIO
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import FloatTensor

from oml.const import BLACK, DEFAULT_SAMPLE_RATE, TColor


def default_spec_repr_func(audio: FloatTensor) -> FloatTensor:
    """
    Generate a default spectral representation (log-scaled MelSpec) from an audio signal.

    Args:
        audio: The input audio tensor.

    Returns:
        The spectral representation of the input audio tensor.
    """
    from torchaudio.transforms import MelSpectrogram

    melspectrogram = MelSpectrogram(sample_rate=DEFAULT_SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128)
    melspec = melspectrogram(audio)
    log_melspec = torch.log1p(melspec).squeeze(0)
    return log_melspec


def _visualize_audio(
    spec_repr: FloatTensor, color: TColor = BLACK, draw_bbox: bool = True, return_b64: bool = False
) -> Union[np.ndarray, str]:
    """
    Internal function to visualize an audio spectrogram.

    Args:
        spec_repr: The spectrogram representation.
        color: The color of the bounding box.
        draw_bbox: Whether to draw a bounding box around the spectrogram.
        return_b64: Whether to return the image as a base64 string.

    Returns:
        If return_b64 is ``False``, returns the image as an array.
        If return_b64 is ``True``, returns the image as a base64 string.
    """
    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)

    # actual image
    ax.imshow(spec_repr, aspect="auto", origin="lower")

    # bbox and axes
    if draw_bbox:
        frame_thickness = 5
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(frame_thickness)
            ax.spines[axis].set_edgecolor([c / 255 for c in color])
    ax.set_xticks([])
    ax.set_yticks([])

    # drawing
    fig.canvas.draw()
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img: Union[np.ndarray, str] = (
        base64.b64encode(buf.getvalue()).decode("ascii") if return_b64 else np.array(Image.open(buf), dtype=np.uint8)
    )
    plt.close(fig)
    return img


def visualize_audio(spec_repr: FloatTensor, color: TColor = BLACK, draw_bbox: bool = True) -> np.ndarray:
    """
    Visualize an audio spectral representation and return it as a NumPy array.

    Args:
        spec_repr: The spectral representation.
        color: The color of the bounding box.
        draw_bbox: Whether to draw a bounding box around the spectral representation.

    Returns:
        The spectral representation image as an array.
    """
    return _visualize_audio(spec_repr, color, draw_bbox, return_b64=False)  # type: ignore


def visualize_audio_with_player(
    audio: FloatTensor, spec_repr: FloatTensor, sample_rate: int, color: TColor = BLACK, draw_bbox: bool = True
) -> str:
    """
    Visualize an audio spectral representation and provide an HTML string with an audio player.

    Args:
        audio: The audio waveform.
        spec_repr: The spectral representation.
        sample_rate: The sampling rate of the audio.
        color: The color of the bounding box.
        draw_bbox: Whether to draw a bounding box around the spectral representation.

    Returns:
        An HTML string that contains the spectral representation image and an audio player.
    """
    import torchaudio

    image_base64 = _visualize_audio(spec_repr, color, draw_bbox, return_b64=True)  # type: ignore

    buf = BytesIO()
    torchaudio.save(buf, audio, sample_rate=sample_rate, format="wav")
    buf.seek(0)
    audio_base64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # generate HTML
    html = f"""
    <div style="border:5px solid {color}; padding: 3px; display: inline-block;">
        <img src="data:image/png;base64,{image_base64}" alt="Mel Spectrogram" />
        <audio controls style="display: block; width: 100%; margin-top: 3px;">
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
    </div>
    """
    return html

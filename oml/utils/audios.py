import base64
from io import BytesIO
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torchaudio
from numpy.typing import NDArray
from PIL import Image
from torch import FloatTensor

from oml.const import BLACK, TColor


def _visualize_audio(
    spec_repr: FloatTensor, color: TColor = BLACK, draw_bbox: bool = True, return_b64: bool = False
) -> Union[NDArray[np.uint8], str]:
    """
    Internal function to visualize an audio spectrogram.

    Args:
        spec_repr: The spectrogram representation as a FloatTensor.
        color: The color of the bounding box.
        draw_bbox: Whether to draw a bounding box around the spectrogram.
        return_b64: Whether to return the image as a base64 string.

    Returns:
        If return_b64 is False, returns the image as a NumPy array.
        If return_b64 is True, returns the image as a base64 string.
    """
    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)

    # actual image
    ax.imshow(spec_repr, aspect="auto", origin="lower")

    # bbox and axes
    if draw_bbox:
        frame_thickness = 5
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(frame_thickness)
            ax.spines[axis].set_edgecolor(
                [c / 255 for c in color]
            )  # TODO: change all colors from arrays to strings in vis functions
    ax.set_xticks([])
    ax.set_yticks([])

    # drawing
    fig.canvas.draw()
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img: Union[NDArray[np.uint8], str] = (
        base64.b64encode(buf.getvalue()).decode("ascii") if return_b64 else np.array(Image.open(buf), dtype=np.uint8)
    )
    plt.close(fig)
    return img


def visualize_audio(spec_repr: FloatTensor, color: TColor = BLACK, draw_bbox: bool = True) -> NDArray[np.uint8]:
    """
    Visualize an audio spectral representation and return it as a NumPy array.

    Args:
        spec_repr: The spectral representation as a FloatTensor.
        color: The color of the bounding box.
        draw_bbox: Whether to draw a bounding box around the spectral representation.

    Returns:
        The spectral representation image as a NumPy array.
    """
    return _visualize_audio(spec_repr, color, draw_bbox, return_b64=False)  # type: ignore


def visualize_audio_with_player(
    audio: FloatTensor, spec_repr: FloatTensor, sr: int, color: TColor = BLACK, draw_bbox: bool = True
) -> str:
    """
    Visualize an audio spectral representation and provide an HTML string with an audio player.

    Args:
        audio: The audio waveform as a FloatTensor.
        spec_repr: The spectral representation as a FloatTensor.
        sr: The sampling rate of the audio.
        color: The color of the bounding box.
        draw_bbox: Whether to draw a bounding box around the spectral representation.

    Returns:
        An HTML string that contains the spectral representation image and an audio player.
    """
    image_base64 = _visualize_audio(spec_repr, color, draw_bbox, return_b64=True)  # type: ignore

    buf = BytesIO()
    torchaudio.save(buf, audio, sample_rate=sr, format="wav")
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
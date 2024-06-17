import base64
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torchaudio
from numpy.typing import NDArray
from torch import FloatTensor

from oml.const import BLACK, TColor


def visualize_audio(spec_repr: FloatTensor, color: TColor = BLACK, draw_bbox: bool = True) -> NDArray[np.uint8]:
    """
    Visualizes an audio spectrogram representation as an image with optional colored bounding box.

    Parameters:
        spec_repr (FloatTensor): The spectral representation data as a 2D torch tensor.
        color (TColor): The color used for the bounding box.
        draw_bbox (bool): A flag indicating whether to draw a bounding box around the image.

    Returns:
        NDArray: An RGB image as a numpy array shaped (height, width, 3).
    """
    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100, frameon=True, linewidth=100)

    ax.imshow(spec_repr, aspect="auto", origin="lower")
    if draw_bbox:
        frame_thickness = 5
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(frame_thickness)
            ax.spines[axis].set_edgecolor(color)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(height, width, 3)
    plt.close(fig)
    return image


def visualize_audio_html(audio: FloatTensor, spec_repr: FloatTensor, sr: int, color: str = "black") -> str:
    """
    Generates an HTML representation including an image of a spectral representation and an audio player.

    Parameters:
        audio (FloatTensor): The raw audio data as a 1D torch tensor.
        spec_repr (FloatTensor): The spectral representation data as a 2D torch tensor.
        sr (int): The sample rate of the audio data.
        color (str): The color used for the HTML border around the spectral representation and audio player.

    Returns:
        str: A string containing HTML that embeds both the image and the audio player.
    """
    # generate base64 image
    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
    fig.canvas.draw()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.imshow(spec_repr, aspect="auto", origin="lower")
    ax.set_axis_off()

    buf = BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("ascii")
    plt.close(fig)

    # generate base64 audio
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

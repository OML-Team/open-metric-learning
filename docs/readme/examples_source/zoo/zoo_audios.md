
You can use an audio model from our Zoo or
use other arbitrary models after you inherited it from [IExtractor](https://open-metric-learning.readthedocs.io/en/latest/contents/interfaces.html#iextractor).

```shell
pip install open-metric-learning[audio]
```

<details style="padding-bottom: 15px">
<summary><b>See how to use models</b></summary>
<p>

[comment]:zoo-audio-start
```python
import torchaudio

from oml.models import ECAPATDNNExtractor
from oml.const import CKPT_SAVE_ROOT as CKPT_DIR, MOCK_AUDIO_DATASET_PATH as DATA_DIR

# replace it by your actual paths
ckpt_path = CKPT_DIR / "ecapa_tdnn_taoruijie.pth"
file_path = DATA_DIR / "voices" / "voice0_0.wav"

model = ECAPATDNNExtractor(weights=ckpt_path, arch="ecapa_tdnn_taoruijie", normalise_features=False).to("cpu").eval()
audio, sr = torchaudio.load(file_path)

if audio.shape[0] > 1:
    audio = audio.mean(dim=0, keepdim=True)  # mean by channels
if sr != 16000:
    audio = torchaudio.functional.resample(audio, sr, 16000)

embeddings = model.extract(audio)
```
[comment]:zoo-audio-end

</p>
</details>

### Audio models zoo

|                            model                             | Vox1_O | Vox1_E | Vox1_H |
|:------------------------------------------------------------:|:------:|:------:|:------:|
| `ECAPATDNNExtractor.from_pretrained("ecapa_tdnn_taoruijie")` |  0.86  |  1.18  |  2.17  |

*The metrics above represent Equal Error Rate (EER). Lower is better.*

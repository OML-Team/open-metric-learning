## Public datasets

We've prepared 4 popular benchmarks used by researchers to evaluate metric learning models
(see [metric learning leaderboard](https://paperswithcode.com/task/metric-learning))
in OML's [format](https://open-metric-learning.readthedocs.io/en/latest/contents/datasets.html).

Two steps are required:
1. Download an original dataset.
2. [Download](https://drive.google.com/drive/folders/12QmUbDrKk7UaYGHreQdz5_nPfXG3klNc?usp=sharing) a prepared `df.csv` or `df_with_bboxes.csv`.

After that, your dataset should look like this:

<details>
<summary>CARS 196</summary>
<p>

[Dataset page.](https://www.kaggle.com/datasets/arghyadutta1/cars196)

The dataset contains 16,185 images of 196 labels of cars.
The data is split into 8,144 training images and 8,041 testing images,
where each label has been split roughly in a 50-50 split.

```
└── CARS196
    ├── df.csv or df_with_bboxes.csv
    ├── cars_train
    │   ├── 00001.jpg
    │   └── ...
    └── cars_test
        ├── 00001.jpg
        └── ...
```
</p>
</details>


<details>
<summary>CUB 200 2011</summary>
<p>

[Dataset page.](https://www.vision.caltech.edu/datasets/cub_200_2011/)

The dataset contains 11,788 images of 200 labels belonging to birds,
5,994 for training and 5,794 for testing.

```
└── CUB_200_2011
    ├── df.csv or df_with_bboxes.csv
    └── images
        ├── 001.Black_footed_Albatross
        │   ├── Black_Footed_Albatross_0001_796111.jpg
        │   └── ...
        ├── 002.Laysan_Albatross
        │   ├── Laysan_Albatross_0001_545.jpg
        │   └── ...
        └── ...
```
</p>
</details>



<details>
<summary>InShop (DeepFashion)</summary>
<p>

[Dataset page](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html).
[Download from Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pVDZFQXRsMDZCX1E?resourcekey=0-4R4v6zl4CWhHTsUGOsTstw).

The dataset contains 52,712 images for 7,982 of clothing items.

```
└── DeepFashion_InShop
    ├── df.csv or df_with_bboxes.csv
    └── img_highres
        ├── MEN
        │   └── ...
        └── WOMEN
            └── ...
```
</p>
</details>


<details>
<summary>SOP (Stanford Online Products)</summary>
<p>

[Dataset page](https://cvgl.stanford.edu/projects/lifted_struct/).
[Download from Google Drive.](https://drive.google.com/uc?export=download&id=1TclrpQOF_ullUP99wk_gjGN8pKvtErG8)

The dataset has 22,634 labels with 120,053 product images. The first 11,318 labels (59,551 images)
are split for training and the other 11,316 (60,502 images) labels are used for testing.

```
└── Stanford_Online_Products
    ├── df.csv
    ├── bicycle_final
    │   ├── 111085122871_0.JPG
    │   └── ...
    └── cabinet_final
        ├── 110715681235_0.JPG
        └── ...
```
</p>
</details>

Note, you can find our pretrained checkpoints for these datasets in the
[Models zoo](https://github.com/OML-Team/open-metric-learning#zoo).

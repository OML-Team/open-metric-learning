To reuse as many OML's functions as possible, you need to prepare a `.csv` file in the required format.
It's not obligatory, especially if you implement your own Dataset in Python, but the format is required in case
of usage built-in datasets or Config-API. You can find the tiny example dataset [here](https://drive.google.com/drive/folders/1plPnwyIkzg51-mLUXWTjREHgc1kgGrF4).

Required columns:
* `label` - integer value indicates the label.
* `path` - path to sample.
* `split` - must be one of 2 values: `train` or `validation`.
* `is_query`, `is_gallery` - have to be `None` where `split == train` and `True`
  or `False` where `split == validation`. Note, that both values can be `True` at
  the same time. Then we will validate every item
  in the validation set using the "1 vs rest" approach (datasets of this kind are `CARS196` or `CUB`).

Optional columns:
* `category` - category which groups sets of similar labels (like `dresses`, or `furniture`).
* `x_1`, `x_2`, `y_1`, `y_2` - integers, the format is `left`, `right`, `top`, `bot` (`y_1` must be less than `y_2`).
  If only part of your images has bounding boxes, just fill the corresponding row with empty values.

[Here](https://drive.google.com/drive/folders/12QmUbDrKk7UaYGHreQdz5_nPfXG3klNc?usp=sharing)
are the tables examples for the public datasets. You can also use helper to check if your dataset
is in the right format:
```python
import pandas as pd
from oml.utils.dataframe_format import check_retrieval_dataframe_format

check_retrieval_dataframe_format(df=pd.read_csv("/path/to/your/table.csv"), dataset_root="/path/to/your/datasets/root/")
```

[comment]:dataset-start
```python
from oml.utils import (
    get_mock_audio_dataset,
    get_mock_images_dataset,
    get_mock_texts_dataset,
)
from oml.utils.dataframe_format import check_retrieval_dataframe_format

# IMAGES
df_train, df_val = get_mock_images_dataset(global_paths=True)
check_retrieval_dataframe_format(df=df_train)
check_retrieval_dataframe_format(df=df_val)

# TEXTS
df_train, df_val = get_mock_texts_dataset()
check_retrieval_dataframe_format(df=df_train)
check_retrieval_dataframe_format(df=df_val)

# AUDIO
df_train, df_val = get_mock_audio_dataset()
check_retrieval_dataframe_format(df=df_train)
check_retrieval_dataframe_format(df=df_val)

```
[comment]:dataset-end

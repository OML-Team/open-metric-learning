from oml.lightning.pipelines.parser import convert_to_new_format_if_needed


def test_old_config_conversion() -> None:
    old_config = {
        "dataframe_name": "df.csv",
        "dataset_root": "/",
        "cache_size": 1,
        "transforms_train": ".",
        "transforms_val": ".",
    }

    expected_config = {
        "dataset": {
            "name": "oml_image_dataset",
            "args": {
                "dataframe_name": "df.csv",
                "dataset_root": "/",
                "cache_size": 1,
                "transforms_train": ".",
                "transforms_val": ".",
            },
        }
    }
    assert convert_to_new_format_if_needed(old_config) == expected_config

# Training a feature extractor on CUB 200 2011 dataset

1. Download and [convert](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/datasets_converters) the dataset to the required format:
`python ../datasets_converters/convert_cub.py --dataset_root=data/CUB_200_2011`
3. Command to train: `python train_cub.py`
3. Command to validate: `python val_cub.py`

The weights of a trained model may be found in the [Models Zoo](https://github.com/OML-Team/open-metric-learning#zoo).

# Training feature extractor on CARS dataset

1. Download and [convert](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/datasets_converters) the dataset to the required format:
`python ../datasets_converters/convert_cars.py --dataset_root=data/CARS196`
2. Command to train: `python train_cars.py`
3. Command to validate: `python val_cars.py`
4. Command to inference: `python predict_cars.py`

The weights of a trained model may be found in the [Models Zoo](https://github.com/OML-Team/open-metric-learning#zoo).

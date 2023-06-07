# Training a feature extractor on DeepFashion InShop dataset

1. Download and [convert](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/datasets_converters) the dataset to the required format:
`python ../datasets_converters/convert_inshop.py --dataset_root=data/DeepFashion_InShop --no_bboxes`
2. Command to train: `python train_inshop.py`
3. Command to validate: `python val_inshop.py`
4. Command to inference: `python predict_inshop.py`

The weights of a trained model may be found in the [Models Zoo](https://github.com/OML-Team/open-metric-learning#zoo).

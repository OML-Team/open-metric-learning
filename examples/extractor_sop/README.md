# Training a feature extractor on Stanford Online Products dataset

1. Download and [convert](https://github.com/OML-Team/open-metric-learning/tree/main/examples/datasets_converters) the dataset to the required format:
`python ../datasets_converters/convert_sop.py --dataset_root=data/Stanford_Online_Products --no_bboxes`
2. Command to train: `python train_sop.py`
3. Command to validate: `python val_sop.py`

The weights of a trained model may be found in the [Models Zoo](https://github.com/OML-Team/open-metric-learning#zoo).

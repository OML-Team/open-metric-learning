# Postprocessing Research

## Stage 1
Train baseline

**DeepFashion Inshop**
```bash
python train_embedder.py dataset_root=data/DeepFashion_Inshop/ logs_root=logs/DeepFashion_Inshop
python validate_embedder.py dataset_root=data/DeepFashion_Inshop/ weights=/path/to/embedder/ckpt
```

**Stanford Online Products**
```bash
python train_embedder.py dataset_root=data/Stanford_Online_Products/ logs_root=logs/Stanford_Online_Products
python val_embedder.py dataset_root=data/DeepFashion_Inshop/ weights=/path/to/embedder/ckpt
```

## Stage 2
Train postprocessor on top of embedder

```bash
python train_postprocessor.py dataset_root=data/Stanford_Online_Products/ logs_root=logs/Stanford_Online_Products
python val_postprocessor.py
```

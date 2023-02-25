# Postprocessing Research

In this section we will train a simple embedder using the vanilla Triplet Margin Loss, after that we will
train a postprocessor, which corrects `top_n` results, retrieved by the first model.

## Train & validate an embedder

**DeepFashion Inshop**
```bash
python train_embedder.py dataset_root=data/InShop/ logs_root=logs/InShop
python validate_embedder.py dataset_root=data/Inshop/ weights=embedder_inshop.ckpt
```

**Stanford Online Products**
```bash
python train_embedder.py dataset_root=data/SOP/ logs_root=logs/SOP
python val_embedder.py dataset_root=data/SOP/ weights=embedder_sop.ckpt
```

## Train & validate a postprocessor

**DeepFashion Inshop**
```bash
python train_postprocessor.py dataset_root=data/InShop/ logs_root=logs/InShop extractor_weights=embedder_inshop.ckpt
python val_postprocessor.py dataset_root=data/InShop/ extractor_weights=embedder_inshop.ckpt postprocessor_weights=postprocessor_inshop.ckpt
```

**Stanford Online Products**
```bash
python train_postprocessor.py dataset_root=data/SOP/ logs_root=logs/SOP extractor_weights=embedder_sop.ckpt
python val_postprocessor.py dataset_root=data/SOP/ extractor_weights=embedder_sop.ckpt postprocessor_weights=postprocessor_sop.ckpt
```

## Pretrained checkpoints
If you don't want to perform training by yourself, you can download all the checkpoints mentioned above
[here](https://drive.google.com/drive/folders/1EIuAJYmgMq9AkUomHaxU8thiYyQ3kCxn?usp=share_link), namely:
* `embedder_inshop.ckpt`
* `embedder_sop.ckpt`
* `postprocessor_inshop.ckpt`
* `postprocessor_sop.ckpt`

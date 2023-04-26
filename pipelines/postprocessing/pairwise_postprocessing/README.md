# Pairwise postprocessing (re-ranking)

This Pipeline is based on the following study, completed by OML's team:

### [STIR: Siamese Transformer for Image Retrieval Postprocessing](link).

In this work, we first construct a baseline model trained with
triplet loss with hard negatives mining that performs at the state of the art
level but remains simple. Second, we introduce a novel
approach for image retrieval postprocessing called Siamese Transformer
for Image Retrieval (STIR) that reranks several top outputs in a single
forward pass. Unlike previously proposed Reranking Transformers, STIR
does not rely on global/local feature extraction and directly compares a
query image and a retrieved candidate on pixel level with the usage of
attention mechanism. The resulting approach defines a new state of the
art on standard image retrieval datasets: Stanford Online Products and
DeepFashion In-shop.

[OPEN INTERACTIVE DEMO](https://dapladoc-oml-postprocessing-demo-srcappmain-pfh2g0.streamlit.app/).

![](https://i.ibb.co/CMd56Dd/stir2.png)

## I. Train & validate a feature extractor

**DeepFashion Inshop**
```bash
python train_extractor.py dataset_root=data/InShop/ logs_root=logs/InShop
python validate_extractor.py dataset_root=data/Inshop/ weights=extractor_inshop.ckpt
```

**Stanford Online Products**
```bash
python train_extractor.py dataset_root=data/SOP/ logs_root=logs/SOP
python validate_extractor.py dataset_root=data/SOP/ weights=extractor_sop.ckpt
```

## II. Train & validate a postprocessor

**DeepFashion Inshop**
```bash
python train_postprocessor.py dataset_root=data/InShop/ logs_root=logs/InShop extractor_weights=extractor_inshop.ckpt
python validate_postprocessor.py dataset_root=data/InShop/ extractor_weights=extractor_inshop.ckpt postprocessor_weights=postprocessor_inshop.ckpt
```

**Stanford Online Products**
```bash
python train_postprocessor.py dataset_root=data/SOP/ logs_root=logs/SOP extractor_weights=extractor_sop.ckpt
python validate_postprocessor.py dataset_root=data/SOP/ extractor_weights=extractor_sop.ckpt postprocessor_weights=postprocessor_sop.ckpt
```

## Pretrained checkpoints
If you don't want to perform training by yourself, you can download all the checkpoints mentioned above
[here](https://drive.google.com/drive/folders/1EIuAJYmgMq9AkUomHaxU8thiYyQ3kCxn?usp=share_link), namely:
* `extractor_inshop.ckpt`
* `extractor_sop.ckpt`
* `postprocessor_inshop.ckpt`
* `postprocessor_sop.ckpt`

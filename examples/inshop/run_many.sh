#!/bin/bash


#########################################################################
#                                                                       #
#                  ARCFACE  WITH  S = 64  AND  M = 0.5                  #
#                                                                       #
#########################################################################

# DEFAULT
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.5 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.5 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + MARGIN 0.1
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.5 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.5 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + MARGIN 0.1 + CLF_WEIGHT 0.25
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.25 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.5 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + MARGIN 0.1 + CLF_WEIGHT 0.75
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.75 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.5 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + CLF_WEIGHT 0.25
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.25 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.5 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + CLF_WEIGHT 0.75
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.75 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.5 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.5 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.5 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + MARGIN 0.1
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.5 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.5 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + CLF_WEIGHT 0.75
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.75 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.5 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + CLF_WEIGHT 0.25
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.25 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.5 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + MARGIN 0.1 + CLF_WEIGHT 0.25
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.25 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.5 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + MARGIN 0.1 + CLF_WEIGHT 0.75
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.75 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.5 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5




#########################################################################
#                                                                       #
#                  ARCFACE  WITH  S = 64  AND  M = 0.4                  #
#                                                                       #
#########################################################################




# DEFAULT
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.5 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + MARGIN 0.1
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.5 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + MARGIN 0.1 + CLF_WEIGHT 0.25
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.25 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + MARGIN 0.1 + CLF_WEIGHT 0.75
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.75 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + CLF_WEIGHT 0.25
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.25 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + CLF_WEIGHT 0.75
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.75 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.5 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + MARGIN 0.1
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.5 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + CLF_WEIGHT 0.75
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.75 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + CLF_WEIGHT 0.25
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.25 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + MARGIN 0.1 + CLF_WEIGHT 0.25
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.25 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + MARGIN 0.1 + CLF_WEIGHT 0.75
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.75 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5




#########################################################################
#                                                                       #
#                  ARCFACE  WITH  S = 48  AND  M = 0.4                  #
#                                                                       #
#########################################################################




# DEFAULT
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.5 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=48 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + MARGIN 0.1
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.5 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=48 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + MARGIN 0.1 + CLF_WEIGHT 0.25
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.25 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=48 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + MARGIN 0.1 + CLF_WEIGHT 0.75
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.75 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=48 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + CLF_WEIGHT 0.25
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.25 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=48 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + CLF_WEIGHT 0.75
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.75 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=48 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.5 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=48 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + MARGIN 0.1
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.5 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=48 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + CLF_WEIGHT 0.75
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.75 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=48 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + CLF_WEIGHT 0.25
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.25 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=48 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + MARGIN 0.1 + CLF_WEIGHT 0.25
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.25 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=48 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + MARGIN 0.1 + CLF_WEIGHT 0.75
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.75 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.4 \
clf_criterion.args.s=48 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5




#########################################################################
#                                                                       #
#                  ARCFACE  WITH  S = 64  AND  M = 0.2                  #
#                                                                       #
#########################################################################




# DEFAULT
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.5 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.2 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + MARGIN 0.1
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.5 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.2 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + MARGIN 0.1 + CLF_WEIGHT 0.25
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.25 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.2 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + MARGIN 0.1 + CLF_WEIGHT 0.75
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.75 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.2 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + CLF_WEIGHT 0.25
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.25 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.2 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# DEFAULT + CLF_WEIGHT 0.75
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.75 \
clf_criterion.name=arcface \
clf_criterion.args.m=0.2 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.5 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.2 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + MARGIN 0.1
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.5 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.2 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + CLF_WEIGHT 0.75
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.75 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.2 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + CLF_WEIGHT 0.25
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.25 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.2 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=null \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + MARGIN 0.1 + CLF_WEIGHT 0.25
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.25 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.2 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5

# MLP ARCFACE + MARGIN 0.1 + CLF_WEIGHT 0.75
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
max_epochs=60 \
clf_loss_weight=0.75 \
clf_criterion.name=mlp_arcface \
clf_criterion.args.in_features=384 \
clf_criterion.args.num_classes=3985 \
clf_criterion.args.mlp_features=[448, 512] \
clf_criterion.args.m=0.2 \
clf_criterion.args.s=64 \
emb_criterion.args.margin=0.1 \
optimizer.name=adamw \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5






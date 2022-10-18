#!/bin/bash

# DEFAULT
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
bs_train=128 \
max_epochs=35 \
criterion.name=arcface \
criterion.args.m=0.5 \
criterion.args.s=64 \
optimizer.name=adam \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5



# DEFAULT WITH PROJECTION 512
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit_with_projection \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
+model.args.feature_size=512 \
bs_train=128 \
max_epochs=35 \
criterion.name=arcface \
criterion.args.m=0.5 \
criterion.args.s=64 \
optimizer.name=adam \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5



# DEFAULT WITH PROJECTION 384
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit_with_projection \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
+model.args.feature_size=384 \
bs_train=128 \
max_epochs=35 \
criterion.name=arcface \
criterion.args.m=0.5 \
criterion.args.s=64 \
optimizer.name=adam \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5



# DEFAULT WITH PROJECTION 256
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit_with_projection \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
+model.args.feature_size=256 \
bs_train=128 \
max_epochs=35 \
criterion.name=arcface \
criterion.args.m=0.5 \
criterion.args.s=64 \
optimizer.name=adam \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5



# DEFAULT WITH PROJECTION 128
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit_with_projection \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
+model.args.feature_size=128 \
bs_train=128 \
max_epochs=35 \
criterion.name=arcface \
criterion.args.m=0.5 \
criterion.args.s=64 \
optimizer.name=adam \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5



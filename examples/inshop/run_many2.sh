#!/bin/bash



# DEFAULT WITH BATCH 256
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit_with_projection \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
+model.args.feature_size=512 \
bs_train=256 \
max_epochs=35 \
criterion.name=arcface \
criterion.args.m=0.5 \
criterion.args.s=64 \
optimizer.name=adam \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5



# DEFAULT WITH BATCH 384
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit_with_projection \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
+model.args.feature_size=512 \
bs_train=384 \
max_epochs=35 \
criterion.name=arcface \
criterion.args.m=0.5 \
criterion.args.s=64 \
optimizer.name=adam \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5



# DEFAULT WITH M 0.4
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit_with_projection \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
+model.args.feature_size=512 \
bs_train=256 \
max_epochs=35 \
criterion.name=arcface \
criterion.args.m=0.4 \
criterion.args.s=64 \
optimizer.name=adam \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5



# DEFAULT WITH S 48
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit_with_projection \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
+model.args.feature_size=512 \
bs_train=256 \
max_epochs=35 \
criterion.name=arcface \
criterion.args.m=0.5 \
criterion.args.s=48 \
optimizer.name=adam \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5



# DEFAULT WITH M 0.4 S 48
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py \
model.name=vit_with_projection \
model.args.arch=vits16 \
model.args.weights=vits16_dino \
+model.args.feature_size=512 \
bs_train=256 \
max_epochs=35 \
criterion.name=arcface \
criterion.args.m=0.4 \
criterion.args.s=48 \
optimizer.name=adam \
optimizer.args.lr=1e-5 \
optimizer.args.weight_decay=1e-5
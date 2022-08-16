# 1. NO BANKS
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py --config-name train_inshop_my.yaml \
criterion.args.miner.args.top_positive=2 \
criterion.args.miner.args.top_negative=2 \
criterion.args.miner.args.top_negative_gap=0 \
model.args.normalise_features=true \
criterion.args.margin=0.1 \
exp=8


# 2. NO BANKS. Negative gap == n_instances
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py --config-name train_inshop_my.yaml \
criterion.args.miner.args.top_positive=2 \
criterion.args.miner.args.top_negative=2 \
criterion.args.miner.args.top_negative_gap=4 \
model.args.normalise_features=true \
criterion.args.margin=0.1 \
exp=9


# 3. NO BANKS. Negative gap < n_instances
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py --config-name train_inshop_my.yaml \
criterion.args.miner.args.top_positive=2 \
criterion.args.miner.args.top_negative=2 \
criterion.args.miner.args.top_negative_gap=2 \
model.args.normalise_features=true \
criterion.args.margin=0.1 \
exp=10


# 4. Bank
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py --config-name train_inshop_my_bank.yaml \
criterion.args.miner.args.bank_size_in_batches=10 \
criterion.args.miner.args.miner.args.top_positive=2 \
criterion.args.miner.args.miner.args.top_negative=2 \
criterion.args.miner.args.miner.args.top_negative_gap=0 \
model.args.normalise_features=true \
criterion.args.margin=0.1 \
exp=11


# 5. Bank. Negative gap == n_instances
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py --config-name train_inshop_my_bank.yaml \
criterion.args.miner.args.bank_size_in_batches=10 \
criterion.args.miner.args.top_positive=2 \
criterion.args.miner.args.top_negative=2 \
criterion.args.miner.args.top_negative_gap=4 \
model.args.normalise_features=true \
criterion.args.margin=0.1 \
exp=12


# 6. Bank. More size
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py --config-name train_inshop_my_bank.yaml \
criterion.args.miner.args.bank_size_in_batches=30 \
criterion.args.miner.args.miner.args.top_positive=2 \
criterion.args.miner.args.miner.args.top_negative=2 \
criterion.args.miner.args.miner.args.top_negative_gap=0 \
model.args.normalise_features=true \
criterion.args.margin=0.1 \
exp=13



# 7. Bank. More size. Negative gap == n_instances
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py --config-name train_inshop_my_bank.yaml \
criterion.args.miner.args.bank_size_in_batches=30 \
criterion.args.miner.args.top_positive=2 \
criterion.args.miner.args.top_negative=2 \
criterion.args.miner.args.top_negative_gap=4 \
model.args.normalise_features=true \
criterion.args.margin=0.1 \
exp=14

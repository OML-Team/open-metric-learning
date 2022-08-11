# 1. NO BANKS
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py --config-name train_inshop_my.yaml \
criterion.args.miner.args.top_positive=2 \
criterion.args.miner.args.top_negative=2 \
criterion.args.miner.args.top_negative_gap=0 \
exp=1


# 2. NO BANKS. Negative gap == n_instances
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py --config-name train_inshop_my.yaml \
criterion.args.miner.args.top_positive=2 \
criterion.args.miner.args.top_negative=2 \
criterion.args.miner.args.top_negative_gap=4 \
exp=2


# 3. NO BANKS. Negative gap < n_instances
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py --config-name train_inshop_my.yaml \
criterion.args.miner.args.top_positive=2 \
criterion.args.miner.args.top_negative=2 \
criterion.args.miner.args.top_negative_gap=2 \
exp=3


# 4. Bank
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py --config-name train_inshop_my_bank.yaml \
criterion.args.miner.args.bank_size_in_batches=10 \
criterion.args.miner.args.miner.args.top_positive=2 \
criterion.args.miner.args.miner.args.top_negative=2 \
criterion.args.miner.args.miner.args.top_negative_gap=0 \
exp=4


# 5. Bank. Negative gap == n_instances
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py --config-name train_inshop_my_bank.yaml \
criterion.args.miner.args.bank_size_in_batches=10 \
criterion.args.miner.args.top_positive=2 \
criterion.args.miner.args.top_negative=2 \
criterion.args.miner.args.top_negative_gap=4 \
exp=5


# 6. Bank. More size
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py --config-name train_inshop_my_bank.yaml \
criterion.args.miner.args.bank_size_in_batches=30 \
criterion.args.miner.args.miner.args.top_positive=2 \
criterion.args.miner.args.miner.args.top_negative=2 \
criterion.args.miner.args.miner.args.top_negative_gap=0 \
exp=6


# 7. Bank. More size. Negative gap == n_instances
ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py --config-name train_inshop_my_bank.yaml \
criterion.args.miner.args.bank_size_in_batches=30 \
criterion.args.miner.args.top_positive=2 \
criterion.args.miner.args.top_negative=2 \
criterion.args.miner.args.top_negative_gap=4 \
exp=7

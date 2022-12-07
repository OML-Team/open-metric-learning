ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=32 sampler.args.n_instances=4 sampler.args.n_categories=2

ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=32 sampler.args.n_instances=4 sampler.args.n_categories=3

ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=24 sampler.args.n_instances=4 sampler.args.n_categories=4

ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=19 sampler.args.n_instances=4 sampler.args.n_categories=5

ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=16 sampler.args.n_instances=4 sampler.args.n_categories=6

ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=13 sampler.args.n_instances=4 sampler.args.n_categories=7

ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=12 sampler.args.n_instances=4 sampler.args.n_categories=8

ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=10 sampler.args.n_instances=4 sampler.args.n_categories=9

ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=9 sampler.args.n_instances=6 sampler.args.n_categories=10

ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=8 sampler.args.n_instances=6 sampler.args.n_categories=11

ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=8 sampler.args.n_instances=6 sampler.args.n_categories=12

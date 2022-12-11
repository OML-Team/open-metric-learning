ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=72 sampler.args.n_instances=4 sampler.args.n_categories=2

ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=48 sampler.args.n_instances=4 sampler.args.n_categories=3

ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=36 sampler.args.n_instances=4 sampler.args.n_categories=4

ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=28 sampler.args.n_instances=4 sampler.args.n_categories=5

ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=24 sampler.args.n_instances=4 sampler.args.n_categories=6

ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=20 sampler.args.n_instances=4 sampler.args.n_categories=7

ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=18 sampler.args.n_instances=4 sampler.args.n_categories=8

ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=16 sampler.args.n_instances=4 sampler.args.n_categories=9

ps aux | grep train_sop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_sop.py sampler.args.n_labels=14 sampler.args.n_instances=4 sampler.args.n_categories=10

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=75 sampler.args.n_instances=4 sampler.args.n_categories=2 valid_period=2

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=50 sampler.args.n_instances=4 sampler.args.n_categories=3 valid_period=2

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=37 sampler.args.n_instances=4 sampler.args.n_categories=4 valid_period=2

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=30 sampler.args.n_instances=4 sampler.args.n_categories=5

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=25 sampler.args.n_instances=4 sampler.args.n_categories=6

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=21 sampler.args.n_instances=4 sampler.args.n_categories=7

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=18 sampler.args.n_instances=4 sampler.args.n_categories=8

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=16 sampler.args.n_instances=4 sampler.args.n_categories=9

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=15 sampler.args.n_instances=4 sampler.args.n_categories=10

# the same, but margin

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=75 sampler.args.n_instances=4 sampler.args.n_categories=2 model.args.normalise_features=True criterion.args.margin=0.15 valid_period=2

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=50 sampler.args.n_instances=4 sampler.args.n_categories=3 model.args.normalise_features=True criterion.args.margin=0.15 valid_period=2

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=37 sampler.args.n_instances=4 sampler.args.n_categories=4 model.args.normalise_features=True criterion.args.margin=0.15 valid_period=2

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=30 sampler.args.n_instances=4 sampler.args.n_categories=5 model.args.normalise_features=True criterion.args.margin=0.15

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=25 sampler.args.n_instances=4 sampler.args.n_categories=6 model.args.normalise_features=True criterion.args.margin=0.15

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=21 sampler.args.n_instances=4 sampler.args.n_categories=7 model.args.normalise_features=True criterion.args.margin=0.15

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=18 sampler.args.n_instances=4 sampler.args.n_categories=8 model.args.normalise_features=True criterion.args.margin=0.15

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=16 sampler.args.n_instances=4 sampler.args.n_categories=9 model.args.normalise_features=True criterion.args.margin=0.15

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=15 sampler.args.n_instances=4 sampler.args.n_categories=10 model.args.normalise_features=True criterion.args.margin=0.15

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py criterion.args.margin=0.2 model.args.normalise_features=True postfix=margin

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=5 sampler.args.n_categories=15 postfix=more_categories

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=5 sampler.args.n_categories=15 criterion.args.margin=0.2 model.args.normalise_features=True postfix=margin_more_categories

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=30 sampler.args.n_categories=2 postfix=more_labels

ps aux | grep train_inshop.py | awk '{print $2}' | xargs kill -9 $1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_labels=30 sampler.args.n_categories=2 criterion.args.margin=0.2 model.args.normalise_features=True postfix=margin_more_labels

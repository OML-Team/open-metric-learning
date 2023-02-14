ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_categories=1 sampler.args.n_labels=20 postfix=categories_1

ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_categories=2 sampler.args.n_labels=10 postfix=categories_2

ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_categories=3 sampler.args.n_labels=6 postfix=categories_3

ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_categories=4 sampler.args.n_labels=5 postfix=categories_4

ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1
python train_inshop.py sampler.args.n_categories=5 sampler.args.n_labels=4 postfix=categories_5

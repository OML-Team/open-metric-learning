export PYTHONWARNINGS=ignore
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

# p = 6, k = 6
python train_inshop.py postfix=p-6-k-6 bs_n_cls=6 bs_n_samples=6 pod=pod1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

python train_inshop.py postfix=p-6-k-6_vanila_tri bs_n_cls=6 bs_n_samples=6 criterion.args.margin=0.1 model.args.normalise_features=True pod=pod1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

python train_inshop.py postfix=p-6-k-6_cat bs_n_cls=6 bs_n_samples=6 bs_n_categories=1 pod=pod1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

python train_inshop.py postfix=p-6-k-6_vanila_cat bs_n_cls=6 bs_n_samples=6 criterion.args.margin=0.1 model.args.normalise_features=True bs_n_categories=1 pod=pod1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

# p = 4 k = 5 c = 2
python train_inshop.py postfix=p-4-k-5-c-2 bs_n_cls=4 bs_n_samples=5 bs_n_categories=2 pod=pod1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

python train_inshop.py postfix=p-4-k-5-c-2_vanila bs_n_cls=4 bs_n_samples=5 bs_n_categories=2 criterion.args.margin=0.1 model.args.normalise_features=True pod=pod1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

# p = 8 k = 5
python train_inshop.py postfix=p-8-k-5 bs_n_cls=8 bs_n_samples=5 pod=pod1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

python train_inshop.py postfix=p-8-k-5_vanila_tri bs_n_cls=8 bs_n_samples=5 criterion.args.margin=0.1 model.args.normalise_features=True pod=pod1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

python train_inshop.py postfix=p-8-k-5_cat bs_n_cls=8 bs_n_samples=5 bs_n_categories=1 pod=pod1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

python train_inshop.py postfix=p-8-k-5_vanila_cat bs_n_cls=8 bs_n_samples=5 criterion.args.margin=0.1 model.args.normalise_features=True bs_n_categories=1 pod=pod1
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

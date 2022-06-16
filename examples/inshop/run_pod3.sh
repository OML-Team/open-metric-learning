export PYTHONWARNINGS=ignore

ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

python train_inshop.py postfix=baseline pod=pod3
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

python train_inshop.py postfix=vanila_tri criterion.args.margin=0.1 model.args.normalise_features=True pod=pod3
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

python train_inshop.py postfix=big_imgs im_size=304 pod=pod3
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

python train_inshop.py postfix=bix_imgs_vanila im_size=304 criterion.args.margin=0.1 model.args.normalise_features=True pod=pod3
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

# the same but with cat sampler

python train_inshop.py postfix=cat bs_n_categories=1 pod=pod3
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

python train_inshop.py postfix=vanila_tri_cat bs_n_categories=1 criterion.args.margin=0.1 model.args.normalise_features=True pod=pod3
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

python train_inshop.py postfix=big_imgs_cat bs_n_categories=1 im_size=304 pod=pod3
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

python train_inshop.py postfix=bix_imgs_vanila_cat bs_n_categories=1 im_size=304 criterion.args.margin=0.1 model.args.normalise_features=True pod=pod3
ps aux | grep multiprocessing.spawn | awk '{print $2}' | xargs kill -9 $1

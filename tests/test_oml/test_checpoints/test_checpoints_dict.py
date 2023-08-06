from pathlib import Path
import torch
from oml.lightning.pipelines.train import extractor_training_pipeline
from oml.models import ResnetExtractor

def create_syntatic_train(root_path):
    # use dataset from 
    # https://drive.google.com/drive/folders/1plPnwyIkzg51-mLUXWTjREHgc1kgGrF4
    
    #1 epoch arcface
    cfg = {'postfix': 'test_delete_criterion',
           'seed': 42, 'precision': 32,
           'accelerator': 'cpu',
           'devices': 1,
           'dataframe_name': 'df.csv',
           'dataset_root': str(root_path) + '/data/',
           'logs_root': 'logs/tiny/',
           'logs_folder': '${now:%Y-%m-%d_%H-%M-%S}_${postfix}',
           'show_dataset_warnings': False, 'num_workers': 3,
           'cache_size': 0,
           'transforms_train': {'name': 'augs_hypvit_torch', 'args': {'im_size': 32}},
           'transforms_val': {'name': 'norm_resize_hypvit_torch', 'args': {'im_size': 32, 'crop_size': 32}},
           'sampler': {'name':'balance','args':{'n_labels':2,'n_instances':3}},
           'bs_train': 10, 'bs_val': 10, 'max_epochs': 1, 'valid_period': 1,
           'metric_args': {'metrics_to_exclude_from_visualization': ['cmc'],
                           'cmc_top_k': [1], 'map_top_k': [5], 'pfc_variance': [0.5, 0.9, 0.99], 'return_only_main_category': True, 'visualize_only_main_category': True},
           'log_images': False, 'metric_for_checkpointing': 'OVERALL/cmc/1',
           'extractor': {'name': 'resnet', 'args': {'arch': 'resnet50', 'weights': 'resnet50_imagenet1k_v1', 'normalise_features': True, 'remove_fc': False, 'gem_p': 1}},
           'criterion': {'name': 'arcface', 'args': {'smoothing_epsilon': 0.0, 'm': 0.4, 's': 64, 'in_features': 1000, 'num_classes': 7}},
           'optimizer': {'name': 'adam', 'args': {'lr': 1e-05}}, 'scheduling': None, 'hydra_dir': '${logs_root}/${logs_folder}/',
           'tags': ['${postfix}', 'test']}
    extractor_training_pipeline(cfg)

def test_exist_criterion(path_weights):
    weights = torch.load(path_weights, map_location=torch.device('cpu'))
    state_dict=weights['state_dict']
    
    assert 'criterion.weight' not in state_dict, 'Error'
    
    try:
        model = ResnetExtractor(path_weights, arch='resnet50', gem_p=1, remove_fc=False, normalise_features=True)
    except Exception as e:
        print('Error load model...')
        print(e)
        
    print('all tests ok')
     

if __name__ == "__main__":
    root_path=Path('/Applications/programming/open-metric-learning/')
    
    create_syntatic_train(root_path)
    test_exist_criterion(root_path / Path('checkpoints/best.ckpt'))
    
    
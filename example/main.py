import argparse
import yaml
import sys
sys.path.append('../') 

from imbalanceddl.utils import fix_all_seed, prepare_store_name, prepare_folders
from imbalanceddl.net.network import build_model
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.strategy.build_trainer import build_trainer


def get_args():
    parser = argparse.ArgumentParser(
        add_help=False, description='PyTorch Deep Imbalanced Training')

    # Load params from config file
    parser.add_argument('-c', '--config', help='Path to configuration file')
    args, _ = parser.parse_known_args()
    config = {}
    # Default settings
    if args.config:
        with open(args.config) as fp:
            config = yaml.load(fp, Loader=yaml.SafeLoader)

    # Imbalance dataset
    parser.add_argument('--dataset',
                        default='cifar10',
                        type=str,
                        help='dataset to use')

    parser.add_argument('--imb_type',
                        default="exp",
                        type=str,
                        choices=['exp', 'step'],
                        help='imbalance type')

    parser.add_argument('--layers',
                        default="layer0",
                        type=str,
                        choices=['layer0','layer1', 'layer2','layer3'],
                        help='imbalance type')

    parser.add_argument('--imb_factor',
                        default=0.01,
                        type=float,
                        help='imbalance factor')

    parser.add_argument('--p',
                        default=0.1,
                        type=float,
                        help='probably')

    parser.add_argument('--dim',
                        default= 1,
                        type=int,
                        nargs='+',
                        # choices=[[1],[2,3],[0,2,3],[1,2,3]],
                        help='pono, in, bn, ln')

    parser.add_argument('--lam',
                        default=1,
                        type=float,
                        help='lam_image')
    # Strategy
    parser.add_argument('--strategy',
                        default="ERM",
                        type=str,
                        choices=[
                            'ERM', 'DRW', 'LDAM_DRW', 'Mixup_DRW',
                            'Remix_DRW','Remix',
                            'Reweight_CB', 'MAMix_DRW',
                            'FAST_DRW','FAST','FAST_LDAM_DRW','FAST_LDAM_DRW_cut',"FAST_LDAM_DRW_mixup"
                        ],
                        help='select strategy for trainer')

    parser.add_argument('--learning_rate',
                        default=0.1,
                        type=float,
                        metavar='LR',
                        help='initial learning rate',
                        dest='lr')

    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        metavar='M',
                        help='momentum')

    parser.add_argument('--wd',
                        '--weight-decay',
                        default=2e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--head_threshold',
                        default= 3,
                        type=int,
                        help='threshold for distinguish head and tail')
    # Seed
    parser.add_argument('--seed',
                        default=2023,
                        type=int,
                        help='seed for initializing training')
    # Log
    parser.add_argument('--root_log', type=str, default='log')
    parser.add_argument('--root_model', type=str, default='checkpoint')
    # Assign GPU
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    # Evaluation with Best Model
    parser.add_argument('--best_model',
                        default=None,
                        type=str,
                        metavar='PATH',
                        help='Path to Best Model')

    # update config from command line
    parser.set_defaults(**config)
    args = parser.parse_args()

    return args


def main():
    config = get_args()
    # Prepare Log
    prepare_store_name(config)
    print("=> Store Name = {}".format(config.store_name))
    prepare_folders(config)

    # Fix Seed
    fix_all_seed(config.seed)

    # Build Model
    model = build_model(config)
    # Build Dataset
    imbalance_dataset = ImbalancedDataset(config, dataset_name=config.dataset)
    # Build Trainer
    trainer = build_trainer(config,
                            imbalance_dataset,
                            model=model,
                            strategy=config.strategy)
    # Test with Best Model or Train from scratch
    if config.best_model is not None:
        print("=> Eval with Best Model !")
        trainer.eval_best_model()
    else:
        print("=> Start Train Val !")
        trainer.do_train_val()
    print("=> All Completed !")


if __name__ == "__main__":
    main()

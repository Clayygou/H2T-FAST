from imbalanceddl.strategy import MixupTrainer
from imbalanceddl.strategy import RemixTrainer
from imbalanceddl.strategy import MAMixTrainer
from imbalanceddl.strategy import ERMTrainer
from imbalanceddl.strategy import DRWTrainer
from imbalanceddl.strategy import LDAMDRWTrainer
from imbalanceddl.strategy import ReweightCBTrainer
from imbalanceddl.strategy import FASTTrainer

from imbalanceddl.strategy import FASTDRWTrainer
from imbalanceddl.strategy import RemixDRWTrainer
from imbalanceddl.strategy import FASTLDAMDRWTrainer
from imbalanceddl.strategy import FASTLDAMDRWTrainer_cut
from imbalanceddl.strategy import FASTLDAMDRWTrainer_mixup

def build_trainer(cfg, imbalance_dataset, model=None, strategy=None):
    """
    Build various strategy (trainer) specified by users
    """
    if strategy == 'Mixup_DRW':
        print("=> Mixup Trainer !")
        trainer = MixupTrainer(cfg,
                               imbalance_dataset,
                               model=model,
                               strategy=strategy)
    elif strategy == 'Remix_DRW':
        print("=> Remix Trainer !")
        trainer = RemixDRWTrainer(cfg,
                               imbalance_dataset,
                               model=model,
                               strategy=strategy)
    elif strategy == 'FAST_LDAM_DRW_cut':
        print("=> FASTLDAMDRWTrainer_cut Trainer !")
        trainer = FASTLDAMDRWTrainer_cut(cfg,
                               imbalance_dataset,
                               model=model,
                               strategy=strategy)
    elif strategy == 'FAST_LDAM_DRW_mixup':
        print("=> FASTLDAMDRWTrainer_cut Trainer !")
        trainer = FASTLDAMDRWTrainer_mixup(cfg,
                               imbalance_dataset,
                               model=model,
                               strategy=strategy)
    elif strategy == 'MAMix_DRW':
        print("=> MAMix Trainer !")
        trainer = MAMixTrainer(cfg,
                               imbalance_dataset,
                               model=model,
                               strategy=strategy)
    elif strategy == 'ERM':
        print("=> ERM Trainer !")
        trainer = ERMTrainer(cfg,
                             imbalance_dataset,
                             model=model,
                             strategy=strategy)
    elif strategy == 'DRW':
        print("=> DRW Trainer !")
        trainer = DRWTrainer(cfg,
                             imbalance_dataset,
                             model=model,
                             strategy=strategy)
    elif strategy == 'LDAM_DRW':
        print("=> LDAM_DRW Trainer !")
        trainer = LDAMDRWTrainer(cfg,
                                 imbalance_dataset,
                                 model=model,
                                 strategy=strategy)
    elif strategy == 'Reweight_CB':
        print("=> Reweight_CB Trainer !")
        trainer = ReweightCBTrainer(cfg,
                                    imbalance_dataset,
                                    model=model,
                                    strategy=strategy)
    elif strategy == 'Remix':
        print("=> DRW Trainer !")
        trainer = RemixTrainer(cfg,
                             imbalance_dataset,
                             model=model,
                             strategy=strategy)
    elif strategy == 'FAST':
        print("=> DRW Trainer !")
        trainer = FASTTrainer(cfg,
                             imbalance_dataset,
                             model=model,
                             strategy=strategy)
    elif strategy == 'FAST_DRW':
        print("=> DRW Trainer !")
        trainer = FASTDRWTrainer(cfg,
                             imbalance_dataset,
                             model=model,
                             strategy=strategy)

    elif strategy == 'FAST_LDAM_DRW':
        print("=> DRW Trainer !")
        trainer = FASTLDAMDRWTrainer(cfg,
                             imbalance_dataset,
                             model=model,
                             strategy=strategy)
    else:
        raise NotImplementedError

    return trainer

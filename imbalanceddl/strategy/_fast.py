import numpy as np
import torch
import torch.nn as nn
from .trainer import Trainer
import random

from imbalanceddl.utils.utils import AverageMeter
from imbalanceddl.utils.metrics import accuracy

from imbalanceddl.loss import LDAMLoss,FocalLoss

import numpy as np
import torch
import torch.nn as nn
from .trainer import Trainer
import random
import math

from imbalanceddl.utils.utils import AverageMeter
from imbalanceddl.utils.metrics import accuracy

from imbalanceddl.loss import LDAMLoss

def FAST_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)                                                             


class FASTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_epoch = 0

    def get_criterion(self):
        if self.strategy == 'FAST':
            per_cls_weights = None
            self.criterion = nn.CrossEntropyLoss(weight=per_cls_weights,
                                                 reduction='none').cuda(
                                                     self.cfg.gpu)
        else:
            raise ValueError("[Warning] Strategy is not supported !")

    def train_one_epoch(self,**kwargs):
        # Record
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # for confusion matrix
        all_preds = list()
        all_targets = list()

        # switch to train mode
        self.model.train()

        for i, (_input, target) in enumerate(self.train_loader):

            if self.cfg.gpu is not None:
                _input = _input.cuda(self.cfg.gpu, non_blocking=True)
                target = target.cuda(self.cfg.gpu, non_blocking=True)

            head_cls = torch.nonzero(target <= self.cfg.head_threshold)
            head_cls = torch.squeeze(head_cls,dim = 1)
        
            tail_cls = torch.nonzero(target > self.cfg.head_threshold)
            tail_cls = torch.squeeze(tail_cls,dim = 1)

            p = np.random.rand(1)

            lam_x = self.cfg.lam 
            if p < self.cfg.p and len(tail_cls) > 0 and len(head_cls) >= len(tail_cls): 

                tail_head_cls = torch.LongTensor(random.sample(list(head_cls), len(tail_cls)))

                # head loss
                out_head, _ = self.model(_input[head_cls])
                loss_head = self.criterion(out_head, target[head_cls]).mean()
                _, pred_head = torch.max(out_head, 1)
                input2 = _input[tail_head_cls]

                out, _ = self.model(_input[tail_cls],input2= input2,layer = self.cfg.layers,lam_image = 1,dim = self.cfg.dim)

                loss_tail = FAST_criterion(self.criterion, out, target[tail_cls],
                                   target[tail_head_cls], lam=lam_x).mean()
                _, pred_tail = torch.max(out, 1)


                # all 
                acc1, acc5 = accuracy(torch.cat([out_head,out],dim=0), torch.cat([target[head_cls],target[tail_cls]],dim=0), topk=(1, 5))

                # acc1, acc5 = accuracy(out, target, topk=(1, 5))
                all_preds.extend(torch.cat([pred_head,pred_tail],dim=0).cpu().numpy())
                all_targets.extend(torch.cat([target[head_cls],target[tail_cls]],dim=0).cpu().numpy())

                l = 0.5  # fix


                loss = l * loss_head + (1-l)*loss_tail 

            else:
                rand_index = torch.randperm(_input.size()[0]).cuda()
                target_a = target
                target_b = target[rand_index]
                # compute output
                out, _ = self.model(_input,input2= _input[rand_index],layer = self.cfg.layers,lam_image = 1,dim = self.cfg.dim)

                loss = FAST_criterion(self.criterion, out, target_a,
                                   target_b, lam=lam_x).mean()

                acc1, acc5 = accuracy(out, target_a, topk=(1, 5))

            # measure accuracy and record loss
            losses.update(loss.item(), _input.size(0))
            top1.update(acc1[0], _input.size(0))
            top5.update(acc5[0], _input.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.cfg.print_freq == 0:
                output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.6f}\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                              self.epoch,
                              i,
                              len(self.train_loader),
                              loss=losses,
                              top1=top1,
                              top5=top5,
                              lr=self.optimizer.param_groups[-1]['lr'] * 0.1))
                self.log_training.write(output + '\n')
                self.log_training.flush()


        self.compute_metrics_and_record(all_preds,
                                        all_targets,
                                        losses,
                                        top1,
                                        top5,
                                        flag='Training')

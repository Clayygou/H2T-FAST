import numpy as np
import torch
import torch.nn as nn
from .trainer import Trainer
import random
import math

from imbalanceddl.utils.utils import AverageMeter
from imbalanceddl.utils.metrics import accuracy

from imbalanceddl.loss import LDAMLoss


def FAST_criterion_cut(criterion, pred, y_a, y_b, lam,cutmix_lam=1):
    return (lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)) * cutmix_lam +  (1-cutmix_lam) * criterion(pred, y_b)
def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

class FASTLDAMDRWTrainer_cut(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_epoch = 0

    def get_criterion(self):
        if self.strategy == 'FAST_LDAM_DRW_cut':
            if self.cfg.epochs == 300:
                idx = self.epoch // 250
            else:
                idx = self.epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], self.cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(
                self.cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(
                self.cfg.gpu)

            print("=> LDAM Loss with Per Class Weight = {}".format(
                per_cls_weights))
            self.criterion = LDAMLoss(cls_num_list=self.cfg.cls_num_list,
                                      max_m=0.5,
                                      s=30,
                                      weight=per_cls_weights).cuda(
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

            if p < 0.7:
                beta = 1.
                cutmix_lam = np.random.beta(beta, beta)
                if p < self.cfg.p and len(tail_cls) > 0 and len(head_cls) >= len(tail_cls):

                    tail_head_cls = torch.LongTensor(random.sample(list(head_cls), len(tail_cls)))

                    with torch.no_grad():
                        bbx1, bby1, bbx2, bby2 = rand_bbox(_input[tail_cls].size(), cutmix_lam)
                        _input[tail_cls][:, :, bbx1:bbx2, bby1:bby2] = _input[tail_head_cls][:, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    cutmix_lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (_input[tail_cls].size()[-1] * _input[tail_cls].size()[-2]))
                    target_for_acc = target[tail_cls] if cutmix_lam >= 0.5 else target[tail_head_cls]

                    # head loss
                    out_head, _ = self.model(_input[head_cls])
                    loss_head = self.criterion(out_head, target[head_cls]).mean()
                    _, pred_head = torch.max(out_head, 1)

                    # tail loss
                    lam_x = self.cfg.lam 

                    input2 = _input[tail_head_cls]

                    out, _ = self.model(_input[tail_cls],input2= input2,layer = self.cfg.layers,lam_image = 1,dim = self.cfg.dim)
                    loss_tail = FAST_criterion_cut(self.criterion, out, target[tail_cls],
                                       target[tail_head_cls], lam=lam_x, cutmix_lam= cutmix_lam).mean()
                    _, pred_tail = torch.max(out, 1)


                    # all 
                    acc1, acc5 = accuracy(torch.cat([out_head,out],dim=0), torch.cat([target[head_cls],target_for_acc],dim=0), topk=(1, 5))

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
                    bbx1, bby1, bbx2, bby2 = rand_bbox(_input.size(), cutmix_lam)
                    _input[:, :, bbx1:bbx2, bby1:bby2] = _input[rand_index, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    cutmix_lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (_input.size()[-1] * _input.size()[-2]))
                    # compute output
                    output,_ = self.model(_input)
                    loss = self.criterion(output, target_a).mean() * cutmix_lam + self.criterion(output, target_b).mean() * (1. - cutmix_lam)

                    acc1, acc5 = accuracy(output, target_a, topk=(1, 5))

                    _, pred = torch.max(output, 1)
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
            else:
                if p < self.cfg.p and len(tail_cls) > 0 and len(head_cls) >= len(tail_cls):

                    tail_head_cls = torch.LongTensor(random.sample(list(head_cls), len(tail_cls)))

                    # head loss
                    out_head, _ = self.model(_input[head_cls])
                    loss_head = self.criterion(out_head, target[head_cls]).mean()
                    _, pred_head = torch.max(out_head, 1)

                    # tail loss
                    lam_x = self.cfg.lam 
                    input2 = _input[tail_head_cls]

                    out, _ = self.model(_input[tail_cls],input2= input2,layer = self.cfg.layers,lam_image = 1,dim = self.cfg.dim)

                    loss_tail = FAST_criterion_cut(self.criterion, out, target[tail_cls],
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
                    out, _ = self.model(_input)
                    loss = self.criterion(out, target).mean()
                    acc1, acc5 = accuracy(out, target, topk=(1, 5))
                    _, pred = torch.max(out, 1)
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
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

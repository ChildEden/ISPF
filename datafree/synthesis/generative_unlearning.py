import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook
from datafree.criterions import kldiv

class UNGenerativeSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, img_size, iterations=1,
                 lr_g=1e-3, synthesis_batch_size=128, sample_batch_size=128, 
                 adv=0, bn=0, oh=0, act=0, balance=0, criterion=None,
                 normalizer=None, device='cpu',
                 unlearning_args=None,
                 # TODO: FP16 and distributed training 
                 autocast=None, use_fp16=False, distributed=False):
        super(UNGenerativeSynthesizer, self).__init__(teacher, student)
        assert len(img_size)==3, "image size should be a 3-dimension tuple"
        self.img_size = img_size 
        self.iterations = iterations
        self.nz = nz
        if criterion is None:
            criterion = kldiv
        self.criterion = criterion
        self.normalizer = normalizer
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size

        # scaling factors
        self.lr_g = lr_g
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.balance = balance
        self.act = act

        # generator
        self.generator = generator.to(device).train()
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.5,0.999))
        self.distributed = distributed
        self.use_fp16 = use_fp16
        self.autocast = autocast # for FP16
        self.device = device

        self.unlearning_args = copy.deepcopy(unlearning_args)
        if "None" in self.unlearning_args['strategy']:
            self.unlearning_args['unlearn_classes'] = []
        # hooks for deepinversion regularization
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append( DeepInversionHook(m) )

    def synthesize(self):
        unlearning_args = self.unlearning_args
        self.student.eval()
        self.generator.train()
        self.teacher.eval()
        for it in range(self.iterations):
            self.optimizer.zero_grad()
            z = torch.randn( size=(self.synthesis_batch_size, self.nz), device=self.device )
            inputs = self.generator(z)
            inputs = self.normalizer(inputs)
            t_out, t_feat = self.teacher(inputs, return_features=True)
            generated_labels = t_out.max(1)[1]
            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy( t_out, t_out.max(1)[1] )
            loss_act = - t_feat.abs().mean()
            if self.adv>0:
                s_out = self.student(inputs)
                if 'IS' in unlearning_args['strategy']:
                    kld_mt = F.kl_div( F.log_softmax(s_out, dim=1), F.softmax(t_out, dim=1), reduction='none' )
                    kld_mt = -1 * kld_mt
                    kld_mt[:, unlearning_args['unlearn_classes']] = -1 * kld_mt[:, unlearning_args['unlearn_classes']]
                    loss_adv = kld_mt.sum() / s_out.size(0)
                else:
                    loss_adv = -self.criterion(s_out, t_out)
            else:
                loss_adv = loss_oh.new_zeros(1)
            p = F.softmax(t_out, dim=1).mean(0)
            loss_balance = (p * torch.log(p)).sum() # maximization

            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv + self.balance * loss_balance + self.act * loss_act
            loss.backward()
            self.optimizer.step()

        losses = {
            'bn': loss_bn.item(),
            'oh': loss_oh.item(),
            'adv': loss_adv.item(),
            'balance': loss_balance.item(),
            'act': loss_act.item(),
        }

        return {
            'synthetic': self.normalizer(inputs.detach(), reverse=True),
            'loss': loss.item(),
            'generated_labels': generated_labels,
            'losses': losses,
        }
    
    @torch.no_grad()
    def sample(self):
        self.generator.eval()
        z = torch.randn( size=(self.sample_batch_size, self.nz), device=self.device )
        inputs = self.normalizer(self.generator(z))
        return inputs
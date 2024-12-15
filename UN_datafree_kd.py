import argparse
from math import gamma
import os
import copy
import random
import shutil
import time
import warnings

import registry
import datafree

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from unlearning_utils.logger import Logger
from unlearning_utils.metric import AggregateScalar, test_all_in_one
import unlearning_utils.dfkd_unlearning_strategies as unlearning_strategies
from tqdm import tqdm

def entropy(logits):
    prob = F.softmax(logits, dim=1)
    log_prob = F.log_softmax(logits, dim=1)
    return - (prob * log_prob).sum(dim=1).sum()

parser = argparse.ArgumentParser(description='Data-free Knowledge Distillation Unlearning')



# Data Free
parser.add_argument('--method', required=True, choices=['zskt', 'dfq'])
parser.add_argument('--adv', default=0, type=float, help='scaling factor for adversarial distillation')
parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
parser.add_argument('--oh', default=0, type=float, help='scaling factor for one hot loss (cross entropy)')
parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
parser.add_argument('--balance', default=0, type=float, help='scaling factor for class balance')

parser.add_argument('--warmup', default=0, type=int, metavar='N',
                    help='which epoch to start kd')

# Basic
parser.add_argument('--data_root', default='~/storage/public_datasets')
parser.add_argument('--teacher', default='allcnn')
parser.add_argument('--student', default='allcnn')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate for KD')
parser.add_argument('--lr_decay_milestones', default="120,150,180", type=str,
                    help='milestones for learning rate decay')

parser.add_argument('--lr_g', default=1e-3, type=float, 
                    help='initial learning rate for generation')
parser.add_argument('--T', default=1, type=float)

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--g_steps', default=1, type=int, metavar='N',
                    help='number of iterations for generation')
parser.add_argument('--kd_steps', default=400, type=int, metavar='N',
                    help='number of iterations for KD after generation')
parser.add_argument('--ep_steps', default=400, type=int, metavar='N',
                    help='number of total iterations in each epoch')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate_only', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--synthesis_batch_size', default=None, type=int,
                    metavar='N',
                    help='mini-batch size (default: None) for synthesis, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# Device
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--fp16', action='store_true',
                    help='use fp16')

# Misc
parser.add_argument('--seed', default=1334, type=int,
                    help='seed for initializing training.')
parser.add_argument('--log_tag', default='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--trial', default=0, type=int)
parser.add_argument('--uncls', default=None, type=int)
parser.add_argument('--strategy', default='None', type=str)
parser.add_argument('--multi_uncls', default=None, type=str)
best_acc1 = 0


def main():
    args = parser.parse_args()
    args.seed += args.trial

    args.epochs += args.warmup
    if args.multi_uncls is None:
        unlearning_args = {
            'strategy': args.strategy,
            'unlearn_classes': [args.uncls],
            'threshold': 0.01,
            'log_freq': 1
        }
    else:
        unlearning_args = {
            'strategy': args.strategy,
            'unlearn_classes': [int(cls) for cls in args.multi_uncls.split(',')],
            'threshold': 0.01,
            'log_freq': 1
        }
    if ('cifar100' in args.dataset) or ('tiny_imagenet' in args.dataset):
        unlearning_args['threshold'] = 0.001
    print(unlearning_args)

    if args.seed is not None:
        from unlearning_utils.seed import set_seed
        torch.backends.cudnn.benchmark = True
        set_seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Simply call main_worker function
    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args, unlearning_args)


def main_worker(gpu, ngpus_per_node, args, unlearning_args):

    ############### IMPLAIN for UNLEARNING ###############
    log_freq = unlearning_args['log_freq']
    original_path = './runs/training_history/original_model/%s/%s/trial_%s/' % (args.dataset, args.teacher, args.trial)
    if 'None' in unlearning_args['strategy']:
        log_path = './runs/training_history/DFKD/%s/%s/%s/trial_%s/' % (args.method, args.dataset, args.teacher, args.trial)
    else:
        log_path = './runs/training_history/DFKD_UN/%s/%s/%s/%s/trial_%s/uncls_%s/' % (args.method, args.dataset, unlearning_args['strategy'], args.teacher, args.trial, str(unlearning_args['unlearn_classes']))
    os.makedirs(log_path, exist_ok=True)

    tb_logger = Logger(log_dir=log_path)
    args.tb_logger = tb_logger

    # Prepare statistics storage
    running_data_time, running_batch_time = AggregateScalar(), AggregateScalar()
    running_student_maxes_avg, running_teacher_maxes_avg = AggregateScalar(), AggregateScalar()
    running_student_total_loss, running_generator_total_loss = AggregateScalar(), AggregateScalar()
    running_uncls_prob_teacher, running_uncls_prob_student = AggregateScalar(), AggregateScalar()
    running_batch_entropy, running_retain_batch_entropy = AggregateScalar(), AggregateScalar()
    student_maxes_distribution, student_argmaxes_distribution = [], []
    teacher_maxes_distribution, teacher_argmaxes_distribution = [], []
    generated_labels_distribution = []
    ######################################################

    global best_acc1
    args.gpu = gpu
    ############################################
    # GPU and FP16
    ############################################
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.fp16:
        from torch.cuda.amp import autocast, GradScaler
        args.scaler = GradScaler() if args.fp16 else None 
        args.autocast = autocast
    else:
        args.autocast = datafree.utils.dummy_ctx


    ############################################
    # Logger
    ############################################
    if args.log_tag != '':
        args.log_tag = '-'+args.log_tag
    log_name = '%s-%s-%s'%(args.dataset, args.teacher, args.student)
    args.logger = datafree.utils.logger.get_logger(log_name, output=f'{log_path}log.txt')

    for k, v in datafree.utils.flatten_dict( vars(args) ).items(): # print args
        args.logger.info( "%s: %s"%(k,v) )

    ############################################
    # Setup dataset
    ############################################
    num_classes, ori_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
    unlearning_args['num_classes'] = num_classes
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    evaluator = datafree.evaluators.classification_evaluator(val_loader)

    ############################################
    # Setup models
    ############################################
    def prepare_model(model):
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
            return model
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            return model
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()
            return model
    student = registry.get_model(args.student, num_classes=num_classes)
    teacher = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
    args.normalizer = normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    teacher.load_state_dict(torch.load(f'{original_path}best_ckpt.pth', map_location='cpu')['state_dict'])
    student = prepare_model(student)
    teacher = prepare_model(teacher)
    criterion = datafree.criterions.KLDiv(T=args.T)
    
    ############################################
    # Setup data-free synthesizers
    ############################################
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size
    
    if args.method in ['zskt', 'dfq']:
        nz = 512 if args.method=='dafl' else 256
        generator = datafree.models.generator.LargeGenerator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        criterion = torch.nn.L1Loss() if args.method=='dfad' else datafree.criterions.KLDiv()
        synthesizer = datafree.synthesis.UNGenerativeSynthesizer(
                 teacher=teacher, student=student, generator=generator, nz=nz, 
                 img_size=(3, 32, 32), iterations=args.g_steps, lr_g=args.lr_g,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, act=args.act, balance=args.balance, criterion=criterion,
                 normalizer=args.normalizer, device=args.gpu, unlearning_args=unlearning_args)

    else: raise NotImplementedError
        
    ############################################
    # Setup optimizer
    ############################################
    optimizer = torch.optim.SGD(student.parameters(), args.lr, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=200, eta_min=2e-4)

    ############################################
    # Resume
    ############################################
    args.current_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            if isinstance(student, nn.Module):
                student.load_state_dict(checkpoint['state_dict'])
            else:
                student.module.load_state_dict(checkpoint['state_dict'])
            best_acc1 = checkpoint['best_acc1']
            try: 
                args.start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            except: print("Fails to load additional model information")
            print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                  .format(args.resume, checkpoint['epoch'], best_acc1))
        else:
            print("[!] no checkpoint found at '{}'".format(args.resume))
        
    ############################################
    # Evaluate
    ############################################
    if args.evaluate_only:
        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        print('[Eval] Acc={acc:.4f}'.format(acc=eval_results['Acc']))
        return

    ############################################
    # Train Loop
    ############################################
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        args.current_epoch=epoch

        for _ in range( args.ep_steps//args.kd_steps ): # total kd_steps < ep_steps
            # 1. Data synthesis
            G_results = synthesizer.synthesize() # g_steps
            # 2. Knowledge distillation
            if epoch >= args.warmup:
                teacher_logits, student_logits, student_loss, is_skip_student_step = train( synthesizer, [student, teacher], criterion, optimizer, args, unlearning_args) # # kd_steps

        if epoch < args.warmup or is_skip_student_step:
            continue
        # Logging at the end of each epoch
        with torch.no_grad():
            uncls_prob_teacher = torch.softmax(teacher_logits, dim=1)[:, unlearning_args['unlearn_classes']].mean(dim=1)
            uncls_prob_student = torch.softmax(student_logits, dim=1)[:, unlearning_args['unlearn_classes']].mean(dim=1)
            running_batch_entropy.update(entropy(student_logits))
            running_retain_batch_entropy.update(entropy(teacher_logits[:, [i for i in range(num_classes) if i not in unlearning_args['unlearn_classes']]]))
            running_uncls_prob_teacher.update(float(uncls_prob_teacher.mean()))
            running_uncls_prob_student.update(float(uncls_prob_student.mean()))
            teacher_maxes, teacher_argmaxes = torch.max(torch.softmax(teacher_logits, dim=1), dim=1)
            student_maxes, student_argmaxes = torch.max(torch.softmax(student_logits, dim=1), dim=1)
            running_generator_total_loss.update(float(G_results['loss']))
            running_student_total_loss.update(float(student_loss.item()))
            running_teacher_maxes_avg.update(float(torch.mean(teacher_maxes)))
            running_student_maxes_avg.update(float(torch.mean(student_maxes)))
            teacher_maxes_distribution.append(teacher_maxes)
            teacher_argmaxes_distribution.append(teacher_argmaxes)
            student_maxes_distribution.append(student_maxes)
            student_argmaxes_distribution.append(student_argmaxes)
            generated_labels_distribution.append(G_results['generated_labels'])

        if epoch % log_freq == 0:
            test_acc = test_all_in_one(student, val_loader, unlearning_args['unlearn_classes'], device='cuda')
            unlearn_acc, retain_acc, overal_acc = test_acc['unlearn_acc'], test_acc['retain_acc'], test_acc['overall_acc']

            with torch.no_grad():
                args.tb_logger.scalar_summary('TRAIN_PSEUDO/generator_loss', running_generator_total_loss.avg(), epoch)
                args.tb_logger.scalar_summary('TRAIN_PSEUDO/student_loss', running_student_total_loss.avg(), epoch)
                args.tb_logger.scalar_summary('TRAIN_PSEUDO/teacher_maxes_avg', running_teacher_maxes_avg.avg(), epoch)
                args.tb_logger.scalar_summary('TRAIN_PSEUDO/student_maxes_avg', running_student_maxes_avg.avg(), epoch)
                args.tb_logger.scalar_summary('TRAIN_PSEUDO/uncls_prob_teacher', running_uncls_prob_teacher.avg(), epoch)
                args.tb_logger.scalar_summary('TRAIN_PSEUDO/uncls_prob_student', running_uncls_prob_student.avg(), epoch)
                args.tb_logger.scalar_summary('TRAIN_PSEUDO/batch_entropy', running_batch_entropy.avg(), epoch)
                args.tb_logger.scalar_summary('TRAIN_PSEUDO/retain_batch_entropy', running_retain_batch_entropy.avg(), epoch)
                args.tb_logger.scalar_summary('TRAIN_PSEUDO/student_lr', scheduler.get_lr()[0], epoch)
                args.tb_logger.scalar_summary('EVALUATE/test_unlearn_acc', unlearn_acc*100, epoch)
                args.tb_logger.scalar_summary('EVALUATE/test_retain_acc', retain_acc*100, epoch)
                args.tb_logger.scalar_summary('EVALUATE/test_overall_acc', overal_acc*100, epoch)
                args.tb_logger.scalar_summary('TRAIN_PSEUDO/sample_count', len(teacher_argmaxes), epoch)
                for k, v in G_results['losses'].items():
                    args.tb_logger.scalar_summary(f'GENERATOR_LOSSES/{k}', v, epoch)
                
                args.tb_logger.image_summary('GENERATED_IMAGES', datafree.utils.vis_images(G_results['synthetic'], 9), epoch)
                args.tb_logger.histo_summary('TEACHER_MAXES_DISTRIBUTION', torch.cat(teacher_maxes_distribution), epoch)
                args.tb_logger.histo_summary('TEACHER_ARGMAXES_DISTRIBUTION', torch.cat(teacher_argmaxes_distribution), epoch, bins=num_classes)
                args.tb_logger.histo_summary('STUDENT_MAXES_DISTRIBUTION', torch.cat(student_maxes_distribution), epoch)
                args.tb_logger.histo_summary('STUDENT_ARGMAXES_DISTRIBUTION', torch.cat(student_argmaxes_distribution), epoch, bins=num_classes)
                args.tb_logger.histo_summary('GENERATED_LABELS', torch.cat(generated_labels_distribution), epoch, bins=num_classes)


                teacher_probs_mean = torch.softmax(teacher_logits, dim=1).mean(dim=0).cpu().numpy()
                student_probs_mean = torch.softmax(student_logits, dim=1).mean(dim=0).cpu().numpy()
                args.tb_logger.vector_record('TEACHER_PROBS', teacher_probs_mean, epoch)
                args.tb_logger.vector_to_csv('teacher_pobs.csv')
                args.tb_logger.writer.flush()
                args.tb_logger.vector_record('STUDENT_PROBS', student_probs_mean, epoch)
                args.tb_logger.vector_to_csv('student_pobs.csv')
                args.tb_logger.writer.flush()

                args.tb_logger.histogram_record('SAMPLE_CLASSES_DIST', torch.cat(teacher_argmaxes_distribution).detach().cpu().numpy(), epoch, bins=num_classes)
                args.tb_logger.histo_to_csv('histo.csv')
                args.tb_logger.writer.flush()

                args.tb_logger.write_to_csv('train_test.csv')
                args.tb_logger.writer.flush()

                running_data_time.reset(), running_batch_time.reset()
                running_teacher_maxes_avg.reset(), running_student_maxes_avg.reset()
                running_generator_total_loss.reset(), running_student_total_loss.reset(),
                running_uncls_prob_teacher.reset(), running_uncls_prob_student.reset()
                running_batch_entropy.reset(), running_retain_batch_entropy.reset()
                teacher_maxes_distribution, teacher_argmaxes_distribution = [], []
                student_maxes_distribution, student_argmaxes_distribution = [], []
                generated_labels_distribution = []

            if 'None' in unlearning_args['strategy']:
                acc1 = overal_acc
            else:
                acc1 = retain_acc
            args.logger.info('[Eval] Epoch={current_epoch} Acc@1={acc1:.4f} Lr={lr:.4f}'
                    .format(current_epoch=args.current_epoch, acc1=acc1, lr=optimizer.param_groups[0]['lr']))
            scheduler.step()
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            _best_ckpt = f'{log_path}best_ckpt.pth'
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.student,
                'generator_state_dict': synthesizer.generator.state_dict(),
                'state_dict': student.state_dict(),
                'best_acc1': float(best_acc1),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, _best_ckpt)
    args.logger.info("Best: %.4f"%best_acc1)


def train(synthesizer, model, criterion, optimizer, args, unlearning_args):
    loss_metric = datafree.metrics.RunningLoss(datafree.criterions.KLDiv(reduction='sum'))
    acc_metric = datafree.metrics.TopkAccuracy(topk=(1,5))
    student, teacher = model
    optimizer = optimizer
    student.train()
    teacher.eval()
    for i in range(args.kd_steps):
        images = []
        masks = []
        sampled_count = 0

        while True:
            images_ = synthesizer.sample()
            with args.autocast():
                with torch.no_grad():
                    images_ = images_.cuda(args.gpu, non_blocking=True)
                    t_out_, t_feat_ = teacher(images_, return_features=True)
                if 'BlockF' in unlearning_args['strategy']:
                    masks_ = unlearning_strategies.block_unlearn_classes(t_out_, unlearning_args['unlearn_classes'])
                elif 'GKT' in unlearning_args['strategy']:
                    masks_ = unlearning_strategies.filter_unlearn_classes(t_out_, unlearning_args['unlearn_classes'], unlearning_args['threshold'])
                
                else:
                    masks_ = torch.ones(images_.size(0), dtype=torch.bool)

                images.append(images_[masks_])
                masks.append(masks_)
                sampled_count += masks_.sum().item()

                if 'repeat' not in unlearning_args['strategy']:
                    images = torch.cat(images, dim=0)
                    masks = torch.cat(masks, dim=0)
                    break

                if sampled_count >= synthesizer.synthesis_batch_size:
                    images = torch.cat(images, dim=0)[:synthesizer.synthesis_batch_size]
                    break
        if len(images) == 0:
            return None, None, None, True
        
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        
        with args.autocast():
            with torch.no_grad():
                t_out, t_feat = teacher(images, return_features=True)
            s_out, s_feat = student(images.detach(), return_features=True)

            if 'PF' in unlearning_args['strategy']:
                t_out = t_out.detach()
                t_prob = torch.softmax(t_out, dim=1)
                retaining_logits_idx = torch.ones_like(t_out, dtype=torch.bool)
                unlearn_logits_idx = torch.zeros_like(t_out, dtype=torch.bool)
                for i in unlearning_args['unlearn_classes']:
                    retaining_logits_idx[:, i] = False
                    unlearn_logits_idx[:, i] = True
                need_to_be_redist_idx = t_prob > unlearning_args['threshold']
                need_to_be_redist_idx = need_to_be_redist_idx & unlearn_logits_idx
                need_to_be_redist_logits = torch.zeros_like(t_out)
                need_to_be_redist_logits[need_to_be_redist_idx] = t_out[need_to_be_redist_idx]
                min_logits = torch.min(t_out, dim=1)[0]
                min_logits_matrix = min_logits.unsqueeze(1).repeat(1, t_out.size(1))
                min_logits_matrix[~need_to_be_redist_idx] = 0
                need_to_be_redist_logits = need_to_be_redist_logits - min_logits_matrix
                delta_logits = need_to_be_redist_logits.sum(dim=1)
                redist_logits = delta_logits / (unlearning_args['num_classes'] - len(unlearning_args['unlearn_classes']))
                redist_logits = redist_logits.unsqueeze(1).repeat(1, t_out.size(1))
                t_out[retaining_logits_idx] += redist_logits[retaining_logits_idx]
                t_out[need_to_be_redist_idx] = min_logits_matrix[need_to_be_redist_idx]
                loss_s = criterion(s_out, t_out)
            else:
                loss_s = criterion(s_out, t_out.detach())

        optimizer.zero_grad()
        if args.fp16:
            scaler_s = args.scaler_s
            scaler_s.scale(loss_s).backward()
            scaler_s.step(optimizer)
            scaler_s.update()
        else:
            loss_s.backward()
            optimizer.step()
        acc_metric.update(s_out, t_out.max(1)[1])
        loss_metric.update(s_out, t_out)
        if args.print_freq>0 and i % args.print_freq == 0:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            args.logger.info('[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'
              .format(current_epoch=args.current_epoch, i=i, total_iters=len(args.kd_steps), train_acc1=train_acc1, train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset()

    return t_out, s_out, loss_s, False
    
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)

if __name__ == '__main__':
    main()

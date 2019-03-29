from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import argparse
import os
import shutil
import time
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from shuffleNetV2 import ShuffleNetV2
from utils import measure_model

parser = argparse.ArgumentParser(description='PyTorch ShuffleNetV2 ImageNet')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='shufflenetv2', type=str, metavar='M',
                    help='model to train the dataset (default: shufflenetv2)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=240, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.5, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--lr-type', default='linear', type=str, metavar='T',
                    help='learning rate strategy (default: cosine)',
                    choices=['cosine', 'multistep', 'linear'])
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model (default: false)')
parser.add_argument('--no-save-model', dest='no_save_model', action='store_true',
                    help='only save best model (default: false)')
parser.add_argument('--manual-seed', default=0, type=int, metavar='N',
                    help='manual seed (default: 0)')
parser.add_argument('--gpu',
                    help='gpu available')
parser.add_argument('--savedir', type=str, metavar='PATH', default='results/savedir',
                    help='path to save result and checkpoint (default: results/savedir)')
parser.add_argument('--resume', action='store_true',
                    help='use latest checkpoint if have any (default: none)')
parser.add_argument('--symbol', default=1, type=int, metavar='DIFFERENT SHUFFLENETV2',
                    help='different settings of shufflenetv2 (default: 1)')
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on validation set (default: false)')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu    
warnings.filterwarnings("ignore")
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

best_prec1 = 0

def main():
    global args
    ### Create model and calculate FLOPs & Param and delete model
    model = ShuffleNetV2(args.symbol)   
    print(model)
    IMAGE_SIZE = 224
    n_flops, n_params = measure_model(model, IMAGE_SIZE, IMAGE_SIZE)
    print('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))
    args.filename = "%s_%s_%s.txt" % (args.model, int(n_params), int(n_flops))
    print(args)
    del(model)
    ### Create model again
    model = ShuffleNetV2(args.symbol)
    # print(model)
    # args.filename = "%s.txt" % (args.model)
    # print(args)
    model = torch.nn.DataParallel(model).cuda()
    ### Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    ### Optionally resume from a checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    cudnn.benchmark = True
    ### Data loading 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if not args.evaluate:
        traindir = os.path.join(args.data, 'train')
        train_set = datasets.ImageFolder(traindir, transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
                                                                       transforms.RandomHorizontalFlip(),
                                                                       transforms.ToTensor(), normalize]))
    valdir = os.path.join(args.data, 'val')
    val_set = datasets.ImageFolder(valdir, transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224),
                                                               transforms.ToTensor(), normalize]))    
    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    ### Evaluate
    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    
    ### Train
    for epoch in range(args.start_epoch, args.epochs):
        ### Train for one epoch
        tr_prec1, tr_prec5, loss, lr = train(train_loader, model, criterion, optimizer, epoch)
        ### Evaluate on validation set
        val_prec1, val_prec5 = validate(val_loader, model, criterion)
        ### Remember best prec@1 and save checkpoint
        is_best = val_prec1 < best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        ### Save checkpoint
        model_filename = 'checkpoint_%03d.pth.tar' % epoch
        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'best_prec1': best_prec1, 'optimizer': optimizer.state_dict()}, 
                        args, is_best, model_filename, "%.4f %.4f %.4f %.4f %.4f %.4f\n" % (val_prec1, val_prec5, tr_prec1, tr_prec5, loss, lr))
        # save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, 
        #                 args, model_filename, "%.4f %.4f %.4f %.4f\n" % (tr_prec1, tr_prec5, loss, lr))

    n_flops, n_params = measure_model(model, IMAGE_SIZE, IMAGE_SIZE)
    print('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))
    return


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    ### Switch to train mode
    model.train()
    running_lr = None
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        ### Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i, nBatch=len(train_loader), method=args.lr_type)
        if running_lr is None:
            running_lr = lr
        ### Measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        ### Compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        ### Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        ### Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f}\t'  # ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f}\t'  # ({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f}\t'  # ({loss.avg:.4f}) '
                  'Prec@1 {top1.val:.3f}\t'  # ({top1.avg:.3f}) '
                  'Prec@5 {top5.val:.3f}\t'  # ({top5.avg:.3f})'
                  'lr {lr: .4f}'.format(epoch, i, len(train_loader), batch_time=batch_time,
                                        data_time=data_time, loss=losses, top1=top1, top5=top5, lr=lr))
    return 100. - top1.avg, 100. - top5.avg, losses.avg, running_lr


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    ### Switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        ### Compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        ### Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(val_loader), batch_time=batch_time, 
                                                                  loss=losses, top1=top1, top5=top5))
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return 100. - top1.avg, 100. - top5.avg


def load_checkpoint(args):
    model_dir = os.path.join(args.savedir, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0].strip()
    else:
        return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state


def save_checkpoint(state, args, is_best, filename, result):
    print(args)
    result_filename = os.path.join(args.savedir, args.filename)
    model_dir = os.path.join(args.savedir, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))
    with open(result_filename, 'a') as fout:
        fout.write(result)
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if is_best:
        shutil.copyfile(model_filename, best_filename)
    print("=> saved checkpoint '{}'".format(model_filename))
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args, batch=None, nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data in ['cifar10', 'cifar100']:
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate**2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            lr = args.lr * (0.1 ** (epoch // 30))
    elif method == 'linear':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = args.lr * (1 - (T_cur / T_total))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

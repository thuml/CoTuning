import argparse
import os
from time import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from module.backbone import ResNet50_F, ResNet50_C
from module.calibrated_concept_mapping import concept_mapping
from utils.transforms import get_transforms
from utils.tools import AccuracyMeter, TenCropsTest


def get_writer(log_dir):
    return SummaryWriter(log_dir)


def get_configs():
    parser = argparse.ArgumentParser(
        description='Pytorch Co-Tuning Training')

    # train
    parser.add_argument('--gpu', default='0', type=int,
                        help='GPU num for training')
    parser.add_argument('--seed', type=int, default=2020)

    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--total_iter', default=9050, type=int)
    parser.add_argument('--eval_iter', default=1000, type=int)
    parser.add_argument('--save_iter', default=9000, type=int)
    parser.add_argument('--print_iter', default=100, type=int)

    # dataset
    parser.add_argument('--data_path', default="/data/finetune",
                        type=str, help='Path of dataset')
    parser.add_argument('--class_num', default=200,
                        type=int, help='number of classes')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Num of workers used in dataloading')

    # optimizer
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate for training')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma value for learning rate decay')
    parser.add_argument('--nesterov', default=True,
                        type=bool, help='nesterov momentum')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optimizer')
    parser.add_argument('--weight_decay', default=5e-4,
                        type=float, help='Weight decay value for optimizer')

    # experiment
    parser.add_argument('--root', default='.', type=str,
                        help='Root of the experiment')
    parser.add_argument('--name', default='StochNorm', type=str,
                        help='Name of the experiment')
    parser.add_argument('--trade_off', default=2.1, type=float,
                        help='Trade off for co-tuning')
    parser.add_argument('--matrix_path', default='matrix.npy', type=str,
                        help='Path of concept matrix')
    parser.add_argument('--save_dir', default="model",
                        type=str, help='Path of saved models')
    parser.add_argument('--visual_dir', default="visual",
                        type=str, help='Path of tensorboard data')

    configs = parser.parse_args()

    return configs


def str2list(v):
    return v.split(',')


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_data_loader(configs):
    data_transforms = get_transforms(resize_size=256, crop_size=224)

    # build dataset
    train_dataset = datasets.ImageFolder(
        os.path.join(configs.data_path, 'train'),
        transform=data_transforms['train'])
    determin_train_dataset = datasets.ImageFolder(
        os.path.join(configs.data_path, 'train'),
        transform=data_transforms['val'])
    val_dataset = datasets.ImageFolder(
        os.path.join(configs.data_path, 'val'),
        transform=data_transforms['val'])
    test_datasets = {
        'test' + str(i):
            datasets.ImageFolder(
                os.path.join(configs.data_path, 'test'),
                transform=data_transforms["test" + str(i)]
        )
        for i in range(10)
    }

    # build dataloader
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True,
                              num_workers=configs.num_workers, pin_memory=True)
    determin_train_loader = DataLoader(determin_train_dataset, batch_size=configs.batch_size, shuffle=False,
                                       num_workers=configs.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                            num_workers=configs.num_workers, pin_memory=True)
    test_loaders = {
        'test' + str(i):
            DataLoader(
                test_datasets["test" + str(i)],
                batch_size=4, shuffle=False, num_workers=configs.num_workers
        )
        for i in range(10)
    }

    return train_loader, determin_train_loader, val_loader, test_loaders


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    configs = get_configs()
    print(configs)
    torch.cuda.set_device(configs.gpu)
    set_seeds(configs.seed)

    train_loader, determin_train_loader, val_loader, test_loaders = get_data_loader(
        configs)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.f_net = ResNet50_F(pretrained=True)
            self.c_net_1 = ResNet50_C(pretrained=True)
            self.c_net_2 = nn.Linear(self.f_net.output_dim, configs.class_num)
            self.c_net_2.weight.data.normal_(0, 0.01)
            self.c_net_2.bias.data.fill_(0.0)

        def forward(self, x):
            feature = self.f_net(x)
            out_1 = self.c_net_1(feature)
            out_2 = self.c_net_2(feature)

            return out_1, out_2

    net = Net().cuda()

    if os.path.exists(configs.matrix_path):
        print('load pre-computed matrix from {}.'.format(configs.matrix_path))
        matrix = np.load(configs.matrix_path)
    else:
        matrix_path = 'matrix.npy'
        if os.path.exists(matrix_path):
            print('load pre-computed matrix')
            matrix = np.load(matrix_path)
        else:
            print('computing matrix')

            def get_feature(loader):
                idx = 0

                for train_inputs, train_labels in tqdm(loader):
                    net.eval()
                    if idx == 0:
                        all_train_labels = train_labels
                    else:
                        all_train_labels = np.concatenate(
                            [all_train_labels, train_labels], 0)

                    train_inputs, train_labels = train_inputs.cuda(), train_labels.cuda()
                    imagenet_labels, _ = net(train_inputs)
                    imagenet_labels = imagenet_labels.detach().cpu().numpy()

                    if idx == 0:
                        all_imagenet_labels = imagenet_labels
                    else:
                        all_imagenet_labels = np.concatenate(
                            [all_imagenet_labels, imagenet_labels], 0)

                    idx += 1

                return all_imagenet_labels, all_train_labels

            train_imagenet_labels, train_train_labels = get_feature(
                determin_train_loader)
            val_imagenet_labels, val_train_labels = get_feature(val_loader)
            matrix = concept_mapping(train_imagenet_labels, train_train_labels,
                                     val_imagenet_labels, val_train_labels)

            np.save(matrix_path, matrix)

    train(configs, train_loader, val_loader, test_loaders, net, matrix)


def train(configs, train_loader, val_loader, test_loaders, net, matrix):
    train_len = len(train_loader) - 1
    train_iter = iter(train_loader)

    # different learning rates for different layers
    params_list = [{"params": filter(lambda p: p.requires_grad, net.f_net.parameters())},
                   {"params": filter(lambda p: p.requires_grad, net.c_net_1.parameters())},
                   {"params": filter(lambda p: p.requires_grad, net.c_net_2.parameters()), "lr": configs.lr * 10}]

    optimizer = torch.optim.SGD(params_list, lr=configs.lr, weight_decay=configs.weight_decay,
                                momentum=configs.momentum, nesterov=configs.nesterov)
    milestones = [6000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma=configs.gamma)

    # check visual path
    visual_path = os.path.join(configs.visual_dir, configs.name)
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    writer = get_writer(visual_path)

    # check model save path
    save_path = os.path.join(configs.save_dir, configs.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for iter_num in range(configs.total_iter):
        net.train()

        if iter_num % train_len == 0:
            train_iter = iter(train_loader)

        # Data Stage
        data_start = time()

        train_inputs, train_labels = next(train_iter)
        imagenet_targets = torch.from_numpy(matrix[train_labels]).cuda().float()
        train_inputs, train_labels = train_inputs.cuda(), train_labels.cuda()

        data_duration = time() - data_start

        # Calc Stage
        calc_start = time()

        imagenet_outputs, train_outputs = net(train_inputs)

        ce_loss = nn.CrossEntropyLoss()(train_outputs, train_labels)
        imagenet_loss = - imagenet_targets * torch.log(nn.Softmax(dim=-1)(imagenet_outputs) + 1e-10)
        imagenet_loss = torch.mean(torch.sum(imagenet_loss, dim=-1))
        loss = ce_loss + configs.trade_off * imagenet_loss

        writer.add_scalar('loss/ce_loss', ce_loss, iter_num)
        writer.add_scalar('loss/imagenet_loss', imagenet_loss, iter_num)
        writer.add_scalar('loss/loss', loss, iter_num)

        net.zero_grad()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        scheduler.step()

        calc_duration = time() - calc_start

        if iter_num % configs.eval_iter == 0:
            acc_meter = AccuracyMeter(topk=(1,))
            with torch.no_grad():
                net.eval()
                for val_inputs, val_labels in tqdm(val_loader):
                    val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()
                    _, val_outputs = net(val_inputs)
                    acc_meter.update(val_outputs, val_labels)
                writer.add_scalar('acc/val_acc', acc_meter.avg[1], iter_num)
                print(
                    "Iter: {}/{} Val_Acc: {:2f}".format(
                        iter_num, configs.total_iter, acc_meter.avg[1])
                )
            acc_meter.reset()

        if iter_num % configs.save_iter == 0 and iter_num > 0:
            test_acc = TenCropsTest(test_loaders, net)
            writer.add_scalar('acc/test_acc', test_acc, iter_num)
            print(
                "Iter: {}/{} Test_Acc: {:2f}".format(
                    iter_num, configs.total_iter, test_acc)
            )
            checkpoint = {
                'state_dict': net.state_dict(),
                'iter': iter_num,
                'acc': test_acc,
            }
            torch.save(checkpoint,
                       os.path.join(save_path, '{}.pkl'.format(iter_num)))
            print("Model Saved.")

        if iter_num % configs.print_iter == 0:
            print(
                "Iter: {}/{} Loss_main: {:2f}, d/c: {}/{}".format(iter_num, configs.total_iter, loss,
                                                                  data_duration, calc_duration))


if __name__ == '__main__':
    print("PyTorch {}".format(torch.__version__))
    print("TorchVision {}".format(torchvision.__version__))
    main()

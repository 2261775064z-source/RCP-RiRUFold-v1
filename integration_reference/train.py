import os
import os.path as osp
import time
import datetime
from argparse import ArgumentParser

import numpy as np
import torch
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models import get_model
from utils.data import *
from utils.loss import SoftLoULoss
from utils.lr_scheduler import *
from utils.metrics import SegmentationMetricTPFNFP
from utils.my_pd_fa import *
from utils.pd_fa import *
from utils.logger import setup_logger


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of RPCANets')

    #
    # Dataset parameters
    #
    parser.add_argument('--base-size', type=int, default=256, help='base size of images')
    parser.add_argument('--crop-size', type=int, default=256, help='crop size of images')
    parser.add_argument('--dataset', type=str, default='sirst', help='choose datasets')
    parser.add_argument('--data-root', type=str, default='./datasets',
                        help='root directory containing the dataset folders')

    #
    # Training parameters
    #

    parser.add_argument('--batch-size', type=int, default=8, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--warm-up-epochs', type=int, default=0, help='warm up epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='GPU number')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--lr-scheduler', type=str, default='poly', help='learning rate scheduler')

    #
    # Net parameters
    #
    parser.add_argument('--net-name', type=str, default='rpcanet',
                        help='net name: fcn')
    parser.add_argument('--admm-stage-num', type=int, default=5,
                        help='number of stages for rirufold_admm only')
    parser.add_argument('--admm-hidden-channels', type=int, default=24,
                        help='correction width for rirufold_admm only')
    parser.add_argument('--dc-weight', type=float, default=0.05,
                        help='decomposition-consistency loss weight for rirufold_admm')
    parser.add_argument('--bg-weight', type=float, default=0.02,
                        help='target-excluded background loss weight for rirufold_admm')
    parser.add_argument('--prox-weight', type=float, default=0.005,
                        help='bounded-correction deviation weight for rirufold_rcp')
    # Rank parameters
    #
    # parser.add_argument('--rank', type=int, default=8,
    #                     help='rank number')

    #
    # Save parameters
    #
    parser.add_argument('--save-iter-step', type=int, default=1, help='save model per step iters')
    parser.add_argument('--log-per-iter', type=int, default=1, help='interval of logging')
    parser.add_argument('--base-dir', type=str, default='./result/', help='saving dir')
    parser.add_argument('--run-name', type=str, default=None,
                        help='optional deterministic subdirectory under base-dir')

    args = parser.parse_args()

    # Save folders
    args.time_name = time.strftime('%Y%m%dT%H-%M-%S', time.localtime(time.time()))
    args.folder_name = args.run_name or '{}_{}_{}'.format(args.time_name, args.net_name, args.dataset)
    args.save_folder = osp.join(args.base_dir, args.folder_name)

    # seed
    if args.seed != 0:
        set_seeds(args.seed)

    # logger
    args.logger = setup_logger("Robust PCA Networks", args.save_folder, 0, filename='log.txt')
    return args


def set_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.iter_num = 0

        ## dataset
        if args.dataset == 'sirstaug':
            trainset = SirstAugDataset(base_dir=osp.join(args.data_root, 'sirst_aug'),
                                       mode='train', base_size=args.base_size)
            valset = SirstAugDataset(base_dir=osp.join(args.data_root, 'sirst_aug'),
                                     mode='test', base_size=args.base_size)
        elif args.dataset == 'irstd1k':
            trainset = IRSTD1kDataset(base_dir=osp.join(args.data_root, 'IRSTD-1k'), mode='train', base_size=args.base_size)
            valset = IRSTD1kDataset(base_dir=osp.join(args.data_root, 'IRSTD-1k'), mode='test', base_size=args.base_size)

        elif args.dataset == 'nudt':
            trainset = NUDTDataset(base_dir=osp.join(args.data_root, 'NUDT-SIRST'), mode='train', base_size=args.base_size)
            valset = NUDTDataset(base_dir=osp.join(args.data_root, 'NUDT-SIRST'), mode='test', base_size=args.base_size)

        elif args.dataset == 'sirst':
            trainset = SirstDataset(base_dir=osp.join(args.data_root, 'sirst'), mode='train', base_size=args.base_size)
            valset = SirstDataset(base_dir=osp.join(args.data_root, 'sirst'), mode='val', base_size=args.base_size)

        elif args.dataset == 'drive':
            trainset = DriveDatasetTrain(base_dir=r'./datasets/DRIVE', mode='train', base_size=args.base_size, patch_size=args.crop_size)
            valset = DriveDatasetTest(base_dir=r'./datasets/DRIVE', mode='test', base_size=args.base_size)

        elif args.dataset == 'CHASEDB1':
            trainset = CHASEDB1DatasetTrain(base_dir=r'./datasets/CHASEDB1', mode='train', base_size=args.base_size, patch_size=args.crop_size)
            valset = CHASEDB1DatasetTest(base_dir=r'./datasets/CHASEDB1', mode='test', base_size=args.base_size)

        elif args.dataset == 'stare':
            trainset = STAREDatasetTrain(base_dir=r'./datasets/STARE', mode='train', base_size=args.base_size, patch_size=args.crop_size)
            valset = STAREDatasetTest(base_dir=r'./datasets/STARE', mode='test', base_size=args.base_size)
        else:
            raise NotImplementedError

        self.train_data_loader = Data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        self.val_data_loader = Data.DataLoader(valset, batch_size=args.batch_size, shuffle=True)
        self.iter_per_epoch = len(self.train_data_loader)
        self.max_iter = args.epochs * self.iter_per_epoch

        ## GPU
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        # CUDA_VISIBLE_DEVICES remaps the requested physical GPU to cuda:0
        # inside this process. Using cuda:<physical-id> after the remap fails
        # for any nonzero GPU id on typical multi-GPU servers.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ## model
        self.net = get_model(
            args.net_name,
            stage_num=args.admm_stage_num,
            hidden_channels=args.admm_hidden_channels,
        )
        self.net = self.net.to(self.device)
        self.is_rirufold_admm = args.net_name.startswith('rirufold_admm')
        self.is_rirufold_rcp = args.net_name.startswith('rirufold_rcp')

        ## criterion
        self.softiou = SoftLoULoss()
        self.mse = torch.nn.MSELoss()

        ## lr scheduler
        self.scheduler = LR_Scheduler_Head(args.lr_scheduler, args.lr,
                                           args.epochs, len(self.train_data_loader), lr_step=10)

        ## optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)

        ## evaluation metrics
        self.metric = SegmentationMetricTPFNFP(nclass=1)
        self.best_iou = 0
        self.best_fmeasure = 0
        self.eval_loss = 0  # tmp values
        self.iou = 0
        self.fmeasure = 0
        self.eval_my_PD_FA = my_PD_FA()
        self.eval_PD_FA = PD_FA()

        ## SummaryWriter
        self.writer = SummaryWriter(log_dir=args.save_folder)
        self.writer.add_text(args.folder_name, 'Args:%s, ' % args)

        ## log info
        self.logger = args.logger
        self.logger.info(args)
        self.logger.info("Using device: {}".format(self.device))

    def compute_loss(self, data, labels, out_background, out_target, aux=None):
        """Use decomposition-aware losses only for the new ADMM model.

        Legacy models retain their historical MSE supervision so their existing
        experiments and checkpoints remain comparable.
        """
        loss_softiou = self.softiou(out_target, labels)
        if not (self.is_rirufold_admm or self.is_rirufold_rcp):
            loss_reconstruction = self.mse(out_background, data)
            loss_background = torch.zeros((), device=data.device)
            loss_all = loss_softiou + 0.01 * loss_reconstruction
            loss_prox = torch.zeros((), device=data.device)
            return loss_all, loss_softiou, loss_reconstruction, loss_background, loss_prox

        sparse_intensity = aux['sparse_intensity'] if self.is_rirufold_rcp else out_target
        loss_reconstruction = torch.mean(torch.abs(data - out_background - sparse_intensity))
        loss_background = torch.mean(torch.abs((1.0 - labels) * (out_background - data)))
        loss_prox = aux['prox_loss'] if self.is_rirufold_rcp else torch.zeros((), device=data.device)
        loss_all = (
            loss_softiou
            + self.args.dc_weight * loss_reconstruction
            + self.args.bg_weight * loss_background
            + self.args.prox_weight * loss_prox
        )
        return loss_all, loss_softiou, loss_reconstruction, loss_background, loss_prox

    def training(self):
        # training step
        start_time = time.time()
        base_log = "Epoch-Iter: [{:d}/{:d}]-[{:d}/{:d}] || Lr: {:.6f} || Loss: {:.4f}=Seg:{:.4f}+DC:{:.4f}+BG:{:.4f}+Prox:{:.4f} || " \
                   "Cost Time: {} || Estimated Time: {}"
        for epoch in range(args.epochs):
            for i, (data, labels) in enumerate(self.train_data_loader):
                self.net.train()

                self.scheduler(self.optimizer, i, epoch, self.best_iou)

                data = data.to(self.device)

                labels = labels.to(self.device)
                if self.is_rirufold_rcp:
                    out_D, out_T, aux = self.net(data, return_aux=True)
                else:
                    out_D, out_T = self.net(data)
                    aux = None

                loss_all, loss_softiou, loss_dc, loss_bg, loss_prox = self.compute_loss(
                    data, labels, out_D, out_T, aux
                )

                self.optimizer.zero_grad()
                loss_all.backward()
                self.optimizer.step()

                self.iter_num += 1

                cost_string = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                eta_seconds = ((time.time() - start_time) / self.iter_num) * (self.max_iter - self.iter_num)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                self.writer.add_scalar('Train Loss/Loss All', np.mean(loss_all.item()), self.iter_num)
                self.writer.add_scalar('Train Loss/Loss SoftIoU', np.mean(loss_softiou.item()), self.iter_num)
                self.writer.add_scalar('Train Loss/Loss Decomposition', np.mean(loss_dc.item()), self.iter_num)
                self.writer.add_scalar('Train Loss/Loss Background', np.mean(loss_bg.item()), self.iter_num)
                self.writer.add_scalar('Train Loss/Loss Proximal Deviation', np.mean(loss_prox.item()), self.iter_num)
                self.writer.add_scalar('Learning rate/', trainer.optimizer.param_groups[0]['lr'], self.iter_num)

                if self.iter_num % self.args.log_per_iter == 0:
                    self.logger.info(base_log.format(epoch + 1, args.epochs, self.iter_num % self.iter_per_epoch,
                                                     self.iter_per_epoch, self.optimizer.param_groups[0]['lr'],
                                                     loss_all.item(), loss_softiou.item(), loss_dc.item(), loss_bg.item(), loss_prox.item(),
                                                     cost_string, eta_string))

                if (self.iter_num % args.save_iter_step) == 0 or self.iter_num % self.iter_per_epoch == 0:
                    self.validation()

    def validation(self):
        self.metric.reset()
        self.net.eval()
        base_log = "Data: {:s}, IoU: {:.4f}/{:.4f}, F1: {:.4f}/{:.4f} "
        for i, (data, labels) in enumerate(self.val_data_loader):
            with torch.no_grad():
                device_data = data.to(self.device)
                device_labels = labels.to(self.device)
                if self.is_rirufold_rcp:
                    out_D, out_T, aux = self.net(device_data, return_aux=True)
                else:
                    out_D, out_T = self.net(device_data)
                    aux = None
                loss_all, loss_softiou, loss_dc, loss_bg, loss_prox = self.compute_loss(
                    device_data, device_labels, out_D, out_T, aux
                )
            out_D, out_T = out_D.cpu(), out_T.cpu()



            self.metric.update(labels, out_T)


        iou, prec, recall, fmeasure = self.metric.get()
        torch.save(self.net.state_dict(), osp.join(self.args.save_folder, 'latest.pkl'))
        if iou > self.best_iou:
            self.best_iou = iou
            torch.save(self.net.state_dict(), osp.join(self.args.save_folder, 'best.pkl'))
        if fmeasure > self.best_fmeasure:
            self.best_fmeasure = fmeasure


        self.writer.add_scalar('Test/IoU', iou, self.iter_num)
        self.writer.add_scalar('Test/F1', fmeasure, self.iter_num)
        self.writer.add_scalar('Best/IoU', self.best_iou, self.iter_num)
        self.writer.add_scalar('Best/Fmeasure', self.best_fmeasure, self.iter_num)
        self.logger.info(base_log.format(self.args.dataset, iou, self.best_iou, fmeasure, self.best_fmeasure))


if __name__ == '__main__':
    args = parse_args()


    trainer = Trainer(args)
    trainer.training()

    print('Best mIoU: %.5f, Best Fmeasure: %.5f\n\n' % (trainer.best_iou, trainer.best_fmeasure))

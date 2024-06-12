from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as io
import os
import random
import time
import socket
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from torch.autograd import Variable
from datetime import datetime
from metric import ref_evaluate, no_ref_evaluate
from loadwv import prepare_training_data, Dataset_Pro, Dataset_Pro_full
import pywt


# Training settings 24.3.13
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--ChDim', type=int, default=8, help='output channel number')  # gf2:4 wv3:8   qb:4
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--save_folder', default='/data1/syh/output/model(msdnn)/TrainedNet_qb/', help='Directory to keep training outputs.')  # /data1/syh/output/model/TrainedNet panmamba gf2
parser.add_argument('--outputpath', type=str, default='/data1/syh/output/model/results/', help='Path to output img')
parser.add_argument('--mode', default=3, type=int, help='Train or Test.')
parser.add_argument('--local_rank', default=1, type=int, help='None')
parser.add_argument('--use_distribute', type=int, default=1, help='None')
parser.add_argument('--dataset', type=str, default='qb', help='dataset')
parser.add_argument('--algorithm', type=str, default='msdnn', help='dataset')
opt = parser.parse_args()
device = 'cuda:2'
print(opt)
if opt.algorithm == 'pannet':
    from sota.pannet import Net
    model = Net(num_channels=opt.ChDim).to(device)

elif opt.algorithm == 'panmamba':
    from panmamba import Net
    model = Net(base_filter=opt.ChDim).to(device)

elif opt.algorithm == 'msdnn':
    from sota.msdnn import Net
    model = Net(num_channels=opt.ChDim).to(device)

elif opt.algorithm == 'gppnn':
    from sota.gppnn import Net
    model = Net(num_channels=opt.ChDim).to(device)

elif opt.algorithm == 'dpfn':
    from sota.dpfn import Net
    model = Net(num_channels=opt.ChDim).to(device)
elif opt.algorithm =='dmld':
    from sota.dmld import LocalDissimilarity
    model = LocalDissimilarity(in_channels=opt.ChDim).to(device)
elif opt.algorithm =='dspnet':
    from sota.dspnet import PSNetwork,PSTrainer
    PS_trainer = PSTrainer(model=PSNetwork(), opt=opt.ChDim)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed(opt.seed)

print('===> Loading datasets')

training_data_loader, testing_data_loader = prepare_training_data(dataname=opt.dataset,
                                                                  batch_size=opt.batchSize)  # gf2 wv3

print('===> Building model')
print("===> distribute model")


print('# network parameters: {}'.format(sum(param.numel() for param in model.parameters())))

optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = MultiStepLR(optimizer, milestones=[100, 150, 175, 190, 195], gamma=0.5)

criterion = nn.L1Loss(reduction='none')

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
CURRENT_DATETIME_HOSTNAME = '/' + current_time + '_' + socket.gethostname()
# tb_logger = SummaryWriter(log_dir='./tb_logger/' + 'unfolding2' + CURRENT_DATETIME_HOSTNAME)
current_step = 0


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")


mkdir(opt.save_folder)
mkdir(opt.outputpath)

if opt.nEpochs != 0:
    load_dict = torch.load(opt.save_folder+"_epoch_{}.pth".format(opt.nEpochs))
    opt.lr = load_dict['lr']
    epoch = load_dict['epoch']
    model.load_state_dict(load_dict['param'])
    optimizer.load_state_dict(load_dict['adam'])

def train(epoch, optimizer, scheduler):
    epoch_loss = 0
    global current_step
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        # with torch.autograd.set_detect_anomaly(True):
        ref_results_all = []
        noref_results_all = []
        X, Z, Y = batch[0].to(device), batch[3].to(device), batch[4].to(device) # gt pan ms
        optimizer.zero_grad()
        Y = Variable(Y).float()
        Z = Variable(Z).float()
        X = Variable(X).float()

        HX= model(Y, Z)  # ms pan
        loss = criterion(HX, X).mean()
        # loss = criterion(HX, X).mean()
        # loss = loss1 + 0.2 * loss2
        epoch_loss += loss.item()
        current_step += 1

        loss.backward()
        optimizer.step()

        if iteration % 1000 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))
            # print('value :', np.mean(ref_results_all, axis=0))  # c_psnr, c_ssim, c_sam, c_ergas, c_scc, c_q, c_rmse
            # print('value :', np.mean(noref_results_all, axis=0))  # c_psnr, c_ssim, c_sam, c_ergas, c_scc, c_q, c_rmse
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    return epoch_loss / len(training_data_loader)


def test():
    print('in test')
    avg_psnr = 0
    avg_time = 0
    model.eval()
    with torch.no_grad():
        ref_results_all = []
        noref_results_all = []
        for batch in testing_data_loader:
            X, Z, Y = batch[0].to(device), batch[3].cuda().to(device), batch[4].cuda().to(device)  # gt pan ms
            Y = Variable(Y).float()
            Z = Variable(Z).float()
            X = Variable(X).float()
            start_time = time.time()

            HX= model(Y, Z)
            end_time = time.time()
            # shape (8,8,64,64) bchw
            for i5 in range(HX.shape[0]):
                temp_ref_results = ref_evaluate(np.transpose(HX[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                np.transpose(X[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                scale=4)
                ref_results_all.append(np.expand_dims(temp_ref_results, axis=0))

                temp_noref_results = no_ref_evaluate(np.transpose(HX[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                     np.transpose(Z[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                     np.transpose(Y[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                     scale=4, block_size=16)
                noref_results_all.append(np.expand_dims(temp_noref_results, axis=0))

    print('value :', np.mean(ref_results_all, axis=0))  # c_psnr, c_ssim, c_sam, c_ergas, c_scc, c_q, c_rmse
    print('value :', np.mean(noref_results_all, axis=0))  # c_psnr, c_ssim, c_sam, c_ergas, c_scc, c_q, c_rmse
    return np.mean(ref_results_all, axis=0)[0] / len(testing_data_loader)


def checkpoint(epoch):
    model_out_path = opt.save_folder + "_epoch_{}.pth".format(epoch)
    if epoch % 1 == 0:
        save_dict = dict(
            lr=optimizer.state_dict()['param_groups'][0]['lr'],
            param=model.state_dict(),
            adam=optimizer.state_dict(),
            epoch=epoch
        )
        torch.save(save_dict, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))


if opt.mode == 1:
    for epoch in range(opt.nEpochs + 1, 201):
        avg_loss = train(epoch, optimizer, scheduler)
        if epoch % 50 == 0:
            avg_psnr = test()
            checkpoint(epoch)
        torch.cuda.empty_cache()
        # if local_rank == 0:
        # tb_logger.add_scalar('psnr', avg_psnr, epoch)
        scheduler.step()

elif opt.mode == 3:  ## test gf2
    print('test\n')
    save_dir = '/data1/syh/output/dct/results_panmamba/'
    if opt.dataset == 'gf2':
        test_set = Dataset_Pro("/data1/syh/syh/GF2/test_gf2_multiExm1.h5")
        testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=2, shuffle=False, pin_memory=True,
                                         drop_last=True)
    elif opt.dataset == 'wv3':
        test_set = Dataset_Pro("/data1/syh/syh/WV3/test_wv3_multiExm1.h5")
        testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=2, shuffle=False, pin_memory=True,
                                         drop_last=True)
    elif opt.dataset == 'qb':
        test_set = Dataset_Pro("/data1/syh/syh/QB/test_qb_multiExm1.h5")
        testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=2, shuffle=False, pin_memory=True,
                                         drop_last=True)
    with torch.no_grad():
        ref_results_all = []
        for batch in testing_data_loader:
            X, Z, Y = batch[0].to(device), batch[3].to(device), batch[4].to(device)  # gt pan ms
            Y = Variable(Y).float()
            Z = Variable(Z).float()
            X = Variable(X).float()
            temp = X.data.cpu().numpy()  # 数据类型转换
            np.save('/data1/syh/output/dcfn_result/{}_{}_gt.npy'.format(opt.algorithm, opt.dataset), temp)

            temp = Y.data.cpu().numpy()  # 数据类型转换
            np.save('/data1/syh/output/dcfn_result/{}_{}_ms.npy'.format(opt.algorithm, opt.dataset), temp)
            temp = Z.data.cpu().numpy()  # 数据类型转换
            np.save('/data1/syh/output/dcfn_result/{}_{}_pan.npy'.format(opt.algorithm, opt.dataset), temp)

            start_time = time.time()
            HX= model(Y, Z)
            temp = HX.data.cpu().numpy()  # 数据类型转换
            np.save('/data1/syh/output/dcfn_result/{}_{}_f.npy'.format(opt.algorithm, opt.dataset), temp)

            print('---')
            input()
            end_time = time.time()
            # shape (8,8,64,64) bchw
            for i5 in range(HX.shape[0]):
                temp_ref_results = ref_evaluate(np.transpose(HX[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                np.transpose(X[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                scale=4)
                ref_results_all.append(np.expand_dims(temp_ref_results, axis=0))
            print('value :', np.mean(ref_results_all, axis=0))  # c_psnr, c_ssim, c_ergas, c_scc, c_q, c_rmse
    print('value :', np.mean(ref_results_all, axis=0))  # c_psnr, c_ssim, c_sam, c_ergas, c_scc, c_q, c_rmse



elif opt.mode == 4:  ## 测试 full
    print('test\n')
    model.eval()
    if opt.dataset =='gf2':
        test_set = Dataset_Pro_full("/data1/syh/syh/GF2/test_gf2_OrigScale_multiExm1.h5")
        testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=2, shuffle=False, pin_memory=True,
                                     drop_last=True)
    elif opt.dataset =='wv3':
        test_set = Dataset_Pro_full("/data1/syh/syh/WV3/test_wv3_OrigScale_multiExm1.h5")
        testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=2, shuffle=False, pin_memory=True,
                                     drop_last=True)
    elif opt.dataset == 'qb':
        test_set = Dataset_Pro_full("/data1/syh/syh/QB/test_qb_OrigScale_multiExm1.h5")
        testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=2, shuffle=False, pin_memory=True,
                                         drop_last=True)
    with torch.no_grad():
        noref_results_all = []

        for batch in testing_data_loader:
            Z, Y = batch[1].to(device), batch[2].to(device) # pan ms
            Y = Variable(Y).float()
            Z = Variable(Z).float()
            start_time = time.time()
            HX= model(Y, Z)
            temp = HX.data.cpu().numpy()  # 数据类型转换
            np.save('/data1/syh/output/dcfn_result/{}_{}_full_f.npy'.format(opt.algorithm, opt.dataset), temp)
            print('---')
            input()
            end_time = time.time()
            # print('D_lambda :', D_lambda)
            for i5 in range(Z.shape[0]):
                temp_noref_results = no_ref_evaluate(np.transpose(HX[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                     np.transpose(Z[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                     np.transpose(Y[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                     scale=4)
                noref_results_all.append(np.expand_dims(temp_noref_results, axis=0))
            print('value :', np.mean(noref_results_all, axis=0))  # c_psnr, c_ssim, c_sam, c_ergas, c_scc, c_q, c_rmse


else:
    test()
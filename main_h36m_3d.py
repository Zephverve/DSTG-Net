import os
import sys

sys.path.append(os.path.abspath('./'))
from utils import h36motion3d as datasets
from utils.opt import Options
from utils import util
from utils import log
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim
from model.dstg_net import DSTGNet


def load_model(args):
    act_mapper = {
        "gelu": nn.GELU,
        'relu': nn.ReLU
    }
    model = DSTGNet(
        num_joints=args.in_features // 3,
        dim_in=3,
        dim_feat=128,
        dim_rep=512,
        n_layers=16,  # 16
        mlp_ratio=4,  # 4
        act_layer=act_mapper["gelu"],  # "gelu"
        attn_drop=0.5,  # 0.0
        drop=0.3,  # 0.0
        drop_path=0.3,  # 0.0
        num_heads=8,  # 8
        use_layer_scale=True,  # True
        qkv_bias=False,  # False
        qkv_scale=None,  # None
        layer_scale_init_value=1e-05,  # 1e-05
        use_adaptive_fusion=True,  # True
        hierarchical=False,  # False
        use_temporal_similarity=True,  # True
        temporal_connection_len=1,  # 1
        use_tcn=True,  # False
        graph_only=False,  # False
        neighbour_num=2,  # 2
        n_frames=args.dct_n  # 20
    )

    return model


def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    print('>>> create models')

    net_pred = load_model(opt)
    net_pred.to(opt.cuda_idx)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    if opt.is_load or opt.is_eval:
        if opt.is_eval:
            model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        else:
            model_path_len = './{}/ckpt_last.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    if not opt.is_eval:

        dataset = datasets.Datasets(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=min(8, os.cpu_count() // 2),  # 使用多线程
            pin_memory=True,
            persistent_workers=True,  # 保持 worker 常驻
            prefetch_factor=4
        )

        valid_dataset = datasets.Datasets(opt, split=2)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=opt.test_batch_size,
            shuffle=True,
            num_workers=min(8, os.cpu_count() // 2),  # 使用多线程
            pin_memory=True,
            persistent_workers=True,  # 保持 worker 常驻
            prefetch_factor=4
        )

    test_dataset = datasets.Datasets(opt, split=2)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.test_batch_size,
        shuffle=True,
        num_workers=min(8, os.cpu_count() // 2),  # 使用多线程
        pin_memory=True,
        persistent_workers=True,  # 保持 worker 常驻
        prefetch_factor=4
    )
    if opt.is_eval:
        ret_test,out_pred,out_tr = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')

    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))
            ret_train,out_pred,out_tr = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))
            ret_valid,out_pred,out_tr = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
            ret_test,out_pred,out_tr = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, epo=epo)
            print('testing error: {:.3f}'.format(ret_test['#40ms']))
            print(ret_test)
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            if ret_valid['m_p3d_h36'] < err_best:
                err_best = ret_valid['m_p3d_h36']
                is_best = True
            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_p3d_h36'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)


def eval(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    print('>>> create models')
    net_pred = load_model(opt)
    net_pred.to(opt.cuda_idx)
    net_pred.eval()

    model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len, weights_only=False,map_location="cuda:0")
    net_pred.load_state_dict(ckpt['state_dict'])

    print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    acts = ["walking", "eating", "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether"]

    data_loader = {}
    for act in acts:
        dataset = datasets.Datasets(opt=opt, split=2, actions=act)
        data_loader[act] = DataLoader(dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                      pin_memory=True)
    is_create = True
    avg_ret_log = []

    for act in acts:
        ret_test,out_pred,out_tr = run_model(net_pred,act, is_train=3, data_loader=data_loader[act], opt=opt)
        ret_log = np.array([act])
        head = np.array(['action'])

        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, ['test_' + k])

        avg_ret_log.append(ret_log[1:])
        log.save_csv_eval_log(opt, head, ret_log, is_create=is_create)

        is_create = False

    avg_ret_log = np.array(avg_ret_log, dtype=np.float64)
    avg_ret_log = np.mean(avg_ret_log, axis=0)

    write_ret_log = ret_log.copy()
    write_ret_log[0] = 'avg'
    write_ret_log[1:] = avg_ret_log
    log.save_csv_eval_log(opt, head, write_ret_log, is_create=False)




import matplotlib.pyplot as plt


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

scaler = torch.cuda.amp.GradScaler()  # 混合精度
decorrelation_losses=[]

def run_model(net_pred,optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
    if is_train == 0:
        net_pred.train()

    else:
        net_pred.eval()

    l_p3d = 0
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = (np.array(range(opt.output_n)) + 1) * 40
        m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    seq_in = opt.kernel_size
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    st = time.time()  # 整个训练的开始时间（原有）
    batch_1000_start_time = st  # 记录每1000批的起始时间（新增）


    st = time.time()
    for i, (p3d_h36) in enumerate(data_loader):

        batch_size, seq_n, all_dim = p3d_h36[0].shape

        p3d_h36 = p3d_h36[0]

        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size

        batch_start_time = time.time()

        p3d_h36 = p3d_h36.float().to(opt.cuda_idx)


        input = p3d_h36[:, :, dim_used].clone()

        p3d_sup_4 = p3d_h36.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])


        idx = list(range(seq_in)) + [seq_in - 1] * out_n  # 把输入帧拿出来，取得是前10帧然后把第10帧一直复制到35帧
        input_gcn = input[:, idx].clone()

        dct_m, idct_m = util.get_dct_matrix(seq_in + out_n)
        dct_m = torch.from_numpy(dct_m).float().to(opt.cuda_idx)  # torch.Size([35, 35])
        idct_m = torch.from_numpy(idct_m).float().to(opt.cuda_idx)  # torch.Size([35, 35])



        input_gcn_dct = torch.matmul(dct_m[:opt.dct_n, :], input_gcn)
        input_gcn_dct = input_gcn_dct.reshape(-1, opt.dct_n, 22, 3)

        with torch.amp.autocast('cuda'):
            p3d_out_all_4, decorrelation_loss = net_pred(
                input_gcn_dct)  # torch.Size([32, 20, 22, 3])输入 torch.Size([32, 20, 22, 3])输出

        p3d_out_all_4 = p3d_out_all_4.reshape([-1, opt.dct_n, 66])

        p3d_out_all_4 = torch.matmul(
            idct_m[:, :opt.dct_n].to(p3d_out_all_4.dtype),
            p3d_out_all_4
        )

        p3d_out_4 = p3d_h36.clone()[:, in_n:in_n + out_n]  # 取出真实值的后25帧
        p3d_out_4[:, :, dim_used] = p3d_out_all_4[:, seq_in:].to(p3d_out_4.dtype)
        p3d_out_4[:, :, index_to_ignore] = p3d_out_4[:, :, index_to_equal]

        p3d_out_4 = p3d_out_4.reshape([-1, out_n, 32, 3])
        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 32, 3])
        p3d_out_all_4 = p3d_out_all_4.reshape([batch_size, seq_in + out_n, len(dim_used) // 3, 3])
        out_pred = p3d_out_4.cpu().data.numpy()
        out_tr = p3d_h36.cpu().data.numpy()
        out_tr = out_tr[:,seq_in:]
        grad_norm = 0
        if is_train == 0:

            v_t_4 = p3d_sup_4[:, 1:] - p3d_sup_4[:, :-1]
            v_p_4 = p3d_out_all_4[:, 1:] - p3d_out_all_4[:, :-1]
            loss_p3d_4 = torch.mean(torch.norm(p3d_out_all_4 - p3d_sup_4, dim=3))
            loss_p3d_4_v = torch.mean(torch.norm(v_p_4 - v_t_4, dim=3))

            loss_all =  0.5 * (loss_p3d_4)  + 0.5 * (loss_p3d_4_v)+ 0.00005 * decorrelation_loss
            optimizer.zero_grad()

            scaler.scale(loss_all).backward()
            scaler.step(optimizer)
            scaler.update()
            l_p3d += loss_p3d_4.cpu().data.numpy() * batch_size

        if is_train <= 1:  # if is validation or train simply output the overall mean error
            mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out_4, dim=3))
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
        else:

            mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out_4, dim=3), dim=2), dim=0)
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()

        batch_end_time = time.time()

        if i % 1000 == 0:
            total_1000batch_time = batch_end_time - batch_1000_start_time
            avg_1000batch_time = total_1000batch_time / 1000

            print("=" * 80)
            print(f"【1000批汇总】 起始批: {i + 1 - 999} ~ 结束批: {i + 1} | "
                  f"1000批总用时: {total_1000batch_time:.2f}s | "
                  f"平均每批用时: {avg_1000batch_time:.3f}s")
            print("=" * 80)

            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - batch_start_time,
                                                           time.time() - st, grad_norm))
            print(m_p3d_h36 / n)
            batch_1000_start_time = time.time()  # 整个训练的开始时间（原有）



    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    else:
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(out_n):
            ret["#{:d}ms".format(titles[j])] = m_p3d_h36[j]

    if is_train == 0:
        plt.savefig(f'decorrelation_loss_epoch_{epo}.png')
    return ret,out_pred,out_tr


if __name__ == '__main__':

    option = Options().parse()

    if option.is_eval == False:
        main(opt=option)
    else:
        eval(option)

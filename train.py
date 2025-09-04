##################################
# Train GIGA-ToF with DVToF dataset
##################################
import time
import os
import argparse

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import logging
from giga.GIGA import GIGAToF
from giga.DVToF_dataloader import pbrt_dv_Dataset
from loss import GLoss, GLoss_test

import warnings

warnings.filterwarnings("ignore")


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def train(args):
    cudaid = "cuda:" + str(args.dev)
    device = torch.device(cudaid)

    # args
    batch_size = args.batch_size
    lr = args.learning_rate
    total_epoch = args.epoch
    out_path = args.destination
    debug_path = args.debug

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)
    out_model = os.path.join(out_path, args.name)
    print(device, out_model)

    # dataset
    train_data = pbrt_dv_Dataset(root=args.train_path, mode='train')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_data = pbrt_dv_Dataset(root=args.train_path, mode='test')
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

    print("============ Data loaded =============")

    # model
    gspn = GIGAToF()
    gspn.to(device)
    
    # init
    start_epoch = 0
    if args.model:
        print("Continue training from: ", args.model)
        try:
            # load checkpoint
            checkpoint = torch.load(args.model, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # loading
                gspn.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
            else:
                # weight only
                gspn.load_state_dict(checkpoint)
        except Exception as e:
            print(f"error: {e}")
            return

    # optimizer
    optimizer = optim.Adam(gspn.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)
    
        
    
    if args.model and start_epoch > 0:
        try:
            checkpoint = torch.load(args.model, map_location=device)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          
                for _ in range(start_epoch):
                    scheduler.step()
                print(f"lr: {scheduler.get_last_lr()[0]:.2e}")
        except Exception as e:
            print(f"error: {e}")
    
    
    #loss_fn = GLoss(device)
    loss_fn = GLoss(device=device, alpha=0.3, lambda_attconf_reg=0.001, lambda_inter_sparse=0.001)
    loss_fn.to(device)
    loss_fn_test = GLoss_test()
    loss_fn_test.to(device)

    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # new logger
    log_path = os.path.join(log_dir, f'exp_{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())}.log')
    logger = get_logger(log_path)
    logger.info('Start logging...\n')

    tstart = time.time()
    best_loss = 100
    ld = len(train_dataloader)

    for epoch in range(0, total_epoch):  # loop over the dataset multiple times

        scheduler.step()
        train_step = 0
        gspn.train()
        train_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):  # start index at 0
            # get the inputs; data is a list of [inputs, labels, ideal_d]
            raw_IQ, raw_IQ_pre, ideal_IQ, ideal_d = data
            raw_IQ = raw_IQ.to(device)  # [batch_size, 6, H, W]
            raw_IQ_pre = raw_IQ_pre.to(device)  # [batch_size, 6, H, W]
            ideal_IQ = ideal_IQ.to(device)  # [batch_size, 1, H, W]
            ideal_d = ideal_d.to(device)
            
            out_0, mu0, inter_graph_0, attconf_0 = gspn(raw_IQ[:, 0:2, :, :], raw_IQ_pre[:, 0:2, :, :])
            out_1, mu1, inter_graph_1, attconf_1 = gspn(raw_IQ[:, 2:4, :, :], raw_IQ_pre[:, 2:4, :, :])
            out_2, mu2, inter_graph_2, attconf_2 = gspn(raw_IQ[:, 4:6, :, :], raw_IQ_pre[:, 4:6, :, :])

            optimizer.zero_grad()
            
            loss = loss_fn(out_0, out_1, out_2, ideal_IQ, ideal_d,
               inter_graph_0, attconf_0,
               inter_graph_1, attconf_1,
               inter_graph_2, attconf_2)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_step += 1

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{total_epoch} Step {train_step}/{ld}] [Loss: {loss.item()}]")

        train_loss /= train_step
        time.ctime()
        info = f"[Epoch {epoch}/{total_epoch}] [Train Loss: {train_loss}] [time eclapsed {time.time() - tstart}]"
        logger.info(f"{info}\n")

        gspn.eval()
        test_step = 0
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                # get the inputs; data is a list of [inputs, labels, ideal_d]
                raw_IQ, raw_IQ_pre, ideal_IQ, ideal_d = data
                raw_IQ = raw_IQ.to(device)  # [batch_size, 6, H, W]
                raw_IQ_pre = raw_IQ_pre.to(device)  # [batch_size, 6, H, W]
                ideal_IQ = ideal_IQ.to(device)  # [batch_size, 1, H, W]
                ideal_d = ideal_d.to(device)  # [batch_size, 1, H, W]
                
                out_0, mu0, inter_graph_0, attconf_0 = gspn(raw_IQ[:, 0:2, :, :], raw_IQ_pre[:, 0:2, :, :])
                out_1, mu1, inter_graph_1, attconf_1 = gspn(raw_IQ[:, 2:4, :, :], raw_IQ_pre[:, 2:4, :, :])
                out_2, mu2, inter_graph_2, attconf_2 = gspn(raw_IQ[:, 4:6, :, :], raw_IQ_pre[:, 4:6, :, :])
                
                loss = loss_fn_test(out_0, out_1, out_2, ideal_IQ, ideal_d)
                test_loss += loss.item()
                test_step += 1

                if (epoch % 10 == 0) & (i % 5 == 0):
                    outputs = torch.concatenate((out_0, out_1, out_2), axis=1).cpu()
                    outputs = outputs[0].detach().numpy()
                    np.save(f"{args.debug}/epoch_{epoch}_{i}.npy", outputs)

        test_loss /= test_step
        info = f"[Epoch {epoch}/{total_epoch}] [Test Loss: {test_loss}]"
        logger.info(f"{info}\n")

        if test_loss < best_loss:
            best_loss = test_loss
            model_state_dict = gspn.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            checkpoint = {
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'epoch': epoch,
                'loss': train_loss
            }
            torch.save(checkpoint, f"{out_path}/checkpoint_best.pth")

        if epoch % 10 == 0:
            print("save @ epoch ", epoch + 1)
            torch.save(gspn.state_dict(), f"{out_path}/checkpoint_{epoch}.pth")

    logger.info("End logging.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dev', type=int, default=0, help='device id')
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4,
                        help="Training learning rate. Default is 1e-3, or 2e-4 for FT")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay (L2 loss on parameters).')

    parser.add_argument("-in", "--train_path", type=str, default='./dataset', help="Train set directory")
    parser.add_argument("-out", "--destination", type=str, default='./result', help="Output destination.")
    parser.add_argument("-d", "--debug", type=str, default='./result_debug', help="Result directory.")
    parser.add_argument("-m", "--model", type=str, default=None, help="Path to the trained GIGAToF.")
    parser.add_argument("-n", "--name", type=str, default='giga.pkl', help="Name of model.",
                        )

    parser.add_argument("-e", "--epoch", type=int, default=200, help="Total epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=10, help="Training batch size. Default is 1")

    parser.add_argument("-barron", "--noise_barron", type=bool, default=False, help="process barron noise or not")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train(args)

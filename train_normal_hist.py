import os
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from dataset_fast_latest import HIPSDataset_fast
from model_normal_hist import model_Regression_add
import clip

if not os.path.exists("log"):
    os.makedirs("log")
if not os.path.exists("plot"):
    os.makedirs("plot")
if not os.path.exists("models"):
    os.makedirs("models")

PREFIX = "Latest"
ARCH = "L1L2_ADP_stat_lstm_64nf_4rb_64h_stepLR_3ai"
CHANNEL = 12
IMGSIZE = (20, 10)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def plog(msg, finetuning=False):
    if finetuning:
        print(msg)
        with open("log/mylog_%s_%s_finetune.txt" % (PREFIX, ARCH), "a") as fp:
            fp.write(msg + "\n")
    else:
        print(msg)
        with open("log/mylog_%s_%s.txt" % (PREFIX, ARCH), "a") as fp:
            fp.write(msg + "\n")


def plot_loss(x, history):
    plt.plot(x, history["train_loss"], label="train", marker="o")
    plt.title("Loss per epoch")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(), plt.grid()
    plt.savefig(f"./plot/loss_{ARCH}.png")
    # plt.show()

def sigmoid_weights(num_matrices, tp_init):
        x = np.linspace(tp_init-1, tp_init-1+num_matrices-1, num_matrices)  
        return 1 / (1 + np.exp(-x))

def coarse_train(batch_size=32, lr=0.0005, epoch=120):
    # create dataset
    train_dataset = HIPSDataset_fast(
        root="/ssd8/chingheng/IPPS/data_latest", single=True, ishist=True
    )

    # split dataset for training and validation
    train_size = len(train_dataset)

    print(f"Traning dataset size: {train_size}")

    device = "cuda"

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    # define model
    # model = MyModel(in_dim=CHANNEL, img_size=IMGSIZE)
    model = model_Regression_add(in_dim=CHANNEL, img_size=IMGSIZE, n_feats=64, add_hidden=64, n_resblocks=4)
    model = nn.DataParallel(model)
    with torch.no_grad():
        model_clip, _ = clip.load("ViT-B/32", device="cuda")
        model_clip.eval()

    # trainig step
    model.to(device)
    model.cuda()
    loss_l2 = nn.MSELoss()
    loss_l1 = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 30, gamma=0.65, last_epoch=-1, verbose=True
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=epoch, eta_min=0, last_epoch=-1, verbose=True
    # )
    t_loss = []
    if not os.path.exists("models"):
        os.makedirs("models")
    for i in range(epoch):
        print(f"-------[Epoch: {i+1}]-------")
        running_loss = 0
        running_loss_L1 = 0
        running_loss_L2 = 0
        model.train()
        for data in train_loader:
            x, t, gt, ai, hist = data["hsi_all"], data["prompt"], data["gt"], data["add_info"], data["hist"]
            optimizer.zero_grad()
            # print(x.shape)
            x = x.to(device).float()
            gt = gt.to(device).float()
            ai = ai.to(device).float()
            hist = hist.to(device).float()

            t_fea = torch.empty(x.shape[0], 1, 512).to(device)
            for j in range(x.shape[0]):
                t_tok = clip.tokenize([t[j]])
                with torch.no_grad():
                    t_fea[j, :, :] = model_clip.encode_text(t_tok.to(device))
            y = model(x, t_fea.float(), ai, hist)
            # print(y, gt)
            loss1 = loss_l1(y, gt)
            loss2 = loss_l2(y, gt)
            loss = loss2 + loss1
            # loss = loss2
            running_loss_L1 += loss1.item()
            running_loss_L2 += loss2.item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        t_loss.append(running_loss*batch_size / train_size)
        scheduler.step()

        print(
            f"[Epoch: {i+1}] L1 loss: {running_loss_L1*batch_size / train_size}, L2 loss: {running_loss_L2*batch_size / train_size}"
        )

        plog(
            "[epoch: %d, batch: %d] train_loss: %.3f,  Leraning Rate: %f"
            % (
                i + 1,
                batch_size,
                (running_loss * batch_size / train_size),
                scheduler.get_last_lr()[0],
            )
        )
        if (i + 1) % 100 == 0:
            torch.save(model, f"models/model_{ARCH}_epoch-{i+1}.pth")

    history = {"train_loss": t_loss}

    plot_loss(np.arange(0, epoch), history)

    return f"{ARCH}_bm.pth"


if __name__ == "__main__":
    model_name = coarse_train(batch_size=8, lr=3e-4, epoch=300)

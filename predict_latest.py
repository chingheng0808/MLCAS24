from dataset_test import HIPSDataset_test
from torch.utils.data import DataLoader
import torch
import clip
import pandas as pd
import os
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('model_path', help='Input pre-trained model path')

# args = parser.parse_args()

if not os.path.exists("results_test"):
    os.makedirs("results_test")

test_dataset = HIPSDataset_test(root="MLCAS24_Competition_data")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

df = pd.read_csv(
    "MLCAS24_Competition_data/test/Test/GroundTruth/test_HIPS_HYBRIDS_2023_V2.3.csv"
)

CHANNEL = 12
IMGSIZE = (20, 10)
device = "cuda"

def predict_result(model_path, device='cuda'):
    model = torch.load(model_path)
    model.to(device)
    model.eval()

    with torch.no_grad():
        model_clip, _ = clip.load("ViT-B/32", device=device)
        model_clip.eval()
    with torch.no_grad():
        for data in test_loader:
            x, x_all, t, ai, hist, tp_idx, img_path_list = data["hsi"], data["hsi_all"], data["prompt"], data["add_info"], data["hist"], data["tp_idx"], data["img_path"]
            # print(x.shape, ai.shape, x_all.shape, hist.shape)
            x = x.to(device).float()
            ai = ai.to(device).float()
            x_all = x_all.to(device).float()
            hist = hist.to(device).float()
            t_fea = torch.empty(x.shape[0], 1, 512).to(device)
            for j in range(x.shape[0]):
                t_tok = clip.tokenize([t[j]])
                with torch.no_grad():
                    t_fea[j, :, :] = model_clip.encode_text(t_tok.to(device))

            row, rng = int(ai[0, 0]), int(ai[0, 1])
            idx = df[(df["row"] == row) & (df["range"] == rng)].index.to_list()[0]
            
            try:
                y = model(x, t_fea.float(), ai)
            except:
                try:
                    y = model(x, t_fea.float(), ai, hist)
                except:
                    try:
                        y = model(x_all, t_fea.float(), ai, hist)
                    except:
                        y = model(x_all, t_fea.float(), ai)
            # multip
            # y = model(x, t_fea.float(), ai)
            # hist
            # y = model(x_all, t_fea.float(), ai, hist)
            # normal
            # y = model(x_all, t_fea.float(), ai)
            # print(t)
            df.at[idx, "yieldPerAcre"] = float(y)

    df.to_csv(f"results_test/results_test_{model_path.split('/')[-1][6:-4]}.csv", index=False)


if __name__ == "__main__":
    predict_result('/ssd8/chingheng/IPPS/Submission/models/model_L1L2_ADP_stat_lstm_64nf_4rb_64h_stepLR_3ai_epoch-200.pth')
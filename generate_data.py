from dataset_dul_latest import HIPSDataset_dul_mulret
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

def calculate_histogram(
        imagecol, num_bins=32, bands=12, max_bin_val=1
    ):
        bin_seq = np.linspace(0, max_bin_val, num_bins + 1)

        hist = []
        for j in range(imagecol.shape[0]): # 6 images in an imagecol
            imhist = []
            im = imagecol[j]
            for i in range(im.shape[0]): #iter bands
                density, _ = np.histogram(im[i, :, :], bin_seq, density=False)
                # max() prevents divide by 0
                density = density / max(1, density.sum())
                if density[0] == 1.:
                    density[0] = 0
                # print(sum(density))
                imhist.append(density)
            
            hist.append(np.stack(imhist))
        return np.stack(hist).astype(np.float32)

if not os.path.exists('data_latest'):
    os.makedirs('data_latest')

device = "cuda"
 
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomApply(
                [transforms.Resize((18, 8)), transforms.Pad(1, fill=0)], p=0.2
            ),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
        ])

train_dataset = HIPSDataset_dul_mulret(
    root="MLCAS24_Competition_data", transform=transform, location=True, 
)
data_loader = DataLoader(
    train_dataset, batch_size=1, shuffle=False, num_workers=1
)

for i, data in enumerate(data_loader):
    hsi, prompt, gt, add_info, hsi_all = data['hsi'], data["prompt"], data["gt"], data["add_info"], data['hsi_all']
    # print(hsi.squeeze(0).shape)
    hist = calculate_histogram(hsi.squeeze(0))
    np.savez(f'data_latest/data{i}.npz', hsi = hsi.squeeze(0), gt = gt.squeeze(0), add_info = add_info.squeeze(0), hsi_all = hsi_all.squeeze(0), hist = hist)
    f = open(f"data_latest/data{i}.txt", "w")
    f.write(prompt[0])
    f.close()

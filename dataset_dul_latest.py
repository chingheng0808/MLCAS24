import rasterio
import os
import warnings
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset


def normHSI(R):
    if isinstance(R, torch.Tensor):
        rmax, rmin = torch.max(R), torch.min(R)
    elif isinstance(R, np.ndarray):
        rmax, rmin = np.max(R), np.min(R)
    else:
        warnings.warn("Unsupport data type of input HSI")
        return
    R = (R - rmin) / (rmax - rmin)
    return R


def normHSI_channelwise(R):
    for i in range(R.shape[0]):
        rmax, rmin = np.max(R[i]), np.min(R[i])
        R[i] = (R[i] - rmin) / (rmax - rmin)
    return R


def read_satelliteimage(img_path):
    warnings.filterwarnings("ignore")
    with rasterio.open(img_path, "r") as src:
        S_images = src.read()
        h, w = 20, 10
        msi = np.zeros((6, h, w), dtype=np.float32)
        msi_ = normHSI(S_images.astype(np.float32))[:, 1:21, 1:11]
        msi[:, : msi_.shape[1], : msi_.shape[2]] = (
            msi_  ## bands: 0-5 --> blue, green, red, nir, re, db
        )

        # hsi = np.zeros((7, h, w), dtype=np.float32)
        hsi = np.zeros((6, h, w), dtype=np.float32)
        # Calculate some pixel-wise indexes
        for height in range(h):
            for width in range(w):
                r, g, b, nir, re, db = (
                    msi[0, height, width],
                    msi[1, height, width],
                    msi[2, height, width],
                    msi[3, height, width],
                    msi[4, height, width],
                    msi[5, height, width],
                )

                GLI = (2 * g - r - b) / (2 * g + r + b)

                GRVIupper = g - r.astype(float)
                GRVIlower = g.astype(float) + r.astype(float)
                GRVI = GRVIupper.astype(float) / GRVIlower.astype(float)

                NGRDIupper = r.astype(float) - g.astype(float)
                NGRDIlower = g.astype(float) + r.astype(float)
                NGRDI = NGRDIupper.astype(float) / NGRDIlower.astype(float)

                NDVIupper = nir.astype(float) - r.astype(float)
                NDVIlower = nir.astype(float) + r.astype(float)
                NDVI = NDVIupper.astype(float) / NDVIlower.astype(float)

                GNDVIupper = nir.astype(float) - g.astype(float)
                GNDVIlower = nir.astype(float) + g.astype(float)
                GNDVI = GNDVIupper.astype(float) / GNDVIlower.astype(float)

                # SAVIupper=1.5*(nir.astype(float)-r.astype(float))
                # SAVIlower=nir.astype(float)+r.astype(float)+0.5
                # SAVI=(SAVIupper.astype(float)/SAVIlower.astype(float))

                NDREupper = nir.astype(float) - re.astype(float)
                NDRElower = nir.astype(float) + re.astype(float)
                NDRE = NDREupper.astype(float) / NDRElower.astype(float)

                # hsi[0,height,width],hsi[1,height,width],hsi[2,height,width],hsi[3,height,width],hsi[4,height,width],hsi[5,height,width],hsi[6,height,width]=GLI,GRVI,NGRDI,NDVI,GNDVI,SAVI,NDRE
                (
                    hsi[0, height, width],
                    hsi[1, height, width],
                    hsi[2, height, width],
                    hsi[3, height, width],
                    hsi[4, height, width],
                    hsi[5, height, width],
                ) = GLI, GRVI, NGRDI, NDVI, GNDVI, NDRE
        # print(hsi.shape, msi.shape)
        hsi = np.append(msi, hsi, axis=0)
        # hsi = msi

    return hsi


def read_data(row_idx, sheet_path, img_root, n_sh1):
    if row_idx > n_sh1 - 1:
        row_idx -= n_sh1
        return get_data(row_idx, sheet_path[1], img_root[1])
    else:
        return get_data(row_idx, sheet_path[0], img_root[0])


def get_data(row_idx, sheet_path, img_root):
    data = pd.read_csv(sheet_path)
    img_locs = os.listdir(img_root)

    # read csv infos and additional infos
    location = data.loc[row_idx]["location"]
    row = data.loc[row_idx]["row"]
    rangeno = data.loc[row_idx]["range"]
    experiment = data.loc[row_idx]["experiment"]
    poundsN2 = data.loc[row_idx]["poundsOfNitrogenPerAcre"]
    # for generating prompt
    genotype = data.loc[row_idx]["genotype"]
    n2level = data.loc[row_idx]["nitrogenTreatment"]
    loc = data.loc[row_idx]["location"]
    block = data.loc[row_idx]["block"]

    gt = data.loc[row_idx]["yieldPerAcre"]

    block_infos = {
        "block": block,
        "genotype": genotype,
        "nitrogenTreatment": n2level,
        "poundsN2": poundsN2,
        "yieldPerAcre": gt,
        "row": row,
        "experiment": experiment,
        "range": rangeno,
        "location": loc,
    }

    for loc_folder in img_locs:
        if not loc_folder == location:
            continue
        locationfolder = os.path.join(img_root, location)

    timepointfolder = os.listdir(locationfolder)
    timepointfolder = sorted(timepointfolder)  # [PT1, PT2, etc] likely
    timepointpath = [os.path.join(locationfolder, x) for x in timepointfolder]

    tp_idx = []
    img_path_list = []
    for timepointpath_ in timepointpath:
        imagefiles = os.listdir(timepointpath_)

        # for image name: "location-timepoint-experiment_range_row.TIF"
        for images in imagefiles:
            range_ = images.split("_")[1]
            row_ = images.split("_")[2].split(".")[0]
            experiment_ = images.split("_")[0].split("-")[2]
            if (
                str(range_) == str(rangeno)
                and str(row_) == str(row)
                and str(experiment) == str(experiment_)
            ):
                tp_idx.append(int(images.split("_")[0].split("-")[1][-1]))
                img_path = os.path.join(timepointpath_, images)
                img_path_list.append(img_path)

    return img_path_list, block_infos, tp_idx


class HIPSDataset_dul_mulret(Dataset):
    def __init__(self, root, transform=None, location=False):
        self.root = root
        self.location = location
        self.sheet_path = [
            os.path.join(
                root,
                "train/2023/DataPublication_final/GroundTruth/train_HIPS_HYBRIDS_2023_V2.3.csv",
            ),
            os.path.join(
                root,
                "train/2022/2022/DataPublication_final/GroundTruth/DropNaN_HYBRID_HIPS_V3.5_ALLPLOTS.csv",
            ),
        ]
        self.img_root = [
            os.path.join(root, "train/2023/DataPublication_final/Satellite"),
            os.path.join(root, "train/2022/2022/DataPublication_final/Satellite"),
        ]

        self.data_length = self.get_data_info(self.sheet_path)
        self.transform = transform

        self.sheet1_len = len(pd.read_csv(self.sheet_path[0]))

    def __getitem__(self, index):
        hsi_all = np.zeros((12, 20, 10), dtype=np.float32)
        # hsi = np.zeros((6, 12, 20, 10), dtype=np.float32) #最多6tp
        # filter out those hsi contains half zero values
        while (
            np.count_nonzero(np.mean(hsi_all[[0, 1, 2], :, :], axis=0) == 0.0)
            >= hsi_all.shape[1] * hsi_all.shape[2] // 2
        ):
            hsi = np.zeros((6, 12, 20, 10), dtype=np.float32)
            img_path_list, block_info, tp_idx = read_data(
                index, self.sheet_path, self.img_root, self.sheet1_len
            )
            while not tp_idx:
                index = np.random.randint(0, self.data_length)
                img_path_list, block_info, tp_idx = read_data(
                    index, self.sheet_path, self.img_root, self.sheet1_len
                )
                # print('hi')
            seed = torch.seed()
            for idx, tp in enumerate(tp_idx):
                x = read_satelliteimage(img_path_list[idx])
                if self.transform is not None:
                    torch.manual_seed(seed)
                    x = self.transform(x.transpose(1, 2, 0))
                    x = torch.nan_to_num(x)
                else:
                    x = torch.nan_to_num(torch.tensor(x))  # prevent nan in hsi
                hsi[tp - 1] = x

            hsi_all = np.zeros((12, 20, 10), dtype=np.float32)
            weights = self.sigmoid_weights(len(tp_idx), tp_idx[0])
            w_t = sum(weights)
            for i, img_path in enumerate(img_path_list):
                hsi_all += (weights[i] / w_t) * hsi[
                    tp_idx[i] - 1
                ]  # weighted sum of HSI in different time points

            index = np.random.randint(0, self.data_length)

        if self.location:
            prompt = f"{len(tp_idx)} images, {block_info['genotype']}, {block_info['nitrogenTreatment']}, {block_info['location']}"
        else:
            prompt = f"{len(tp_idx)} images, {block_info['genotype']}, {block_info['nitrogenTreatment']}"
        addInfo = torch.tensor(
            [
                block_info["row"],
                block_info["range"],
                block_info["block"],
                block_info["poundsN2"],
            ]
        )

        self.data = {
            "hsi": hsi,
            "add_info": addInfo,
            "prompt": prompt,
            "gt": torch.tensor([block_info["yieldPerAcre"]]),
            "hsi_all": hsi_all,
        }

        return self.data

    def __len__(self):
        return self.data_length

    @staticmethod
    def get_data_info(sheet_path):
        data_length = 0
        for k in range(len(sheet_path)):
            df = pd.read_csv(sheet_path[k])
            data_length += len(df)
        return data_length

    @staticmethod
    def sigmoid_weights(num_matrices, tp_init):
        x = np.linspace(tp_init - 1, tp_init - 1 + num_matrices - 1, num_matrices)
        return 1 / (1 + np.exp(-x))

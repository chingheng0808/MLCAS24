import os
import numpy as np
from torch.utils.data import Dataset


class HIPSDataset_fast(Dataset):
    def __init__(self, root, single=True, ishist=True):
        self.root = root
        self.single = single
        self.ishist = ishist

    def __getitem__(self, index):
        data = np.load(os.path.join(self.root, f'data{index}.npz'))

        txt = open(os.path.join(self.root, f'data{index}.txt'), 'r')
        prompt = txt.read()
        txt.close()

        if self.single:
            self.data = {
            "hsi": data['hsi'],
            "add_info": data['add_info'],
            "prompt": prompt,
            "gt": data['gt'],
            "hsi_all": data['hsi_all']
            }
        else:
            self.data = {
            "hsi": data['hsi'],
            "add_info": data['add_info'],
            "prompt": prompt,
            "gt": data['gt'],
            }

        if self.ishist:
            self.data['hist'] = data['hist']

        return self.data

    def __len__(self):
        return len(os.listdir(self.root))//2

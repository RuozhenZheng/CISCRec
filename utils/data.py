import torch.utils.data as data
import torch
import numpy as np


class Data_sampler(data.Dataset):
    def __init__(self, batch):
        """
        Dataset formatter adapted pair-wise algorithms
        Parameters
        ----------
        data_set : List,
        is_training : bool,
        """
        super(Data_sampler, self).__init__()
        self.batch_u, self.batch_is, self.batch_uj_s, self.batch_uj_sp, self.batch_j, self.batch_jp = batch
        self.batch_is = torch.LongTensor(self.batch_is)

    def __len__(self):
        return len(self.batch_u)

    def __getitem__(self, idx):
        u = self.batch_u[idx]
        i_s = self.batch_is[idx]
        uj_s = self.batch_uj_s[idx]
        uj_sp = self.batch_uj_sp[idx]
        j = self.batch_j[idx]
        jp = self.batch_jp[idx]

        return u, i_s, uj_s, uj_sp, j, jp

class Data_topk(data.Dataset):
    def __init__(self, batch):
        """
        Dataset formatter adapted pair-wise algorithms
        Parameters
        ----------
        data_set : List,
        is_training : bool,
        """
        super(Data_topk, self).__init__()
        self.features_fill = []
        for u,i_s,uj_s,j in batch:
            i_s = torch.LongTensor(i_s)
            self.features_fill.append([u,i_s,uj_s,j])

    def __len__(self):
        return len(self.features_fill)

    def __getitem__(self, idx):

        features = self.features_fill
        u = features[idx][0]
        i_s = features[idx][1]
        uj_s = features[idx][2]
        j = features[idx][3]

        return u, i_s, uj_s, j

class Data_property(data.Dataset):
    def __init__(self, batch):
        """
        Dataset formatter adapted pair-wise algorithms
        Parameters
        ----------
        data_set : List,
        is_training : bool,
        """
        super(Data_property, self).__init__()
        self.batch_u, self.batch_is, self.batch_uj_s, self.batch_uj_sp, self.batch_j, self.batch_jp, self.batch_user_property = batch
        self.batch_is = torch.LongTensor(self.batch_is)
        self.batch_user_property = torch.Tensor(np.array(self.batch_user_property))

    def __len__(self):
        return len(self.batch_u)

    def __getitem__(self, idx):
        u = self.batch_u[idx]
        i_s = self.batch_is[idx]
        uj_s = self.batch_uj_s[idx]
        uj_sp = self.batch_uj_sp[idx]
        j = self.batch_j[idx]
        jp = self.batch_jp[idx]
        u_prop = self.batch_user_property[idx]

        return u, i_s, uj_s, uj_sp, j, jp, u_prop

class Data_prop_topk(data.Dataset):
    def __init__(self, batch):
        """
        Dataset formatter adapted pair-wise algorithms
        Parameters
        ----------
        data_set : List,
        is_training : bool,
        """
        super(Data_prop_topk, self).__init__()
        self.features_fill = []
        for u,i_s,uj_s,j,u_prop in batch:
            i_s = torch.LongTensor(i_s)
            u_prop = torch.Tensor(np.array(u_prop))
            self.features_fill.append([u,i_s,uj_s,j,u_prop])

    def __len__(self):
        return len(self.features_fill)

    def __getitem__(self, idx):

        features = self.features_fill
        u = features[idx][0]
        i_s = features[idx][1]
        uj_s = features[idx][2]
        j = features[idx][3]
        u_prop = features[idx][4]

        return u, i_s, uj_s, j, u_prop
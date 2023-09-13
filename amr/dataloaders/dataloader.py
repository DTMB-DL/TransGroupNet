import importlib
from .prefetcher import *
from torch.utils.data import DataLoader
from .signaldataset import *
from .preprocess import *

__all__ = ['AMRDataLoader']


class AMRDataLoader(object):
    def __init__(self, dataset, Xmode, batch_size, num_workers, pin_memory, mod_type=[], ismulti=False):
        X_train, Y_train, Z_train, X_valid, Y_valid, Z_valid, X_test, Y_test, Z_test, self.snrs, self.mods = getattr(
            importlib.import_module("amr.dataloaders.dataloader_" + dataset), "SignalDataLoader")(mod_type)()

        if not ismulti:
            datapreprocess = DataPreprocess(Xmode)
        else:
            datapreprocess = []
            for idx in range(len(Xmode)):
                datapreprocess.append(DataPreprocess(Xmode[idx]))

        train_dataset = SignalDataset(X_train, Y_train, Z_train, datapreprocess, ismulti)
        valid_dataset = SignalDataset(X_valid, Y_valid, Z_valid, datapreprocess, ismulti)
        test_dataset = SignalDataset(X_test, Y_test, Z_test, datapreprocess, ismulti)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                       pin_memory=pin_memory, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers,
                                       pin_memory=pin_memory, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                      pin_memory=pin_memory, shuffle=False)

        if pin_memory:
            self.train_loader = PreFetcher(self.train_loader, ismulti)
            self.valid_loader = PreFetcher(self.valid_loader, ismulti)
            self.test_loader = PreFetcher(self.test_loader, ismulti)

    def __call__(self):
        return self.train_loader, self.valid_loader, self.test_loader, self.snrs, self.mods


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = AMRDataLoader("RML2016.10a_dict.pkl",{"type":"IQ","options":{"IQ_norm":False,"zero_mask":False}},batch_size=100,num_workers=4, pin_memory=False)()
    print(train_loader)
    print(valid_loader)
    print(test_loader)

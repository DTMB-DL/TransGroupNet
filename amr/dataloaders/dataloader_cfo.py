import random
import numpy as np
import h5py

__all__ = ["SignalDataLoader"]


class SignalDataLoader(object):
    def __init__(self, mod_type=[]):
        mods = ["BPSK", "QPSK", "8PSK", "PAM4", "QAM16", "QAM32", "QAM64", "QAM128", "QAM256", "GFSK", "WBFM", "AM-DSB",
                "AM-SSB", "OOK", "4ASK", "8ASK", "16PSK", "32PSK", "8APSK", "GMSK", "DQPSK", "16APSK", "32APSK",
                "64APSK", "128APSK"]  # "CPFSK"
        with h5py.File('dataset/cfo.hdf5', 'r+') as h5file:
            allX = np.asarray(h5file['X'][:])
            allY = np.asarray([mods[i] for i in h5file['mod'][:]])
            allZ = np.asarray(h5file['snr'][:])

        X = []
        Y = []
        Z = []
        for idx in range(allX.shape[0]):
            if allY[idx] in mod_type:
                X.append(allX[idx])
                Y.append(mod_type.index(allY[idx]))
                Z.append(allZ[idx])
        del allX
        del allY
        del allZ
        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)

        self.snrs = np.unique(Z).tolist()
        self.mods = mod_type

        n_examples = X.shape[0]
        n_train = int(0.5 * n_examples)
        n_valid = int(0.25 * n_examples)

        allnum = list(range(0, n_examples))
        random.shuffle(allnum)

        train_idx = allnum[0:n_train]
        valid_idx = allnum[n_train:n_train + n_valid]
        test_idx = allnum[n_train + n_valid:]
        self.X_train = X[train_idx]
        self.Y_train = Y[train_idx]
        self.Z_train = Z[train_idx]
        self.X_valid = X[valid_idx]
        self.Y_valid = Y[valid_idx]
        self.Z_valid = Z[valid_idx]
        self.X_test = X[test_idx]
        self.Y_test = Y[test_idx]
        self.Z_test = Z[test_idx]
        del X
        del Y
        del Z

    def __call__(self):
        return self.X_train, self.Y_train, self.Z_train, self.X_valid, self.Y_valid, self.Z_valid, self.X_test, self.Y_test, self.Z_test, self.snrs, self.mods
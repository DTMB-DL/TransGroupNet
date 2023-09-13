import torch
import numpy as np
import random
__all__ = ['normalize_IQ', 'get_amp_phase', 'get_apf1', 'get_apf2', 'zero_mask', 'get_iq_framed']


def normalize_IQ(t):  # 2*1024
    t_max = np.max(t)
    t_min = np.min(t)
    diff = t_max - t_min
    t = (t - t_min) / diff
    return t


def get_amp_phase(data):  # 2*1024
    signal_len = data.shape[-1]
    X_cmplx = data[0, :] + 1j * data[1, :]
    X_amp = np.abs(X_cmplx)
    X_ang = np.arctan2(data[1, :], data[0, :]) / np.pi
    X = np.stack((X_amp, X_ang), axis=0)
    X[0, :] = X[0, :] / np.linalg.norm(X[0, :], 2)
    return X


def get_apf1(data):  # 2*1024
    signal_len = data.shape[-1]
    X_cmplx = data[0, :] + 1j * data[1, :]
    X_amp = np.abs(X_cmplx)
    gamma = np.max(np.abs(np.fft.fftshift(np.fft.fft(X_amp/(np.mean(X_amp)-1), signal_len)))**2/signal_len)
    X_amp = X_amp/gamma
    X_ang = np.arctan2(data[1, :], data[0, :]) / np.pi
    X_tmp = np.unwrap(np.angle(X_cmplx))
    X_freq = np.hstack((X_tmp[0], np.diff(X_tmp))) / np.pi
    X = np.stack((X_amp, X_ang, X_freq), axis=0)
    return X


def get_apf2(data):  # 2*1024
    signal_len = data.shape[-1]
    X_cmplx = data[0, :] + 1j * data[1, :]
    X_amp = np.abs(X_cmplx)
    X_nc = X_amp/np.mean(X_amp)-1
    sigma = np.sqrt((X_nc**2).mean()-(np.abs(X_nc).mean())**2)
    X_amp = X_nc/sigma
    X_ang = np.arctan2(data[1, :], data[0, :]) / np.pi
    X_tmp = np.unwrap(np.angle(X_cmplx))
    X_freq = np.hstack((X_tmp[0], np.diff(X_tmp))) / np.pi
    X = np.stack((X_amp, X_ang, X_freq), axis=0)
    return X


def zero_mask(X_train, p=0.1):  # 2*1024
    num = int(X_train.shape[1] * p)
    res = X_train
    res[:, random.sample(range(X_train.shape[1]), num)] = 0
    return res


def get_iq_framed(data, L=32, R=16):
    X = normalize_IQ(data)
    # [2, 1024]
    Y = []
    for idx in range(0, X.shape[-1]-L+1, R):
        Y.append(X[:, idx:idx+L].reshape(-1))  # (2, L=32)
    Y = np.vstack(Y)  # (F, 2L) = (63, 64)  F=(1024-L)/R+1
    return Y

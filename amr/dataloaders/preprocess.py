import torch
from .transform import *
import numpy as np
__all__ = ['DataPreprocess']


class DataPreprocess(object):
    def __init__(self, Xmode):
        super(DataPreprocess, self).__init__()
        self.Xmode = Xmode
        self.Xtype = self.Xmode["type"]

    def datapreprocess(self, X):  # 根据Xmode定制批数据
        NX = X.copy()
        if ("IQ_norm" in self.Xmode["options"]) and self.Xmode["options"]["IQ_norm"]:
            X = normalize_IQ(X)
        if ("zero_mask" in self.Xmode["options"]) and self.Xmode["options"]["zero_mask"]:
            X = zero_mask(X)
        if self.Xmode["type"] == "IQ_framed":
            X = get_iq_framed(NX)
        if self.Xmode["type"] == "AP":
            X = get_amp_phase(NX)
        if self.Xmode["type"] == "APF":
            X = get_apf1(NX)
        if self.Xmode["type"] == "APF_ours":
            X = get_apf2(NX)

        return X

    def __call__(self, X):
        return self.datapreprocess(X)


if __name__ == "__main__":
    Xmode = {"type":"star","options":{"IQ_norm":True, "zero_mask":False, "img_size":[224,224]}}
    pre = DataPreprocess(Xmode)
    x = np.random.randn(3, 2, 20)
    pre(x)

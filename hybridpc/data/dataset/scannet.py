from hybridpc.data.dataset import GeneralDataset


class Scannet(GeneralDataset):
    def __int__(self, cfg, split):
        super().__init__(cfg, split)

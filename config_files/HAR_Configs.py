class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 9
        self.kernel_size = 8
        self.stride = 1

        self.num_classes = 6
        self.embed_dim = 64

        # training configs
        self.num_epoch = 40

        # optimizer parameters
        self.lr = 5e-4
        self.eta_min = 1e-5

        # data parameters
        self.drop_last = True
        self.batch_size = 128
        
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8

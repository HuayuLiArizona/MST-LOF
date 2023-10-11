
class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.num_classes = 5
        
        self.kernel_size = 25
        self.stride = 6
        
        self.embed_dim = 64

        # training configs
        self.num_epoch = 40
        
        self.lr = 5e-4
        self.eta_min = 1e-5

        # data parameters
        self.drop_last = True
        self.batch_size = 128
        
        self.augmentation = augmentations()

class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.5
        self.jitter_ratio = 2
        self.max_seg = 12

class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.kernel_size = 32
        self.stride = 4
        self.embed_dim = 64

        self.num_classes = 3
        
        # training configs
        self.num_epoch = 40
        self.batch_size = 128

        # optimizer parameters
        self.lr = 5e-4
        
        # data parameters
        self.drop_last = True
        
        self.augmentation = augmentations()

class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 2
        self.jitter_ratio = 0.1
        self.max_seg = 5
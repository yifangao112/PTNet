import torch
from training.nnUNetTrainerV2 import nnUNetTrainerV2
from network_architecture.utils import softmax_helper
from batchgenerators.utilities.file_and_folder_operations import *
from network_architecture.ptnet import PTNet


class PTNetTrainer(nnUNetTrainerV2):
    """
    PTNet launch
    """
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 200
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True
        self.oversample_foreground_percent = 0.33

    def initialize(self, training=True, force_load_plans=False, 
                   change_patch_size=None, change_pool_op_kernel_sizes=None,
                   change_conv_kernel_sizes=None, change_num_pool_per_axis=None):

        if change_pool_op_kernel_sizes is None:
            change_pool_op_kernel_sizes = [[1, 2, 2],
                                           [1, 2, 2],
                                           [2, 2, 2],
                                           [2, 2, 2],
                                           [2, 2, 2]]
            self.new_change_pool_op_kernel_sizes = change_pool_op_kernel_sizes

        super().initialize(training, force_load_plans, change_patch_size, change_pool_op_kernel_sizes,
                           change_conv_kernel_sizes, change_num_pool_per_axis)

    def initialize_network(self):
        
        pool_op_kernel_sizes = [[1, 2, 2],
                                [1, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2]]
        first_patch_size = [2, 4, 4]

        self.network = PTNet(input_channels=self.num_input_channels,
                                num_classes=self.num_classes,
                                crop_size=self.patch_size, 
                                pool_op_kernel_sizes=pool_op_kernel_sizes, 
                                patch_size=first_patch_size,
                                embed_share_weight=False)

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

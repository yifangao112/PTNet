from network_architecture.ptnet import PTNet
import torch


pool_op_kernel_sizes = [[1, 2, 2],
                        [1, 2, 2],
                        [2, 2, 2],
                        [2, 2, 2]]
patch_size = [2, 4, 4]

model = PTNet(crop_size=[8, 320, 320], 
              pool_op_kernel_sizes=pool_op_kernel_sizes, 
              patch_size=patch_size).cuda()

data = torch.randn(size=(1, 3, 8, 320, 320)).cuda()
output = model(data)
print(output.shape)
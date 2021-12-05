from torch import nn

from src.line_detection.modules.resnet_dcn import get_resnet_dcn


class ResDCN(nn.Module):
    def __init__(self, num_layer, num_classes):
        super(ResDCN, self).__init__()
        self.model = get_resnet_dcn(num_layer,
                                    heads={
                                        'hm': num_classes,
                                        'wh': 2,
                                        'reg': 2
                                    },
                                    head_conv=64)

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

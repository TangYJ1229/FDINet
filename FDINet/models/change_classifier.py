import torch
import torchvision
from models.layers import *
import torch.nn as nn


class Demo(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self._backbone = _get_backbone(bkbn_name="efficientnet_b4", pretrained=True, output_layer_bkbn="4", freeze_backbone=False)

        self.wavelet_skip = nn.ModuleList(
            [
                Wavelet_Skip(24),
                Wavelet_Skip(32),
                Wavelet_Skip(56),
                Wavelet_Skip(112),
            ]
        )

        self.length = len(self.wavelet_skip)

        self.FAM = nn.ModuleList(
            [
                First_FAM([32, 24]),
                Feature_Aggreation_Module([56, 32, 24]),
                Feature_Aggreation_Module([112, 56, 32]),
            ]
        )

        self.classifier = Classifier(24, out_channel)

    def dowasample(self, x1, x2):
        x1_downsample = []
        x2_downsample = []

        for num, layer in enumerate(self._backbone):
            x1 = layer(x1)
            x2 = layer(x2)
            if num != 0:
                x1_downsample.append(x1)
                x2_downsample.append(x2)

        return x1_downsample, x2_downsample

    def skip_connection(self, x1_downsample, x2_downsample):
        skip1 = []
        skip2 = []
        for i in range(len(self.wavelet_skip)):
            ws1 = self.wavelet_skip[i](x1_downsample[i], x2_downsample[i])
            ws2 = self.wavelet_skip[i](x2_downsample[i], x1_downsample[i])

            skip1.append(ws1)
            skip2.append(ws2)

        return skip1, skip2

    def decoder(self, skip):
        x = skip[-1]
        x = self.FAM[2](x, skip[2], skip[1])
        x = self.FAM[1](x, skip[1], skip[0])
        x = self.FAM[0](x, skip[0])
        return x

    def forward(self, x1, x2):
        x1_downsample, x2_downsample = self.dowasample(x1, x2)
        skip1, skip2 = self.skip_connection(x1_downsample, x2_downsample)

        pred1 = self.decoder(skip1)
        pred2 = self.decoder(skip2)
        pred1 = self.classifier(pred1)
        pred2 = self.classifier(pred2)

        return pred1, pred2


def _get_backbone(bkbn_name, pretrained, output_layer_bkbn, freeze_backbone):
    # The whole model:
    entire_model = getattr(torchvision.models, bkbn_name)(pretrained=pretrained).features

    # Slicing it:
    derived_model = nn.ModuleList([])
    for name, layer in entire_model.named_children():
        derived_model.append(layer)
        if name == output_layer_bkbn:
            break

    # Freezing the backbone weights:
    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False
    return derived_model


if __name__ == '__main__':
    x1 = torch.randn(1, 3, 256, 256).cuda()
    x2 = torch.randn(1, 3, 256, 256).cuda()
    net1 = Demo(3, 3).cuda()
    out,_ = net1(x1,x2)
    print(out.shape)
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(net1, (x1, x2))
    total = sum([param.nelement() for param in net1.parameters()])
    print("Params_Num: %.2fM" % (total/1e6))
    print(flops.total()/1e9)
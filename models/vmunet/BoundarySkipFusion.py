import torch
import torch.nn as nn
import torch.nn.functional as F


class BSF(nn.Module):
    """
    BSF模块，避免动态创建卷积以兼容FLOPs计算
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 预定义可能的通道调整卷积
        self.channel_adjust_384_to_192 = nn.Conv2d(384, 192, 1) if dim == 192 else None
        self.channel_adjust_192_to_96 = nn.Conv2d(192, 96, 1) if dim == 96 else None

        # 边界增强模块
        self.boundary_refinement = nn.Sequential(
            nn.Conv2d(dim + 1, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, 3, padding=1),
            nn.Sigmoid()
        )

        # 多尺度边界感知融合
        self.multi_scale_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, dilation=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=2, dilation=2),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=4, dilation=4),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            )
        ])

        # 特征融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim * 4, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1)
        )

    def forward(self, decoder_feat, encoder_feat, bca_weights):
        # 调整编码器通道数 - 使用预定义的卷积
        if encoder_feat.shape[1] != decoder_feat.shape[1]:
            if encoder_feat.shape[1] == 384 and decoder_feat.shape[
                1] == 192 and self.channel_adjust_384_to_192 is not None:
                encoder_feat = self.channel_adjust_384_to_192(encoder_feat)
            elif encoder_feat.shape[1] == 192 and decoder_feat.shape[
                1] == 96 and self.channel_adjust_192_to_96 is not None:
                encoder_feat = self.channel_adjust_192_to_96(encoder_feat)
            else:
                # 备用方案：使用平均池化调整通道
                if encoder_feat.shape[1] > decoder_feat.shape[1]:
                    encoder_feat = encoder_feat.mean(dim=1, keepdim=True).repeat(1, decoder_feat.shape[1], 1, 1)
                else:
                    encoder_feat = encoder_feat.repeat(1, decoder_feat.shape[1] // encoder_feat.shape[1], 1, 1)

        # 调整encoder_feat的空间维度以匹配decoder_feat
        if encoder_feat.shape[2:] != decoder_feat.shape[2:]:
            encoder_feat = F.interpolate(encoder_feat, size=decoder_feat.shape[2:], mode='bilinear',
                                         align_corners=False)

        # 调整bca_weights的空间维度以匹配decoder_feat
        if bca_weights.shape[2:] != decoder_feat.shape[2:]:
            bca_weights = F.interpolate(bca_weights, size=decoder_feat.shape[2:], mode='bilinear', align_corners=False)

        # 确保BCA权重是单通道
        if bca_weights.shape[1] != 1:
            bca_weights = bca_weights.mean(dim=1, keepdim=True)


        # 边界引导的特征选择
        bca_expanded = bca_weights.repeat(1, self.dim, 1, 1)
        boundary_guide_input = torch.cat([decoder_feat, bca_weights], dim=1)
        boundary_weights = self.boundary_refinement(boundary_guide_input)

        # 对编码器特征应用BCA边界权重
        boundary_enhanced_encoder = encoder_feat * bca_expanded

        # 多尺度边界感知
        multi_scale_features = [decoder_feat]
        for fusion_layer in self.multi_scale_fusion:
            scale_feat = fusion_layer(boundary_enhanced_encoder)
            multi_scale_features.append(scale_feat)

        # 特征融合
        fused_multi_scale = torch.cat(multi_scale_features, dim=1)
        output = self.fusion_conv(fused_multi_scale)

        # 边界权重引导的残差连接
        final_output = output * boundary_weights + decoder_feat

        return final_output
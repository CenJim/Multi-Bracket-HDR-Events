import torch
import torch.nn as nn
from ops.dcn import ModulatedDeformConvPack, modulated_deform_conv


# Define a Basic Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.match_channels = nn.Conv2d(in_channels, 64, kernel_size=1)  # 1x1卷积用于通道数匹配

    def forward(self, x):
        identity = self.match_channels(x)  # 使用1x1卷积调整x的通道数
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity  # Add the input x to the output of the conv2d
        out = self.relu(out)  # Apply ReLU activation after adding the identity
        return out


class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """

    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        #  seq_len=10 for moving_mnist
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],
                                self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)  # gates: S, num_features*4, H, W
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)


# Encoder Network
class Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()
        self.res_blocks = nn.Sequential(*[ResBlock(64) for _ in range(4)])
        self.down_sample = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.initial_conv = ResBlock(input_channels)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        l1 = x
        x = self.down_sample(x)
        l2 = x
        x = self.down_sample(x)
        l3 = x
        return [l1, l2, l3]


class LSTMEvent(nn.Module):
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(LSTMEvent, self).__init__()
        self.convLSTM = CLSTM_cell(shape, input_channels, filter_size, num_features)

    def forward(self, inputs, hidden_state, seq_len):
        hidden_outputs, (h_output, c_output) = self.convLSTM(inputs, hidden_state, seq_len)
        return h_output


# Reconstruction Module
class ReconstructionModule(nn.Module):
    def __init__(self):
        super(ReconstructionModule, self).__init__()
        self.res_blocks = nn.Sequential(*[ResBlock(64) for _ in range(10)])

    def forward(self, x):
        x = self.res_blocks(x)
        return x


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            print(f'Offset abs mean is {offset_absmean}, larger than 50.')

        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)


class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.

    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 3, num_feat, 3, 1,
                                                 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
                                                     1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3,
                                                     1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
                                                     1)
            self.dcn_pack[level] = DCNv2Pack(
                num_feat,
                num_feat,
                3,
                padding=1,
                deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
                                                  1)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(
            num_feat,
            num_feat,
            3,
            padding=1,
            deformable_groups=deformable_groups)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l, event_feat_l):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            event_feat_l (list[Tensor]): Event feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], event_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat(
                    [offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(
            self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat


class PairwiseAttention(nn.Module):
    def __init__(self, num_feat=64):
        super(PairwiseAttention, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.weights_cal = nn.Sequential(
            nn.Conv2d(num_feat * 2, 64, 3, 1, 1),
            self.lrelu,
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, exposure_list):
        """The first fusion method.

        Args:
            exposure_list (list[Tensor]): Exposure feature list. It
                contains three exposure version (I0, I-1, I1),
                each with shape (b, c, h, w).

        Returns:
            Tensor: Pairwise fused features.
        """
        under_exposure_weights = self.weights_cal(self.conv1(torch.cat([exposure_list[0], exposure_list[1]], dim=1)))
        reference_exposure_weights = self.weights_cal(
            self.conv1(torch.cat([exposure_list[0], exposure_list[0]], dim=1)))
        over_exposure_weights = self.weights_cal(self.conv1(torch.cat([exposure_list[0], exposure_list[2]], dim=1)))

        pairwise_fusion = under_exposure_weights * exposure_list[1] + reference_exposure_weights * exposure_list[
            0] + over_exposure_weights * exposure_list[2]

        return pairwise_fusion


class SpatialAttention(nn.Module):
    def __init__(self, num_feat=64, num_frame=3):
        super(SpatialAttention, self).__init__()
        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, pairwise_fusion, exposure_list):
        attn = self.lrelu(self.attn1(torch.cat([exposure_list[1], exposure_list[0], exposure_list[2]], dim=1)))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(
            self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(
            self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(
            self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = pairwise_fusion * attn * 2 + attn_add
        return feat


class EHDR_network(nn.Module):
    def __init__(self, event_shape, num_feat=64, num_frame=3):
        super(EHDR_network, self).__init__()
        self.frame_encoder = Encoder(input_channels=6)
        self.event_encoder = Encoder(input_channels=5)
        self.event_lstm = LSTMEvent(shape=event_shape, input_channels=5, filter_size=3, num_features=64)
        self.feature_alignment = PCDAlignment(num_feat=64)
        self.pairwise_attention = PairwiseAttention(num_feat=64)
        self.spatial_attention = SpatialAttention(num_feat=64, num_frame=3)
        self.reconstruction = ReconstructionModule()
        self.decode = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, reference, under_exposure, over_exposure, events_under, events_over):
        """The whole network

        :param reference: the reference exposure image, 6 channels
        :param under_exposure: the under exposure image, 6 channels
        :param over_exposure: the over exposure image, 6 channels
        :param events_under: shape=(b, num_voxel_grids, c, h, w)
        :param events_over: shape=(b, num_voxel_grids, c, h, w)
        :return: a 3 channel reconstructed image tensor
        """

        reference_feature = self.frame_encoder(reference)
        under_exposure_feature = self.frame_encoder(under_exposure)
        over_exposure_feature = self.frame_encoder(over_exposure)

        events_slices = []
        for i in range(events_under.size(1)):
            events_slices.append(self.event_encoder(events_under[:, i, :, :, :]))
        events_encoded = torch.stack(events_slices, dim=1)
        events_under_feature = self.event_lstm(events_encoded, seq_len=events_under.shape(1))

        events_slices = []
        for i in range(events_under.size(1)):
            events_slices.append(self.event_encoder(events_over[:, i, :, :, :]))
        events_encoded = torch.stack(events_slices, dim=1)
        events_over_feature = self.event_lstm(events_encoded, seq_len=events_over.shape(1))

        under_exposure_alignment = self.feature_alignment(under_exposure_feature, reference_feature,
                                                          events_under_feature)
        over_exposure_alignment = self.feature_alignment(over_exposure_feature, reference_feature, events_over_feature)
        exposure_list = [reference_feature, under_exposure_alignment, over_exposure_alignment]
        pairwise_fusion_feature = self.pairwise_attention(exposure_list)
        all_fusion_alignment = self.spatial_attention(pairwise_fusion_feature, exposure_list)
        reconstruction = self.reconstruction(all_fusion_alignment)
        output = self.decode(torch.cat([reconstruction, reference_feature], dim=1))

        return output

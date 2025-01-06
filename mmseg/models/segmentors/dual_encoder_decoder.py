# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional
from mmseg.registry import MODELS
from mmseg.utils import ConfigType, OptConfigType, OptMultiConfig
from .encoder_decoder import GuidedMix_EncoderDecoder
from torch import Tensor
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
@MODELS.register_module()
class DualGuidedMix_EncoderDecoder(GuidedMix_EncoderDecoder):
    """双网络版本的GuidedMix_EncoderDecoder,包含教师网络和学生网络"""
    
    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None, 
                 init_cfg: OptMultiConfig = None,
                 mitrans=None,
                 kl_loss=None,
                 components=None,
                 lambda_factor=0.5):
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
            mitrans=mitrans,
            kl_loss=kl_loss,
            components=components,
            lambda_factor=lambda_factor
        )
        
        # 创建教师网络
        self.teacher_net = GuidedMix_EncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=None,  # 教师网络使用不同的初始化
            init_cfg=init_cfg,
            mitrans=mitrans,
            kl_loss=kl_loss,
            components=components,
            lambda_factor=lambda_factor
        )

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """计算教师网络和学生网络的loss并取平均"""
        # 获取学生网络的loss
        student_losses = super().loss(inputs, data_samples)
        
        # 获取教师网络的loss
        teacher_losses = self.teacher_net.loss(inputs, data_samples)
        
        # 合并losses并取平均
        losses = {}
        for key in student_losses.keys():
            losses[key] = (student_losses[key] + teacher_losses[key]) * 0.5
            
        return losses

    def predict(self, inputs: Tensor, data_samples: OptSampleList = None) -> SampleList:
        """推理时只使用学生网络"""
        return super().predict(inputs, data_samples) 
# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
from mmseg.models import build_loss, build_neck, build_head
from ..utils import resize
import torch
import numpy as np

# torch.autograd.set_detect_anomaly(True)

@MODELS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

@MODELS.register_module()
class GuidedMix_EncoderDecoder(EncoderDecoder):
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
                 lambda_factor=0.5,
                 ):
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg
            )
        # if pretrained is not None:
        #     assert backbone.get('pretrained') is None, \
        #         'both backbone and segmentor set pretrained weight'
        #     backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        # if neck is not None:
        #     self.neck = MODELS.build(neck)
        # self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        if kl_loss is not None: 
            self.kl_loss = build_loss(kl_loss)
        if mitrans is not None: 
            self.mitrans = build_neck(mitrans)
        else:
            self.mitrans = None
            
        self.components = components
        self.decode_head_mix = build_head(decode_head)
        
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # assert self.with_decode_head
        self.outputs = dict()
        self.prepare_hook_modules()
        self.lambda_factor = lambda_factor
    
    # def extract_feat(self, inputs: Tensor) -> List[Tensor]:
    #     """Extract features from images."""
    #     x = self.backbone(inputs)
    #     if self.with_neck:
    #         x = self.neck(x)
    #     return x
        
    # def encode_decode(self, inputs: Tensor,
    #                   batch_img_metas: List[dict]) -> Tensor:
    #     """Encode images with backbone and decode into a semantic segmentation
    #     map of the same size as input."""
    #     x = self.extract_feat(inputs)
    #     seg_logits = self.decode_head.predict(x, batch_img_metas,
    #                                           self.test_cfg)

    #     return seg_logits
    
    def prepare_hook_modules(self):
        # Record the mapping relationship between modules and module
        # names.
        self.module2name = {}
        for name, module in self.named_modules():
            self.module2name[module] = name
        self.name2module = dict(self.named_modules())
        
        for component in self.components:
            module_names = self.components[component]
            
            for name in module_names:
                self.outputs[name] = list()
                self.name2module[name].register_forward_hook(self.module_forward_output_hook)
                
    # def _forward(self,
    #              inputs: Tensor,
    #              data_samples: OptSampleList = None) -> Tensor:
    #     """Network forward process.

    #     Args:
    #         inputs (Tensor): Inputs with shape (N, C, H, W).
    #         data_samples (List[:obj:`SegDataSample`]): The seg
    #             data samples. It usually includes information such
    #             as `metainfo` and `gt_sem_seg`.

    #     Returns:
    #         Tensor: Forward output of model without any post-processes.
    #     """
    #     x = self.extract_feat(inputs)
    #     return self.decode_head.forward(x)
    
    # def predict(self,
    #             inputs: Tensor,
    #             data_samples: OptSampleList = None) -> SampleList:
    #     """Predict results from a batch of inputs and data samples with post-
    #     processing.

    #     Args:
    #         inputs (Tensor): Inputs with shape (N, C, H, W).
    #         data_samples (List[:obj:`SegDataSample`], optional): The seg data
    #             samples. It usually includes information such as `metainfo`
    #             and `gt_sem_seg`.

    #     Returns:
    #         list[:obj:`SegDataSample`]: Segmentation results of the
    #         input images. Each SegDataSample usually contain:

    #         - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
    #         - ``seg_logits``(PixelData): Predicted logits of semantic
    #             segmentation before normalization.
    #     """
    #     if data_samples is not None:
    #         batch_img_metas = [
    #             data_sample.metainfo for data_sample in data_samples
    #         ]
    #     else:
    #         batch_img_metas = [
    #             dict(
    #                 ori_shape=inputs.shape[2:],
    #                 img_shape=inputs.shape[2:],
    #                 pad_shape=inputs.shape[2:],
    #                 padding_size=[0, 0, 0, 0])
    #         ] * inputs.shape[0]

    #     seg_logits = self.inference(inputs, batch_img_metas)

    #     return self.postprocess_result(seg_logits, data_samples)
                
    def module_forward_output_hook(self, module, inputs, outputs):
        """Save the module's forward output.
        Args:
            module (:obj:`torch.nn.Module`): The module to register hook.
            inputs (tuple): The input of the module.
            outputs (tuple): The output of the module.
        """
        if self.training:
            self.outputs[self.module2name[module]].append(
                outputs)
            
    def get_module_outputs(self, module_name):
        """Get the outputs according module name."""
        outputs = []
        for name in module_name:
            outputs.append(self.outputs[name])
        return outputs

    def reset_outputs(self, outputs):
        """Reset the teacher's outputs or student's outputs."""
        for key in outputs.keys():
            outputs[key] = list()
    
    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses
    
    def _second_decode_head_forward_train(self, img, unsup_img, sup_feats, unsup_feats, alpha=1.0):
        
        seg_logit_unlabeled_main = self.decode_head.forward(unsup_feats)
        
        # Get the teacher's outputs.
        module_names = self.components['out_modules']
        outputs = self.get_module_outputs(module_names)
        
        feats_aspp_labeled = self.decode_head._transform_inputs(sup_feats)
        feats_aspp_labeled = [
            resize(
                self.decode_head.image_pool(feats_aspp_labeled),
                size=feats_aspp_labeled.size()[2:],
                mode='bilinear',
                align_corners=self.decode_head.align_corners)
        ]
        feats_aspp_labeled = torch.cat(feats_aspp_labeled + outputs[1][0], 1)
        
        feats_aspp_unlabeled = self.decode_head._transform_inputs(unsup_feats)
        feats_aspp_unlabeled = [
            resize(
                self.decode_head.image_pool(feats_aspp_unlabeled),
                size=feats_aspp_unlabeled.size()[2:],
                mode='bilinear',
                align_corners=self.decode_head.align_corners)
        ]
        feats_aspp_unlabeled = torch.cat(feats_aspp_unlabeled + outputs[1][1], 1)
        
        assert img.shape[0] == unsup_img.shape[0]
        
        lam = np.random.beta(alpha, alpha)
        lam = min(lam, 1-lam)
        while lam >= self.lambda_factor:
            lam = np.random.beta(alpha, alpha)
            lam = min(lam, 1-lam)
            
        indices = torch.randperm(img.size(0), device=img.device, dtype=torch.long)
        feats_mixed = lam * feats_aspp_labeled + (1-lam) * feats_aspp_unlabeled[indices]
        if self.mitrans:
            feats_mixed = self.mitrans(feats_mixed)
            
        seg_logit_secondmain = self.decode_head_mix.bottleneck(feats_mixed)
        seg_logit_secondmain = self.decode_head_mix.cls_seg(seg_logit_secondmain)
        
        seg_logit_labeled_firstmain = outputs[0][0]
        seg_logit_unlabeled_decoupled_main = seg_logit_secondmain - \
            lam * seg_logit_labeled_firstmain
            
        loss_unlabeled_main = self.kl_loss(seg_logit_unlabeled_decoupled_main, \
            seg_logit_unlabeled_main.detach()[indices])
        
        losses = dict()
        losses['decode.loss_decouple_unlabeled_ce'] = loss_unlabeled_main
        
        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.reset_outputs(self.outputs)

        x = self.extract_feat(inputs)
        
        img = inputs[:int(inputs.shape[0]/2)]
        unsup_img = inputs[int(inputs.shape[0]/2):]
        
        sup_feats = [feat[:int(inputs.shape[0]/2)] for feat in x]
        unsup_feats = [feat[int(inputs.shape[0]/2):] for feat in x]
        
        sup_data_samples = data_samples[:int(inputs.shape[0]/2)]
        unsup_data_samples = data_samples[int(inputs.shape[0]/2):]

        losses = dict()

        loss_decode = self._decode_head_forward_train(sup_feats, sup_data_samples)
        
        losses.update(loss_decode)

        unsup_losses = self._second_decode_head_forward_train(img, unsup_img,  sup_feats, unsup_feats)
        
        losses['decode.loss_ce'] = losses['decode.loss_ce'] + unsup_losses['decode.loss_decouple_unlabeled_ce']
        
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)
            
        losses.update(unsup_losses)
        
        
        # -------------------------------------
        # self.reset_outputs(self.outputs)
        
        # x = self.extract_feat(inputs)

        # losses = dict()

        # loss_decode = self._decode_head_forward_train(x, data_samples)
        # losses.update(loss_decode)

        # if self.with_auxiliary_head:
        #     loss_aux = self._auxiliary_head_forward_train(x, data_samples)
        #     losses.update(loss_aux)
            
        return losses
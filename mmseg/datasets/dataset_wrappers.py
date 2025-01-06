# Copyright (c) OpenMMLab. All rights reserved.
import collections
import copy, os
from typing import List, Optional, Sequence, Union

from mmengine.dataset import ConcatDataset, force_full_init

from mmseg.registry import DATASETS, TRANSFORMS
from .voc import SemiPascalVOCDataset
import mmcv, glob
import numpy as np, math
import copy, os.path as osp

@DATASETS.register_module()
class MultiImageMixDataset:
    """A wrapper of multiple images mixed dataset.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup.

    Args:
        dataset (ConcatDataset or dict): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
    """

    def __init__(self,
                 dataset: Union[ConcatDataset, dict],
                 pipeline: Sequence[dict],
                 skip_type_keys: Optional[List[str]] = None,
                 lazy_init: bool = False) -> None:
        assert isinstance(pipeline, collections.abc.Sequence)

        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, ConcatDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`ConcatDataset` instance, but got {type(dataset)}')

        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = TRANSFORMS.build(transform)
                self.pipeline.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self._metainfo = self.dataset.metainfo
        self.num_samples = len(self.dataset)

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the multi-image-mixed dataset.

        Returns:
            dict: The meta information of multi-image-mixed dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        return self.dataset.get_data_info(idx)

    @force_full_init
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset[idx])
        for (transform, transform_type) in zip(self.pipeline,
                                               self.pipeline_types):
            if self._skip_type_keys is not None and \
                    transform_type in self._skip_type_keys:
                continue

            if hasattr(transform, 'get_indices'):
                indices = transform.get_indices(self.dataset)
                if not isinstance(indices, collections.abc.Sequence):
                    indices = [indices]
                mix_results = [
                    copy.deepcopy(self.dataset[index]) for index in indices
                ]
                results['mix_results'] = mix_results

            results = transform(results)

            if 'mix_results' in results:
                results.pop('mix_results')

        return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys.

        It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys

@DATASETS.register_module()
class SemiDataset:

    def __init__(self, dataset, unsup_dataset, pipeline=None, skip_type_keys=None):
        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        if pipeline is None:
            self.pipeline = pipeline
        else:
            self.pipeline = []
            self.pipeline_types = []
            for transform in pipeline:
                if isinstance(transform, dict):
                    self.pipeline_types.append(transform['type'])
                    transform = TRANSFORMS.build(transform)
                    self.pipeline.append(transform)
                else:
                    raise TypeError('pipeline must be a dict')
        
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        
        if isinstance(unsup_dataset, dict):
            self.unsup_dataset = DATASETS.build(unsup_dataset)

        self.sup_img_infos = self.dataset.img_infos

        self.unsup_dataset = self.unsup_dataset
        self.unsup_img_infos = self.unsup_dataset.img_infos
        
        self.dataset = self.dataset        
        self.CLASSES = self.dataset.CLASSES
        self.PALETTE = self.dataset.PALETTE
        self.num_samples = len(self.dataset)
        
        self.__gen_sample_data_infos()
        self._progress_in_iter = 1

    def __gen_sample_data_infos(self):
        self.num_samples = max(len(self.dataset), len(self.unsup_img_infos))
        if len(self.dataset) == self.num_samples:
            self._sup_img_infos = self.sup_img_infos
        else:
            repeat_times = math.ceil(self.num_samples / len(self.sup_img_infos))
            indices = np.random.permutation(np.arange(len(self.sup_img_infos)))

            temp = [self.sup_img_infos[idx] for idx in indices]
            self._sup_img_infos = (self.sup_img_infos + temp * repeat_times)[:self.num_samples]

        if len(self.unsup_img_infos) == self.num_samples:
            self._unsup_img_infos = self.unsup_img_infos
        else:
            repeat_times = math.ceil(self.num_samples / len(self.unsup_img_infos))

            indices = np.random.permutation(np.arange(len(self.unsup_img_infos)))
            temp = [self.unsup_img_infos[idx] for idx in indices]
            self._unsup_img_infos = (self.unsup_img_infos + temp * repeat_times)[:self.num_samples]
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_info = self._sup_img_infos[idx]
        ann_info = img_info['ann']
        results = dict(img_info=img_info, ann_info=ann_info)
        # results['img_path'] = os.path.join(self.dataset.img_dir, img_info['filename'])
        results['seg_fields'] = []
        results['img_prefix'] = self.dataset.img_dir
        results['seg_prefix'] = self.dataset.ann_dir
        results['img_path'] = os.path.join(self.dataset.img_dir, img_info['filename'])
        results['seg_map_path'] = os.path.join(self.dataset.ann_dir[:-1], img_info['ann']['seg_map'])
        results['reduce_zero_label']=False
        if self.dataset.custom_classes:
            results['label_map'] = self.dataset.label_map
        
        segmap_info = glob.glob(osp.join(results['seg_prefix'], img_info['ann']['seg_map']))
        assert len(segmap_info) > 0
        
        # results['seg_prefix'] =  '/'.join(segmap_info[0].split('/')[:-1]), for pascal voc
        # if isinstance(self.dataset, SemiCityscapesDataset):
        #     results['seg_prefix'] =  '/'.join(segmap_info[0].split('/')[:-2])
        if isinstance(self.dataset, SemiPascalVOCDataset):
            results['seg_prefix'] =  '/'.join(segmap_info[0].split('/')[:-1])
            
        results = self.dataset.pipeline(results)
        
        ## results of unsup dataset
        unsup_img_info = self._unsup_img_infos[idx]
        unsup_ann_info = unsup_img_info['ann']
        unsup_results = dict(img_info=unsup_img_info, ann_info=unsup_ann_info)
        
        unsup_results['seg_fields'] = []
        unsup_results['img_prefix'] = self.unsup_dataset.img_dir
        unsup_results['seg_prefix'] = self.unsup_dataset.ann_dir
        unsup_results['img_path'] = os.path.join(self.unsup_dataset.img_dir, unsup_img_info['filename'])
        unsup_results['seg_map_path'] = os.path.join(self.unsup_dataset.ann_dir[:-1]+'Aug', unsup_img_info['ann']['seg_map'])
        unsup_results['reduce_zero_label']=False    
        if self.unsup_dataset.custom_classes:
            unsup_results['label_map'] = self.unsup_dataset.label_map
        
        unsup_segmap_info = glob.glob(osp.join(unsup_results['seg_prefix'], unsup_results['ann_info']['seg_map']))
        assert len(unsup_segmap_info) > 0
        
        # if isinstance(self.dataset, SemiCityscapesDataset):
        #     unsup_results['seg_prefix'] =  '/'.join(unsup_segmap_info[0].split('/')[:-2])
        if isinstance(self.dataset, SemiPascalVOCDataset):
            unsup_results['seg_prefix'] =  '/'.join(unsup_segmap_info[0].split('/')[:-1])
            
        unsup_results = self.unsup_dataset.pipeline(unsup_results)
        unsup_results = {'unsup.'+k: v for k, v in unsup_results.items()}
        
        results.update(unsup_results)
        return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys.

        It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys
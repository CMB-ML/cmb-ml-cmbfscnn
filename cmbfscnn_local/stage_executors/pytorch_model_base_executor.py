from typing import List, Dict, Callable
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from omegaconf import DictConfig

from cmbml.core import (
    BaseStageExecutor, 
    Split,
    )

# from ..dataset import CMBMapDataset
# from ..dummymodel import DummyNeuralNetwork
from cmbfscnn_local.cmbfscnn_wrapper import make_cmbfscnn

from cmbml.torch.pytorch_model_handler import PyTorchModel  # Must be imported to get it registered
from cmbml.utils import make_instrument, Instrument, Detector


logger = logging.getLogger(__name__)


class BasePyTorchModelExecutor(BaseStageExecutor):
    dtype_mapping = {
        "float": torch.float32,
        "double": torch.float64
    }

    def __init__(self, cfg: DictConfig, stage_str) -> None:
        super().__init__(cfg, stage_str)
        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.n_dets = len(self.instrument.dets)
        self.nside = cfg.scenario.nside

    def choose_device(self, force_device=None) -> None:
        if force_device:
            self.device = force_device
        else:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

    def make_fn_template(self, split: Split, asset):
        context = dict(
            split=split.name,
            sim=self.name_tracker.sim_name_template,
            freq="{freq}"
        )
        with self.name_tracker.set_contexts(contexts_dict=context):
        # with self.name_tracker.set_context("split", split.name):
        #     # The following set_context is a bit hacky; we feed the template into itself so it is unchanged
        #     with self.name_tracker.set_context("sim", self.name_tracker.sim_name_template):

            this_path_pattern = str(asset.path)
        return this_path_pattern

    # TODO: Remove this? It's replaced in children classes
    def make_model(self):
        raise NotImplementedError("This method must be implemented in a child class.")
    #     logger.debug(f"Using {self.device} device")
    #     model = self.make_model(self.cfg)
    #     # logger.info(model)
    #     return model

    def match_data_precision(self, tensor):
        # TODO: Revisit
        # data_precision is the precision with which the data is written to file
        # model_precision is the precision with which the model is created
        # tensor is the loaded data
        # If the tensor precision doesn't match the models, convert it
        # If the tensor precision doesn't match data_precision... is there an issue?
        if self.model_precision == "float" and tensor.dtype is torch.float64:
            return tensor.float()
        if self.model_precision == "float" and tensor.dtype is torch.float32:
            return tensor
        else:
            message = f"BasePyTorchModelExecutor data conversion is partially implemented. Received from config model precision: {self.model_precision}, data precision: {self.data_precision}. Received a tensor with dtype: {tensor.dtype}."
            logger.error(message)
            raise NotImplementedError(message)


class BaseCMBFSCNNModelExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str) -> None:
        super().__init__(cfg, stage_str)

        # self.max_filters = cfg.model.network.max_filters
        
        # nside = cfg.scenario.nside
        input_channels = cfg.scenario.detector_freqs
        # kernels_size = cfg.model.network.kernels_size
        # strides = cfg.model.network.strides
        # mainActive = cfg.model.network.mainActive
        # finalActive = cfg.model.network.finalActive
        # finalBN = cfg.model.network.finalBN

        # self.unet_to_make = cfg.model.network.unet_to_make

        # input_c = len(input_channels)
        # sides = (nside, nside)

        self.cmbfscnn_level = cfg.model.cmbfscnn_level
        # in_channels = cfg.model.in_channels
        in_channels = len(input_channels)
        out_channels = cfg.model.out_channels
        n_feats = cfg.model.n_feats

        self.model_dict = dict(in_channels=in_channels,
                               out_channels=out_channels,
                               n_feats=n_feats)

    def try_model(self, model):
        dummy_input = torch.rand(1, self.n_dets, 4*self.nside, 3*self.nside, device=self.device)
        result = model(dummy_input)
        logger.debug(f"Output result size: {result.size()}")

    def make_model(self):
        return make_cmbfscnn(self.model_dict, self.cmbfscnn_level)

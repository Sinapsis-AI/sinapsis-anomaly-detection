# -*- coding: utf-8 -*-
import os
from typing import Literal

import numpy as np
import torch
import torchvision.transforms.v2 as T
from anomalib.deploy import TorchInferencer

from sinapsis_anomalib.templates.anomalib_base_inference import (
    AnomalibBaseInference,
    AnomalibInferenceAttributes,
)


class AnomalibTorchInferenceAttributes(AnomalibInferenceAttributes):
    """PyTorch-specific inference attribute configuration.

    Attributes:
        device (Literal["cuda", "cpu"]): Target device for inference (either 'cuda' or 'cpu').
    """

    device: Literal["cuda", "cpu"]


class AnomalibTorchInference(AnomalibBaseInference):
    """PyTorch-specific inference implementation for Anomalib models.

    Extends base inference to provide native PyTorch model execution.

    Usage example:

        agent:
        name: my_test_agent
        templates:
        - template_name: InputTemplate
        class_name: InputTemplate
        attributes: {}
        - template_name: AnomalibTorchInference
        class_name: AnomalibTorchInference
        template_input: InputTemplate
        attributes:
            model_path: 'path/to/model.pt'
            transforms: null
            device: 'cuda'
    """

    AttributesBaseModel = AnomalibTorchInferenceAttributes

    def get_inferencer(self) -> TorchInferencer:
        """Get PyTorch Inferencer instance.

        Returns:
            TorchInferencer: Inferencer instance with model loaded on specified device.
        """
        model_path: str = os.path.expanduser(self.attributes.model_path)
        return TorchInferencer(path=model_path, device=self.attributes.device)

    @staticmethod
    def postprocess_segmentation_mask(binary_mask: torch.TensorType, image: np.ndarray) -> np.ndarray:
        """Apply rescaling, squeezing and conversion from torch to numpy array format.

        Args:
            binary_mask (np.ndarray): Mask produced by TorchInferencer.
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Postprocessed mask.
        """
        height, width = image.shape[:2]
        rescaled_mask = torch.squeeze(T.Resize(size=[height, width])(binary_mask))
        return rescaled_mask.cpu().numpy().astype(np.uint8)

    def get_transformation_list(self) -> list[T.Transform]:
        """Construct the list of transformations for the TorchInferencer.

        Returns:
            list[T.Transform]: List of transformations.
        """
        transforms = [T.ToImage(), T.ToDtype(torch.float32, scale=True)]

        transforms.extend(self._convert_transform_names())
        return transforms

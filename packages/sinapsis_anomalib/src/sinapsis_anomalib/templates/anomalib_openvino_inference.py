# -*- coding: utf-8 -*-
import os
from typing import Literal

import cv2 as cv
import numpy as np
import torchvision.transforms.v2 as T
from anomalib.deploy import OpenVINOInferencer

from sinapsis_anomalib.templates.anomalib_base_inference import (
    AnomalibBaseInference,
    AnomalibInferenceAttributes,
)


class AnomalibOpenVINOInferenceAttributes(AnomalibInferenceAttributes):
    """OpenVINO-specific inference attribute configuration.

    Attributes:
        device (Literal["CPU", "GPU"]): Target hardware accelerator for inference.
            Must be either 'CPU' or 'GPU'.
        model_height (int): The image height expected by OV model.
        model_width (int): The image width expected by OV model.
    """

    device: Literal["CPU", "GPU"]
    model_height: int
    model_width: int


class AnomalibOpenVINOInference(AnomalibBaseInference):
    """OpenVINO-specific inference implementation for Anomalib models.

    Extends base inference to provide optimized model execution using OpenVINO toolkit.

    Usage example:

        agent:
        name: my_test_agent
        templates:
        - template_name: InputTemplate
        class_name: InputTemplate
        attributes: {}
        - template_name: AnomalibOpenVINOInference
        class_name: AnomalibOpenVINOInference
        template_input: InputTemplate
        attributes:
            model_path: 'path/to/model.xml'
            transforms: null
            device: CPU
            model_height: 256
            model_width: 256
    """

    AttributesBaseModel = AnomalibOpenVINOInferenceAttributes

    def get_inferencer(self) -> OpenVINOInferencer:
        """Initialize OpenVINO inferencer with model and metadata.

        Returns:
            OpenVINOInferencer: Inferencer instance with model and metadata loaded.
        """
        model_path: str = os.path.expanduser(self.attributes.model_path)

        return OpenVINOInferencer(
            path=model_path,
            device=self.attributes.device,
        )

    @staticmethod
    def postprocess_segmentation_mask(binary_mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Apply resizing and squeezing to mask.

        Args:
            binary_mask (np.ndarray): Mask produced by OpenVinoInfencer.
            image (np.ndarray): Original image.

        Returns:
            np.ndarray: Postprocessed mask.
        """
        height, width = image.shape[:2]
        binary_mask = np.squeeze(binary_mask).astype(np.uint8)
        return cv.resize(binary_mask, (width, height))

    def get_transformation_list(self) -> list[T.Transform]:
        """Construct the list of transformations for the OpenVinoInferencer.

        Returns:
            list[T.Transform]: List of transformations.
        """
        transforms = [T.ToImage(), T.Resize([self.attributes.model_height, self.attributes.model_width])]
        transforms.extend(self._convert_transform_names())
        return transforms

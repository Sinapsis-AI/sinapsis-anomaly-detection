# -*- coding: utf-8 -*-
from abc import abstractmethod
from copy import deepcopy

import cv2 as cv
import numpy as np
import torchvision.transforms.v2 as T
from anomalib.data.utils.label import LabelName
from anomalib.deploy import OpenVINOInferencer, TorchInferencer
from pydantic import Field
from pydantic.dataclasses import dataclass
from sinapsis_core.data_containers.annotations import (
    BoundingBox,
    ImageAnnotations,
    Segmentation,
)
from sinapsis_core.data_containers.data_packet import DataContainer, ImagePacket
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    TemplateAttributeType,
    UIPropertiesMetadata,
)


@dataclass(frozen=True)
class AnomalibInferenceKeys:
    """Constants representing keys used in Anomalib inference operations.

    Attributes:
        TRANSFORMS (str): Key for image transformation configuration
        TASK (str): Key for task type specification
        ANOMALOUS (str): Key representing anomalous/abnormal classification
        NORMAL (str): Key representing normal classification
    """

    TRANSFORMS: str = "transforms"
    TASK: str = "task"
    ANOMALOUS: str = "anomalous"
    NORMAL: str = "normal"


class AnomalibInferenceAttributes(TemplateAttributes):
    """Configuration attributes for Anomalib inference templates.

    Attributes:
        model_path (str): Path to the exported model file
        transforms (list[str] | None): Optional list of additional image transformation names to apply
        anomaly_area_threshold (float): The minimum area to be considered a valid anomaly detection. Defaults to 100.
    """

    model_path: str
    transforms: dict = Field(default_factory=dict)
    anomaly_area_threshold: float = 100


class AnomalibBaseInference(Template):
    """Base class for Anomalib model inference implementations.

    Provides common functionality for processing images through Anomalib models,
    handling classification, segmentation, and detection tasks.
    """

    AttributesBaseModel = AnomalibInferenceAttributes
    UiProperties = UIPropertiesMetadata(category="Anomalib", output_type=OutputTypes.IMAGE)

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.inferencer = self.get_inferencer()
        self.transform = self._build_transforms()

    @abstractmethod
    def get_inferencer(self) -> TorchInferencer | OpenVINOInferencer:
        """Initialize and return the appropriate inferencer instance. Subclasses must implement this method.

        Returns:
            TorchInferencer | OpenVINOInferencer: Either a TorchInferencer or OpenVINOInferencer instance
        """

    def _convert_transform_names(self) -> list[T.Transform]:
        """Convert transform names to actual transform callables.

        Returns:
            list[T.Transform]: List of corresponding transform callable objects
        """
        transforms = []
        for name, params in self.attributes.transforms.items():
            if hasattr(T, name):
                transform_class = getattr(T, name)
                if params:
                    transforms.append(transform_class(**params))
                else:
                    transforms.append(transform_class())

        return transforms

    @abstractmethod
    def get_transformation_list(self) -> list[T.Transform]:
        """Construct the list of transformation according to the specified inferencer.

        Returns:
            list[T.Transform]: List of transformations.
        """

    def _build_transforms(self) -> T.Compose:
        """Build the complete transform pipeline for image preprocessing.

        Returns:
            T.Compose: A composed transform pipeline
        """
        transforms = self.get_transformation_list()

        return T.Compose(transforms)

    @staticmethod
    @abstractmethod
    def postprocess_segmentation_mask(binary_mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Apply postprocessing operations to the generated mask according to the used inferenceer.

        Args:
            binary_mask (np.ndarray): Mask produced by inferencer.
            image (np.ndarray): Original image.

        Returns:
            np.ndarray: Post-processed mask.
        """

    def get_boxes_from_mask(self, np_mask: np.ndarray) -> list[list[float]]:
        """Produce a list of bounding boxes from the predicted segmentation mask.

        Args:
            np_mask (np.ndarray): Predicted binary mask.

        Returns:
            list[list[float]]: List of bboxes in [x, y, w, h] format.
        """
        contours, _ = cv.findContours(np_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            contours_poly = cv.approxPolyDP(contour, 3, True)
            x, y, w, h = cv.boundingRect(contours_poly)
            bbox_area = w * h
            if bbox_area > self.attributes.anomaly_area_threshold:
                boxes.append([x, y, w, h])
        return boxes

    def process_packet(self, data_packet: ImagePacket) -> None:
        """Process an individual image packet through the inference pipeline.

        Args:
            data_packet (ImagePacket): The image packet containing the image to process
        """
        image = deepcopy(data_packet.content)
        processed_image = self.transform(image)
        result = self.inferencer.predict(processed_image)

        label = result.pred_label
        label_str = AnomalibInferenceKeys.ANOMALOUS if label == LabelName.ABNORMAL else AnomalibInferenceKeys.NORMAL

        pred_score = result.pred_score * 100

        if result.pred_mask is None:
            annotations = [ImageAnnotations(label=label, label_str=label_str, confidence_score=pred_score)]
        else:
            np_mask = self.postprocess_segmentation_mask(result.pred_mask, image)

            boxes = self.get_boxes_from_mask(np_mask)

            annotations = self._create_detection_annotations(
                boxes=boxes, pred_score=pred_score, label=label, label_str=label_str, pred_mask=np_mask
            )

        data_packet.annotations = annotations

    @staticmethod
    def _create_classification_annotation(pred_score: float, label: str, label_str: str) -> ImageAnnotations:
        """Create classification annotation from prediction results.

        Args:
            pred_score (float): Confidence score of the prediction (0-100)
            label (str): Numeric label value
            label_str (str): String representation of the label

        Returns:
            ImageAnnotations: Object containing classification results
        """
        return ImageAnnotations(
            label=label,
            label_str=label_str,
            confidence_score=pred_score,
        )

    @staticmethod
    def _create_detection_annotations(
        boxes: list[list[float]],
        pred_score: float,
        label: int,
        label_str: str,
        pred_mask: np.ndarray,
    ) -> list[ImageAnnotations]:
        """Create detection annotations from prediction results.

        Args:
            boxes (list[list[float]]): Array of bounding box coordinates in xyxy format
            pred_score (float): Confidence score of the prediction (0-100)
            label (int): Numeric label value
            label_str (str): String representation of the label
            pred_mask (np.ndarray): Optional segmentation mask array

        Returns:
            list[ImageAnnotations]: List of objects containing detection results
        """
        annotations = []
        for box in boxes:
            x, y, w, h = box
            bbox = BoundingBox(x=x, y=y, w=w, h=h)

            box_mask = np.zeros_like(pred_mask, dtype=np.uint8)
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            box_mask[y1:y2, x1:x2] = np.where(pred_mask[y1:y2, x1:x2] > 0, 1, 0)

            segmentation = Segmentation(mask=box_mask)

            annotations.append(
                ImageAnnotations(
                    label=label,
                    label_str=label_str,
                    confidence_score=pred_score,
                    bbox=bbox,
                    segmentation=segmentation,
                )
            )
        return annotations

    def execute(self, container: DataContainer) -> DataContainer:
        """Process all images in the data container through the inference pipeline.

        Args:
            container (DataContainer): Input data container with images to process

        Returns:
            DataContainer: The processed data container with inference results
        """
        for image_packet in container.images:
            self.process_packet(image_packet)
        return container

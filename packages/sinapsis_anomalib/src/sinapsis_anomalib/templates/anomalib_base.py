# -*- coding: utf-8 -*-
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import torch
from anomalib import models as anomalib_models
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.utils.types import NORMALIZATION, THRESHOLD
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from pydantic import BaseModel, ConfigDict, Field
from pydantic.dataclasses import dataclass
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    TemplateAttributeType,
    UIPropertiesMetadata,
)
from sinapsis_core.template_base.dynamic_template import BaseDynamicWrapperTemplate, WrapperEntryConfig

from sinapsis_anomalib.helpers.config_factory import CallbackFactory, LoggerFactory
from sinapsis_anomalib.helpers.configs import FolderConfig
from sinapsis_anomalib.helpers.env_var_keys import ANOMALIB_ROOT_DIR
from sinapsis_anomalib.helpers.tags import Tags

EXCLUDED_MODELS = [
    "EfficientAd",
    "VlmAd",
    "Cfa",
    "Dfkde",
    "Fastflow",
    "Supersimplenet",
    "AiVad",
]


@dataclass(frozen=True)
class AnomalibKeys:
    """Constants for accessing Anomalib Engine configuration sections.

    Attributes:
        CALLBACKS (str): Key for callback configurations in Engine setup
        LOGGER (str): Key for logger configurations in Engine setup
    """

    CALLBACKS: str = "callbacks"
    LOGGER: str = "logger"


METRICS_TYPE: TypeAlias = list[str] | str | dict[str, dict[str, Any]]
PATH_TYPE: TypeAlias = str | Path | None
PATH_SEQUENCE_TYPE: TypeAlias = str | Path | Sequence[str | Path] | None
LOGGER_TYPE: TypeAlias = Logger | Iterable[Logger] | bool | None


class EngineConfig(BaseModel):
    """Pydantic model for Anomalib Engine configuration.

    Attributes:
        callbacks (list[Callback] | None): List of PyTorch Lightning callbacks
        normalization (NORMALIZATION | None): Input normalization configuration
        threshold (THRESHOLD | None): Anomaly threshold configuration
        image_metrics (METRICS_TYPE | None): Image-level evaluation metrics
        pixel_metrics (METRICS_TYPE | None): Pixel-level evaluation metrics
        logger (LOGGER_TYPE): Logger configuration
        default_root_dir (PATH_TYPE): Root directory for outputs
        callback_configs (dict[str, dict[str, Any]] | None): Callback configurations
        logger_configs (dict[str, dict[str, Any]] | None): Logger configurations
    """

    callbacks: list[Callback] | None = None
    normalization: NORMALIZATION | None = None
    threshold: THRESHOLD | None = None
    image_metrics: METRICS_TYPE | None = None
    pixel_metrics: METRICS_TYPE | None = None
    logger: LOGGER_TYPE = None
    default_root_dir: PATH_TYPE = ANOMALIB_ROOT_DIR
    callback_configs: dict[str, dict[str, Any]] | None = None
    logger_configs: dict[str, dict[str, Any]] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class HasSUFFIX(Protocol):
    """Protocol for classes that have a SUFFIX attribute.

    Attributes:
        SUFFIX (str): Class attribute suffix string
    """

    SUFFIX: str


class DynamicWrapperEntry:
    """Descriptor that dynamically generates WrapperEntryConfig based on owner's SUFFIX."""

    def __get__(self, _instance: object, owner: type[HasSUFFIX]) -> WrapperEntryConfig:
        """Dynamically create WrapperEntryConfig based on the SUFFIX value.

        Args:
            _instance (object): Unused instance reference
            owner (type[HasSUFFIX]): Owning class with SUFFIX attribute

        Returns:
            WrapperEntryConfig: Configured wrapper entry
        """
        return WrapperEntryConfig(
            wrapped_object=anomalib_models,
            template_name_suffix=owner.SUFFIX,
            exclude_module_atts=EXCLUDED_MODELS,
        )


class AnomalibBaseAttributes(TemplateAttributes):
    """Base attributes for Anomalib model templates.

    Attributes:
        folder_attributes (FolderConfig): Configuration for Folder datamodule. Required for training, optional
            for export.
        callbacks (list[Callback] | None): Lightning callbacks
        normalization (NORMALIZATION | None): Input normalization
        threshold (THRESHOLD | None): Prediction threshold
        image_metrics (METRICS_TYPE | None): Image metrics
        pixel_metrics (METRICS_TYPE | None): Pixel metrics
        logger (LOGGER_TYPE): Lightning logger
        callback_configs (dict[str, dict[str, Any]] | None): Callback configs
        logger_configs (dict[str, dict[str, Any]] | None): Logger configs
    """

    folder_attributes: FolderConfig = Field(default_factory=FolderConfig)
    callbacks: list[Callback] | None = None
    normalization: NORMALIZATION | None = None
    threshold: THRESHOLD | None = None
    image_metrics: METRICS_TYPE | None = None
    pixel_metrics: METRICS_TYPE | None = None
    logger: LOGGER_TYPE = None
    callback_configs: dict[str, dict[str, Any]] | None = None
    logger_configs: dict[str, dict[str, Any]] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AnomalibBase(BaseDynamicWrapperTemplate):
    """Base class for Anomalib model Train and Export templates.

    Notes:
        - Subclasses must override SUFFIX as needed for their specific purpose
        - When using 'Train' suffix, all essential Folder attributes must be provided
        - INT8_PTQ/INT8_ACQ compression requires complete Folder configuration
        - Callback and logger configurations are optional but must follow Anomalib specs
    """

    SUFFIX: str = "Wrapper"
    AttributesBaseModel = AnomalibBaseAttributes
    UIProperties = UIPropertiesMetadata(
        category="Anomalib",
        output_type=OutputTypes.IMAGE,
        tags=[Tags.ANOMALIB, Tags.ANOMALY_DETECTION, Tags.DYNAMIC, Tags.MODELS],
    )
    WrapperEntry = DynamicWrapperEntry()

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes=attributes)
        self.model = self.wrapped_callable
        self.callback_factory = CallbackFactory()
        self.logger_factory = LoggerFactory()
        self.engine = self.setup_engine()

    def _create_callbacks(self) -> list[Callback]:
        """Create callbacks from configuration.

        Returns:
            list[Callback]: Instantiated callback objects
        """
        return [self.callback_factory.create(name, config) for name, config in self.attributes.callback_configs.items()]

    def _create_loggers(self) -> list[Logger]:
        """Create loggers from configuration.

        Returns:
            list[Logger]: Instantiated logger objects
        """
        return [self.logger_factory.create(name, config) for name, config in self.attributes.logger_configs.items()]

    def setup_engine(self) -> Engine:
        """Configure and initialize the Anomalib Engine.

        Returns:
            Engine: Configured engine instance
        """
        engine_kwargs = EngineConfig(**self.attributes.model_dump()).model_dump(
            exclude={"callback_configs", "logger_configs"}, exclude_none=True
        )
        if self.attributes.callback_configs:
            engine_kwargs[AnomalibKeys.CALLBACKS] = self._create_callbacks()

        if self.attributes.logger_configs:
            engine_kwargs[AnomalibKeys.LOGGER] = self._create_loggers()

        return Engine(**engine_kwargs)

    def setup_data_loader(self) -> Folder:
        """Initialize the data loader from folder attributes if provided.

        Returns:
            Folder: Configured data module instance
        """
        if self.attributes.folder_attributes is None:
            raise ValueError("'folder_attributes' must be provided to set up the data loader.")
        return Folder(**self.attributes.folder_attributes.model_dump(exclude_none=True))

    @abstractmethod
    def execute(self, container: DataContainer) -> DataContainer:
        """Template method to be implemented by subclasses.

        Args:
            container (DataContainer): Input data container

        Returns:
            DataContainer: Processed data container
        """

    def reset_state(self, template_name: str | None = None) -> None:
        """Re-instantiates the template.

        Args:
            template_name (str | None, optional): The name of the template instance being reset. Defaults to None.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().reset_state(template_name)

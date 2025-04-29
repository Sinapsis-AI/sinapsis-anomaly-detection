# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Literal

from anomalib.engine.engine import _TrainerArgumentsCache
from pydantic import Field
from pydantic.dataclasses import dataclass
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import TemplateAttributeType
from sinapsis_core.template_base.dynamic_template_factory import make_dynamic_template
from sinapsis_core.utils.env_var_keys import SINAPSIS_BUILD_DOCS

from sinapsis_anomalib.templates.anomalib_base import (
    AnomalibBase,
    AnomalibBaseAttributes,
)


@dataclass(frozen=True)
class AnomalibTrainKeys:
    """Constants for accessing training-related configuration keys.

    Attributes:
        ACCELERATOR (str): Key for accessing accelerator field in trainer args dict.
        DEVICES: (str): Key for accessing devices field in trainer args dict.
        CALLBACK_METRICS (str): Key for accessing callback metrics ('callback_metrics')
        MAX_EPOCHS (str): Key for maximum epochs setting ('max_epochs')
        CKPT_PATH (str): Key for checkpoint path ('ckpt_path')
        BEST_MODEL_PATH (str): Key for best model path ('best_model_path')
    """

    ACCELERATOR: str = "accelerator"
    CALLBACK_METRICS: str = "callback_metrics"
    MAX_EPOCHS: str = "max_epochs"
    CKPT_PATH: str = "ckpt_path"
    BEST_MODEL_PATH: str = "best_model_path"


@dataclass(frozen=True, slots=True)
class AnomalibTrainDataClass:
    """Container for training results and artifacts.

    Attributes:
        metrics (dict[str, float]): Dictionary of training metrics (metric_name: value)
        checkpoint_path (Path | str | None): Path to the best model checkpoint,
            or None if no checkpoint was saved
    """

    metrics: dict[str, float]
    checkpoint_path: Path | str | None


class AnomalibTrainAttributes(AnomalibBaseAttributes):
    """Training-specific configuration attributes.

    Attributes:
        max_epochs (int | None): Maximum number of training epochs.
            If None, uses default from model configuration.
        accelerator: (Literal["cpu", "gpu", "tpu", "hpu", "auto"]): Define the device to be used during training.
            Defaults to "cpu".
        ckpt_path (str | Path | None): Path to checkpoint for resuming training.
            If None, starts training from scratch.
        trainer_args: (dict[str, Any]): General trainer arguments. For more details see:
            https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags
    """

    max_epochs: int | None = None
    accelerator: Literal["cpu", "gpu", "tpu", "hpu", "auto"] = "cpu"
    ckpt_path: str | Path | None = None
    trainer_args: dict[str, Any] = Field(default_factory=dict)


class AnomalibTrain(AnomalibBase):
    """Training implementation for Anomalib models.

    Usage example:

        agent:
        name: my_test_agent
        templates:
        - template_name: InputTemplate
        class_name: InputTemplate
        attributes: {}
        - template_name: CfaTrain
        class_name: CfaTrain
        template_input: InputTemplate
        attributes:
            folder_attributes_config_path: 'path/to/config.yaml'
            generic_key: 'my_generic_key'
            callbacks: null
            normalization: null
            threshold: null
            image_metrics: null
            pixel_metrics: null
            logger: null
            default_root_dir: null
            callback_configs: null
            logger_configs: null
            max_epochs: null
            ckpt_path: null
            accelerator: gpu
            trainer_args:
                devices: "0"
            cfa_init:
                backbone: wide_resnet50_2
                gamma_c: 1
                gamma_d: 1
                num_nearest_neighbors: 3
                num_hard_negative_features: 3
                radius: 1.0e-05
    For a full list of options use the sinapsis cli: sinapsis info --all-template-names.
    If you want to see all available models, please visit:
    https://anomalib.readthedocs.io/en/v1.2.0/markdown/guides/reference/models/image/index.html.
    """

    SUFFIX = "Train"
    AttributesBaseModel = AnomalibTrainAttributes

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.data_module = self.setup_data_loader()

    def _update_trainer_args(self) -> None:
        """Updates the trainer configuration with current settings.

        Specifically sets the maximum number of training epochs
        based on the attributes configuration.
        """
        existing_args = self.engine._cache.args
        existing_args[AnomalibTrainKeys.MAX_EPOCHS] = self.attributes.max_epochs

        self.attributes.trainer_args[AnomalibTrainKeys.ACCELERATOR] = self.attributes.accelerator
        existing_args.update(self.attributes.trainer_args)
        self.engine._cache = _TrainerArgumentsCache(**existing_args)

    def _get_training_metrics(self) -> dict[str, Any]:
        """Extracts training metrics from the model.

        Returns:
            dict[str, float]: Dictionary of metric names and their values.
                Returns empty dict if no metrics available.
        """
        if not hasattr(self.engine.trainer, AnomalibTrainKeys.CALLBACK_METRICS):
            return {}
        return {k: v.item() if hasattr(v, "item") else v for k, v in self.engine.trainer.callback_metrics.items()}

    def train_model(self) -> AnomalibTrainDataClass:
        """Executes model training process.

        Returns:
            AnomalibTrainDataClass: Contains:
                - metrics: Collected training metrics
                - checkpoint_path: Path to best model checkpoint

        Note:
            - Updates trainer configuration before starting
            - Uses provided checkpoint if available for resuming
        """
        self._update_trainer_args()

        self.engine.train(
            model=self.model,
            datamodule=self.data_module,
            ckpt_path=self.attributes.ckpt_path,
        )

        metrics = self._get_training_metrics()
        checkpoint_path = (
            self.engine.best_model_path if hasattr(self.engine, AnomalibTrainKeys.BEST_MODEL_PATH) else None
        )

        return AnomalibTrainDataClass(metrics=metrics, checkpoint_path=checkpoint_path)

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes training and stores results.

        Args:
            container: Input data container

        Returns:
            Modified container with training results stored under generic_key
        """
        result = self.train_model()
        self._set_generic_data(container, result)
        return container


def __getattr__(name: str) -> Template:
    """Creat dynamic templates.

    Only create a template if it's imported, this avoids creating all the base models for all templates
    and potential import errors due to not available packages.
    """
    if name in AnomalibTrain.WrapperEntry.module_att_names:
        return make_dynamic_template(name, AnomalibTrain)
    raise AttributeError(f"template `{name}` not found in {__name__}")


__all__ = AnomalibTrain.WrapperEntry.module_att_names

if SINAPSIS_BUILD_DOCS:
    dynamic_templates = [__getattr__(template_name) for template_name in __all__]
    for template in dynamic_templates:
        globals()[template.__name__] = template
        del template

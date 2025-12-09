# -*- coding: utf-8 -*-
import os
from pathlib import Path

from anomalib.deploy import CompressionType, ExportType
from pydantic.dataclasses import dataclass
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import TemplateAttributeType
from sinapsis_core.template_base.dynamic_template_factory import make_dynamic_template
from sinapsis_core.utils.env_var_keys import SINAPSIS_BUILD_DOCS
from torchmetrics import Metric

from sinapsis_anomalib.helpers.configs import OpenVINOArgs
from sinapsis_anomalib.helpers.env_var_keys import ANOMALIB_ROOT_DIR
from sinapsis_anomalib.helpers.tags import Tags
from sinapsis_anomalib.templates.anomalib_base import (
    AnomalibBase,
    AnomalibBaseAttributes,
)
from sinapsis_anomalib.templates.anomalib_train import AnomalibTrainDataClass

AnomalibExportUIProperties = AnomalibBase.UIProperties
AnomalibExportUIProperties.tags.extend([Tags.EXPORT])


@dataclass(frozen=True, slots=True)
class AnomalibExportDataClass:
    """Container for export results.

    Attributes:
        exported_model_path (Path | str): Path to the exported model file(s).
            Can be either a Path object or string path.
    """

    exported_model_path: Path | str


class AnomalibExportAttributes(AnomalibBaseAttributes):
    """Export-specific attribute configuration for Anomalib models.

    Attributes:
        export_type (ExportType | str): Target export format. Defaults to TORCH.
        export_root (str | Path | None): Root directory for exported files.
        input_size (tuple[int, int] | None): Expected input dimensions (height, width).
        compression_type (CompressionType | None): Quantization/compression method.
        metric (Metric | str | None): Metric used for compression calibration.
        ov_args (dict[str, Any] | None): OpenVINO-specific export arguments.
        ckpt_path (str | None): Explicit path to model checkpoint.
        generic_key_chkpt (str | None): Key to retrieve training results.
    """

    folder_attributes: dict | None = None
    export_type: ExportType = ExportType.TORCH
    export_root: str | Path | None = None
    input_size: tuple[int, int] | None = None
    compression_type: CompressionType | None = None
    metric: Metric | str | None = None
    ov_args: OpenVINOArgs | None = None
    ckpt_path: str | None = None
    generic_key_chkpt: str | None = None


class AnomalibExport(AnomalibBase):
    """Export functionality for trained Anomalib models.

    Usage example:

        agent:
        name: my_test_agent
        templates:
        - template_name: InputTemplate
        class_name: InputTemplate
        attributes: {}
        - template_name: CfaExport
        class_name: CfaExport
        template_input: InputTemplate
        attributes:
            folder_attributes_config_path: null
            generic_key: 'my_generic_key'
            callbacks: null
            normalization: null
            threshold: null
            task: null
            image_metrics: null
            pixel_metrics: null
            logger: null
            callback_configs: null
            logger_configs: null
            export_type: 'openvino'
            export_root: null
            input_size: null
            transform: null
            compression_type: null
            metric: null
            ov_args: null
            ckpt_path: null
            generic_key_chkpt: null
            cfa_init:
                backbone: wide_resnet50_2
                gamma_c: 1
                gamma_d: 1
                num_nearest_neighbors: 3
                num_hard_negative_features: 3
                radius: 1.0e-05

    For a full list of options use the sinapsis cli: sinapsis info --all-template-names
    If you want to see all available models, please visit:
    https://anomalib.readthedocs.io/en/v1.2.0/markdown/guides/reference/models/image/index.html

    Notes:
        - Supports multiple export formats via ExportType
        - Enables model compression through CompressionType.
        - Can load models from checkpoints or previous training results.
    """

    AttributesBaseModel = AnomalibExportAttributes
    SUFFIX = "Export"
    UIProperties = AnomalibExportUIProperties

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.data_module = None
        if self.attributes.compression_type in (
            CompressionType.INT8_ACQ,
            CompressionType.INT8_PTQ,
        ):
            if self.attributes.folder_attributes is None:
                raise ValueError(
                    f"Compression type '{self.attributes.compression_type.value}' requires "
                    "'folder_attributes' to be provided."
                )
            self.data_module = self.setup_data_loader()

    def _get_checkpoint_path(self, container: DataContainer) -> str | Path:
        """Resolves the checkpoint path for model export.

        Args:
            container (DataContainer): Container with potential training results

        Raises:
            ValueError: If no valid checkpoint path is found

        Returns:
            str | Path : Path to model checkpoint

        Note:
            Priority order:
            1. Explicit ckpt_path from attributes
            2. Training results from container (if generic_key_chkpt provided)
        """
        if self.attributes.ckpt_path:
            return self.attributes.ckpt_path

        generic_data = self._get_generic_data(container, self.attributes.generic_key_chkpt)
        if generic_data and isinstance(generic_data, AnomalibTrainDataClass):
            return generic_data.checkpoint_path
        if generic_data and isinstance(generic_data, dict):
            return generic_data.get("checkpoint_path")
        raise ValueError("No checkpoint path found")

    def export_model(self, container: DataContainer) -> AnomalibExportDataClass:
        """Exports the model to specified format.

        Args:
            container (DataContainer): Input data container

        Returns:
            AnomalibExportDataClass: Contains exported model path
        """
        ckpt_path = self._get_checkpoint_path(container)
        export_dir = (
            os.path.join(ANOMALIB_ROOT_DIR, self.attributes.export_root)
            if self.attributes.export_root
            else ANOMALIB_ROOT_DIR
        )

        exported_path = self.engine.export(
            model=self.model,
            export_type=self.attributes.export_type,
            export_root=export_dir,
            input_size=self.attributes.input_size,
            compression_type=self.attributes.compression_type,
            datamodule=self.data_module,
            metric=self.attributes.metric,
            ov_args=self.attributes.ov_args.model_dump(exclude_none=True) if self.attributes.ov_args else None,
            ckpt_path=ckpt_path,
        )

        return AnomalibExportDataClass(exported_model_path=str(exported_path))

    def execute(self, container: DataContainer) -> DataContainer:
        """Performs model export and stores results.

        Args:
            container (DataContainer): Input data container

        Returns:
            DataContainer: Container with export results stored as generic data
        """
        result = self.export_model(container)
        self._set_generic_data(container, result)
        return container


def __getattr__(name: str) -> Template:
    """Creat dynamic templates.

    Only create a template if it's imported, this avoids creating all the base models for all templates
    and potential import errors due to not available packages.
    """
    if name in AnomalibExport.WrapperEntry.module_att_names:
        return make_dynamic_template(name, AnomalibExport)
    raise AttributeError(f"template `{name}` not found in {__name__}")


__all__ = AnomalibExport.WrapperEntry.module_att_names

if SINAPSIS_BUILD_DOCS:
    dynamic_templates = [__getattr__(template_name) for template_name in __all__]
    for template in dynamic_templates:
        globals()[template.__name__] = template
        del template

# -*- coding: utf-8 -*-
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from anomalib.data.utils import TestSplitMode, ValSplitMode
from pydantic import BaseModel, ConfigDict


class FolderConfig(BaseModel):
    """Configuration for the Anomalib Folder datamodule.

    This model defines parameters for loading image data from a directory structure.
    It is flexible and allows any other valid `Folder` arguments
    (like `train_augmentations`) to be passed through.

    Attributes:
        name (str): A descriptive name for the dataset.
        root (str | Path | None): The root directory where the dataset is located.
        normal_dir (str | Path | Sequence[str | Path]): Path to the directory containing normal images.
        abnormal_dir (str | Path | Sequence[str | Path] | None): Path to the directory for abnormal images.
        train_batch_size (int): The number of samples per batch for training. Defaults to 32.
        eval_batch_size (int): The number of samples per batch for evaluation. Defaults to 32.
        num_workers (int): The number of subprocesses to use for data loading. Defaults to 8.
        seed (int | None): A random seed for reproducibility in data splitting.
    """

    name: str
    root: str | Path | None = None
    normal_dir: str | Path | Sequence[str | Path]
    abnormal_dir: str | Path | Sequence[str | Path] | None = None
    normal_test_dir: str | Path | Sequence[str | Path] | None = None
    mask_dir: str | Path | Sequence[str | Path] | None = None
    normal_split_ratio: float = 0.2
    extensions: list[str] | None = None
    train_batch_size: int = 32
    eval_batch_size: int = 32
    num_workers: int = 8
    test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR
    test_split_ratio: float = 0.2
    val_split_mode: ValSplitMode | str = ValSplitMode.FROM_TEST
    val_split_ratio: float = 0.5
    seed: int | None = None
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class TrainerConfig(BaseModel):
    """Defines and validates parameters for the Trainer.

    This model is flexible and allows any other valid Trainer arguments to be passed.

    Attributes:
        devices (int | list[int] | str | None): The devices to use for training (e.g., 1, [0, 1], "auto").
            Defaults to "auto".
        accelerator (Literal["gpu", "cpu", "auto"]): The hardware accelerator to use.
            Defaults to "auto".
        min_epochs (int): The minimum number of epochs to train for. Defaults to 1.
        max_epochs (int): The maximum number of epochs to train for. Defaults to 5.
    """

    devices: int | list[int] | str | None = "auto"
    accelerator: Literal["cpu", "gpu", "tpu", "hpu", "auto"] = "cpu"
    min_epochs: int = 1
    max_epochs: int | None = 5
    model_config = ConfigDict(extra="allow")


class OpenVINOArgs(BaseModel):
    """Defines and validates parameters for OpenVINO model export.

    This model is flexible and allows any other valid arguments for `openvino.save_model`.
    For a full list of options, see the OpenVINO documentation.

    Attributes:
        compress_to_fp16 (bool | None): If True, compresses the model weights to FP16 precision.
            Defaults to None.
        compress_to_fp16 (bool | None): Print detailed information about conversion.
    """

    compress_to_fp16: bool | None = None
    verbose: bool = False
    model_config = ConfigDict(extra="allow")

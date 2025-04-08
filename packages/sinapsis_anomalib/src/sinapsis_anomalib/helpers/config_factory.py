# -*- coding: utf-8 -*-
from typing import Any

from anomalib.loggers import (
    AnomalibCometLogger,
    AnomalibMLFlowLogger,
    AnomalibTensorBoardLogger,
    AnomalibWandbLogger,
)
from lightning.pytorch.callbacks import (
    Callback,
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import (
    CometLogger,
    MLFlowLogger,
    TensorBoardLogger,
    WandbLogger,
)


class CallbackFactory:
    """A factory class for creating and managing PyTorch Lightning callbacks.

    This class provides a centralized way to create and register callbacks for use in PyTorch Lightning
    training workflows. It supports built-in callbacks like `RichProgressBar`, `EarlyStopping`, and
    `ModelCheckpoint`, and allows users to register custom callbacks.
    """

    def __init__(self) -> None:
        self.callbacks = {
            "rich_progress": RichProgressBar,
            "early_stopping": EarlyStopping,
            "model_checkpoint": ModelCheckpoint,
            "lr_monitor": LearningRateMonitor,
            "device_stats": DeviceStatsMonitor,
        }

    def create(self, callback_name: str, config: dict[str, Any] | None = None) -> Callback:
        """Creates an instance of the specified callback.

        Args:
            callback_name (str): The name of the callback to create. Must be a key in the `callbacks` dictionary.
            config (dict[str, Any] | None, optional): A dictionary of configuration parameters for the callback.
                If provided, these parameters will be passed to the callback's constructor. Defaults to None.

        Raises:
            ValueError: If the specified `callback_name` is not found in the `callbacks` dictionary.

        Returns:
            Callback: An instance of the requested callback.
        """
        callback_class = self.callbacks.get(callback_name)
        if not callback_class:
            raise ValueError(f"Callback {callback_name} not supported")

        if config:
            return callback_class(**config)
        return callback_class()

    def register_callback(self, name: str, callback_class: type[Callback]) -> None:
        """Registers a new callback class with the factory.

        Args:
            name (str): The name to associate with the callback.
            callback_class (type[Callback]): The callback class to register.
        """
        self.callbacks[name] = callback_class


class LoggerFactory:
    """A factory class for creating and managing PyTorch Lightning loggers.

    This class provides a centralized way to create and register loggers for use in PyTorch Lightning
    training workflows. It supports built-in loggers like `TensorBoardLogger`, `WandbLogger`, and
    `MLFlowLogger`, and allows users to register custom loggers.
    """

    def __init__(self) -> None:
        self.loggers = {
            "tensorboard": AnomalibTensorBoardLogger,
            "wandb": AnomalibWandbLogger,
            "mlflow": AnomalibMLFlowLogger,
            "comet": AnomalibCometLogger,
        }

    def create(
        self, logger_name: str, config: dict[str, Any] | None = None
    ) -> TensorBoardLogger | WandbLogger | MLFlowLogger | CometLogger:
        """Creates an instance of the specified logger.

        Args:
            logger_name (str): The name of the logger to create. Must be a key in the `loggers` dictionary.
            config (dict[str, Any] | None, optional): A dictionary of configuration parameters for the logger.
                If provided, these parameters will be passed to the logger's constructor. Defaults to None.

        Raises:
            ValueError: If the specified `logger_name` is not found in the `loggers` dictionary.

        Returns:
            TensorBoardLogger | WandbLogger | MLFlowLogger | CometLogger: An instance of the requested logger.
        """
        logger_class = self.loggers.get(logger_name)
        if not logger_class:
            raise ValueError(f"Logger {logger_name} not supported")

        if config:
            return logger_class(**config)
        return logger_class()

    def register_logger(self, name: str, logger_class: type) -> None:
        """Registers a new logger class with the factory.

        Args:
            name (str): The name to associate with the logger.
            logger_class (type): The logger class to register.
        """
        self.loggers[name] = logger_class

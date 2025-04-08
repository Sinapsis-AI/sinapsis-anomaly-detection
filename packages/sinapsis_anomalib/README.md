<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a><br>
Sinapsis Anomalib
<br>
</h1>

<h4 align="center">Module to provide anomaly detection training, inference and export with Anomalib.</h4>

<p align="center">
<a href="#installation">🐍  Installation</a> •
<a href="#features"> 🚀 Features</a> •
<a href="#example"> 📚 Usage Example</a> •
<a href="#documentation">📙 Documentation</a> •
<a href="#license"> 🔍 License </a>
</p>

<h2 id="installation"> 🐍 Installation </h2>

Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-anomalib --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-anomalib --extra-index-url https://pypi.sinapsis.tech
```

> [!IMPORTANT]
> Templates in each package may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:
>

with <code>uv</code>:

```bash
  uv pip install sinapsis-anomalib[all] --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-anomalib[all] --extra-index-url https://pypi.sinapsis.tech
```

<h2 id="features">🚀 Features</h2>

<h3> Templates Supported</h3>

The **Sinapsis Anomalib** provides a powerful and flexible implementation for anomaly detection with [Anomalib library](https://anomalib.readthedocs.io/en/v1.2.0/).
<details>
<summary><strong><span style="font-size: 1.25em;">AnomalibTorchInference</span></strong></summary>

The following attributes configure PyTorch model inference:
- **`model_path`** (str, required): Path to the trained PyTorch model file (.pt).
- **`transforms`** (list[str], optional): List of torchvision transforms to apply (e.g., ["Resize", "Normalize"]).
- **`device`** (Literal["cuda", "cpu"], required): Hardware acceleration target ("cuda" for GPU, "cpu" for CPU).
</details>

<details>
<summary><strong><span style="font-size: 1.25em;">AnomalibOpenVINOInference</span></strong></summary>

The following attributes configure OpenVINO-optimized inference:
- **`model_path`** (str, required): Path to OpenVINO model directory (containing .xml and .bin).
- **`transforms`** (dict, optional): Preprocessing steps matching the model's requirements.
- **`device`** (Literal["CPU", "GPU"], optional): OpenVINO device plugin to use (default: "CPU").
- **`model_height`** (int, required): The image height expected by OV model.
- **`model_width`** (int, required): The image width expected by OV model.
</details>
<details>
<summary><strong><span style="font-size: 1.25em;">AnomalibTrain</span></strong></summary>

The following attributes apply to AnomalibTrain template:
- **`folder_attributes_config_path`** (str | Path, required): Path to datamodule configuration YAML file. This must follow Anomalib's [Folder data format specification](https://anomalib.readthedocs.io/en/v1.2.0/markdown/guides/reference/data/image/folder.html). An example configuration is provided at `packages/sinapsis_anomalib/src/sinapsis_anomalib/configs/datamodule_config.yml`.
- **`generic_key`** (str, required): Unique identifier for training artifacts.
- **`callbacks`** (list[Callback], optional): PyTorch Lightning callbacks.
- **`normalization`** (dict, optional): Input normalization configuration.
- **`threshold`** (dict, optional): Anomaly threshold settings.
- **`task`** (TaskType, optional): Task type (classification/detection/segmentation).
- **`image_metrics`** (list, optional): Image-level evaluation metrics.
- **`pixel_metrics`** (list, optional): Pixel-level evaluation metrics.
- **`logger`** (Logger, optional): Training logger configuration.
- **`default_root_dir`** (Path, required): Output directory for training artifacts.
- **`callback_configs`** (dict, optional): Callback initialization parameters.
- **`logger_configs`** (dict, optional): Logger initialization parameters.
- **`max_epochs`** (int, optional): Maximum training epochs.
- **`ckpt_path`** (str | Path, optional): Checkpoint path for resuming training.
- **`accelerator`** (Literal["cpu", "gpu", "tpu", "hpu", "auto"]): Define the device to be used during training. Defaults to "cpu".
- **`trainer_args`** (dict[str, Any]): General trainer asrguments. For more details see:
  https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags

Additional model-specific attributes can be dynamically assigned through the class initialization dictionary (`*_init` attributes). These attributes correspond directly to the arguments used in Anomalib Models. Typically used for hyperparameters directly assigned to the corresponding model or to modify the model's architecture.
</details>

<details>
<summary><strong><span style="font-size: 1.25em;">AnomalibExport</span></strong></summary>

The following attributes apply to AnomalibExport template:
- **`folder_attributes_config_path`** (str | Path, required for INT8_ACQ/INT8_PTQ compression): Path to datamodule configuration YAML (only required when using INT8 quantization). Must follow Anomalib's [Folder data format](https://anomalib.readthedocs.io/en/v1.2.0/markdown/guides/reference/data/image/folder.html).
- **`callbacks`** (list[Callback], optional): PyTorch Lightning callbacks.
- **`normalization`** (dict, optional): Input normalization configuration.
- **`threshold`** (dict, optional): Anomaly threshold settings.
- **`image_metrics`** (list, optional): Image-level evaluation metrics.
- **`pixel_metrics`** (list, optional): Pixel-level evaluation metrics.
- **`logger`** (Logger, optional): Training logger configuration.
- **`default_root_dir`** (Path, required): Output directory for training artifacts.
- **`callback_configs`** (dict, optional): Callback initialization parameters.
- **`logger_configs`** (dict, optional): Logger initialization parameters.
- **`export_type`** (ExportType | str, required): Export format (TORCH/ONNX/OPENVINO).
- **`export_root`** (str | Path, optional): Output directory for exported models.
- **`input_size`** (tuple[int, int], optional): Model input dimensions.
- **`compression_type`** (CompressionType, optional): Model compression method.
- **`metric`** (Metric | str, optional): Calibration metric.
- **`ov_args`** (dict, optional): OpenVINO-specific arguments.
- **`ckpt_path`** (str, optional): Explicit checkpoint path override.
- **`generic_key_chkpt`** (str, optional): Alternate key for checkpoint loading.

Additional model-specific attributes can be dynamically assigned through the class initialization dictionary (`*_init` attributes). These attributes correspond directly to the arguments used in Anomalib Models. Typically used for hyperparameters directly assigned to the corresponding model or to modify the model's architecture.
</details>

> [!TIP]
> Use CLI command ``` sinapsis info --all-template-names``` to show a list with all the available Template names installed with Sinapsis Anomalib.

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for ***CfaTrain*** use ```sinapsis info --example-template-config CfaTrain``` to produce the following example config:

```yaml
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
    folder_attributes_config_path: null
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
    cfa_init:
      backbone: wide_resnet50_2
      gamma_c: 1
      gamma_d: 1
      num_nearest_neighbors: 3
      num_hard_negative_features: 3
      radius: 1.0e-05
```

<details>
<summary><strong><span style="font-size: 1.25em;">🚫 Excluded Models</span></strong></summary>

Some models that required additional configuration have been excluded and support for this will be included in future releases.

- **EfficientAd**
- **VlmAd**
- **Cfa**
- **Dfkde**
- **Fastflow**
- **Supersimplenet**
- **AiVad**

For all other supported models, refer to the Anomalib documentation linked above.
</details>

<h2 id="example"> 📚 Usage Example </h2>
Below is an example configuration for **Sinapsis Anomalib** using a CFLOW model. This setup trains an anomaly detection model with configurable hyperparameters, including learning rate and epochs, and exports it in OpenVINO format for optimized inference. The pipeline includes training, model export, and predefined paths for outputs.

<details>
<summary><strong><span style="font-size: 1.25em;">Example config</span></strong></summary>

```yaml
agent:
  name: anomalib_train_export

templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}

- template_name: CflowTrain
  class_name: CflowTrain
  attributes:
    folder_attributes_config_path: "configs/datamodule_config.yml"
    default_root_dir: "results/model"
    max_epochs: 1
    cflow_init:
      lr: 0.0001

- template_name: CflowExport
  class_name: CflowExport
  attributes:
    generic_key_chkpt: "CflowTrain"
    export_type: "openvino"
    export_root: "results/model/exported"
```
</details>
This configuration defines an **agent** and a sequence of **templates** to train and export a model based on a certain data configuration.

> [!IMPORTANT]
>Attributes specified under the `*_init` keys (e.g., `cflow_init`) correspond directly to the Anomalib models parameters. Ensure that values are assigned correctly according to the official [Anomalib documentation](https://anomalib.readthedocs.io/en/v1.2.0/), as they affect the behavior and performance of the model.
>

To run the config, use the CLI:
```bash
sinapsis run name_of_config.yml
```

</details>



<h2 id="documentation">📙 Documentation</h2>

Documentation for this and other sinapsis packages is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)


<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.

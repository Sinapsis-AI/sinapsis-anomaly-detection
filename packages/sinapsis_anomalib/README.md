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

- **AnomalibTorchInference**\
_Run anomaly detection inference using PyTorch models._
- **AnomalibOpenVINOInference**\
_Perform optimized inference using OpenVINO-accelerated models._
- **AnomalibTrain**\
_Train custom anomaly detection models with Anomalib._
- **AnomalibExport**\
_Export trained models for deployment in different formats._




> [!TIP]
> Use CLI command ``` sinapsis info --all-template-names``` to show a list with all the available Template names installed with Sinapsis Anomalib.

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for ***CflowTrain*** use ```sinapsis info --example-template-config CflowTrain``` to produce the following example config:

```yaml
agent:
  name: my_test_agent
templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}
- template_name: CflowTrain
  class_name: CflowTrain
  template_input: InputTemplate
  attributes:
    folder_attributes:
      name: 'dataset'
      root: null
      normal_dir: 'images/normal'
      abnormal_dir: null
      normal_test_dir: null
      mask_dir: null
      normal_split_ratio: 0.2
      extensions: null
      train_batch_size: 32
      eval_batch_size: 32
      num_workers: 8
      test_split_mode:
      - from_dir
      test_split_ratio: 0.2
      val_split_mode:
      - from_test
      val_split_ratio: 0.5
      seed: null
    callbacks: null
    normalization: null
    threshold: null
    image_metrics: null
    pixel_metrics: null
    logger: null
    callback_configs: null
    logger_configs: null
    ckpt_path: null
    train_root: null
    trainer_args:
      devices: auto
      accelerator: cpu
      min_epochs: 1
      max_epochs: 5
    cflow_init:
      backbone: wide_resnet50_2
      layers:
      - layer2
      - layer3
      - layer4
      pre_trained: true
      fiber_batch_size: 64
      decoder: freia-cflow
      condition_vector: 128
      coupling_blocks: 8
      clamp_alpha: 1.9
      permute_soft: false
      lr: 0.0001
      pre_processor: true
      post_processor: true
      evaluator: true
      visualizer: true

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
    train_root: "model"
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

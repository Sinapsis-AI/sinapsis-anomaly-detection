<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a><br>
Sinapsis Anomaly Detection
<br>
</h1>

<h4 align="center">Monorepo with packages to provide anomaly detection training, inference and export for computer vision.</h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#packages">📦 Packages</a> •
<a href="#webapp"> 🌐 Webapp </a>  •
<a href="#documentation">📙 Documentation</a> •
<a href="#license">🔍 License</a>
</p>

<h2 id="installation">🐍 Installation</h2>

This monorepo currently consists of the following packages for anomaly detection:

* <code>sinapsis-anomalib</code>

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

> [!TIP]
> You can also install all the packages within this project:
>
```bash
  uv pip install sinapsis-anomaly-detection[all] --extra-index-url https://pypi.sinapsis.tech
```

<h2 id="packages">📦 Packages</h2>
<details id='packages'><summary><strong><span style="font-size: 1.0em;"> Packages summary</span></strong></summary>


- **Sinapsis Anomalib**
    - **AnomalibTorchInference**\
    _Run anomaly detection inference using PyTorch models._
    - **AnomalibOpenVINOInference**\
    _Perform optimized inference using OpenVINO-accelerated models._
    - **AnomalibTrain**\
    _Train custom anomaly detection models with Anomalib._
    - **AnomalibExport**\
    _Export trained models for deployment in different formats._
</details>

> [!TIP]
> Use CLI command ``` sinapsis info --all-template-names``` to show a list with all the available Template names installed with Sinapsis Anomaly Detection.

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for ***AnomalibTorchInference*** use ```sinapsis info --example-template-config AnomalibTorchInference``` to produce the following example config:


```yaml
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
    model_path: '/path/to/model.pt'
    transforms: null
    device: cuda
```





<h2 id="webapp">🌐 Webapp</h2>

The webapp offers an interface for anomaly detection on images using pretrained models. Upload images and visualize results (labels, bboxes, or masks) based on the provided agent configuration.

> [!IMPORTANT]
> To run the app you first need to clone this repository:

```bash
git clone git@github.com:Sinapsis-ai/sinapsis-anomaly-detection.git
cd sinapsis-anomaly-detection
```

> [!NOTE]
> If you'd like to enable external app sharing in Gradio, `export GRADIO_SHARE_APP=True`

> [!NOTE]
> Model training is performed when starting the webapp if an exported model does not exist in the `MODEL_PATH` location.

<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">🐳 Docker</span></strong></summary>

**IMPORTANT** This docker image depends on the sinapsis-nvidia:base image. Please refer to the official [sinapsis](https://github.com/Sinapsis-ai/sinapsis?tab=readme-ov-file#docker) instructions to Build with Docker.

1. **Build the sinapsis-anomalib image**:
```bash
docker compose -f docker/compose.yaml build
```

2. **Start the app container**:
```bash
docker compose -f docker/compose_apps.yaml up sinapsis-anomalib-gradio -d
```
3. **Check the status**:
```bash
docker logs -f sinapsis-anomalib-gradio
```
3. The logs will display the URL to access the webapp, e.g.:

NOTE: The url can be different, check the output of the logs
```bash
Running on local URL:  http://127.0.0.1:7860
```
4. To stop the app:
```bash
docker compose -f docker/compose_apps.yaml down
```

<details>
<summary id="uv"><strong><span style="font-size: 1.25em;">Webapp Configuration</span></strong></summary>

Customize the webapp behavior by updating the `environment` fields in `docker/compose_apps.yaml`:

For custom inference agent:
```yaml
AGENT_CONFIG_PATH: "/app/configs/inference/custom_torch_demo_agent.yml"
```

For custom training agent:
```yaml
TRAINING_CONFIG: "/app/configs/custom_train_export_agent.yaml"
```

For custom inference model path:
```yaml
MODEL_PATH: "/app/artifacts/exported_models/weights/torch/custom_model.pt"
```

For custom test data:
```yaml
TEST_DIR: "/app/artifacts/data/custom_test_data"
```

</details>


</details>


<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">💻 UV</span></strong></summary>

To run the webapp using the <code>uv</code> package manager, please:

1. **Create the virtual environment and sync the dependencies**:
```bash
uv sync --frozen
```
2. **Install the wheel**:
```bash
uv pip install sinapsis-anomaly-detection[all] --extra-index-url https://pypi.sinapsis.tech
```

3. **Run the webapp**:
```bash
uv run webapps/anomalib_gradio_app.py
```
4. **The terminal will display the URL to access the webapp, e.g.**:

NOTE: The url can be different, check the output of the terminal
```bash
Running on local URL:  http://127.0.0.1:7860
```

<details>
<summary id="uv"><strong><span style="font-size: 1.25em;">Webapp Configuration</span></strong></summary>


Customize the webapp behavior by exporting the following variables with your custom values before running the app:

For custom inference agent:
```bash
export AGENT_CONFIG_PATH="packages/sinapsis_anomalib/src/sinapsis_anomalib/configs/inference/custom_torch_demo_agent.yml"
```

For custom training agent:
```bash
export TRAINING_CONFIG="packages/sinapsis_anomalib/src/sinapsis_anomalib/configs/custom_train_export_agent.yaml"
```

For custom inference model path:
```bash
export MODEL_PATH="artifacts/exported_models/weights/torch/custom_model.pt"
```

For custom test data:
```bash
export TEST_DIR="artifacts/data/custom_test_data"
```



</details>

</details>

<h2 id="documentation">📙 Documentation</h2>

Documentation for this and other sinapsis packages is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)

<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.

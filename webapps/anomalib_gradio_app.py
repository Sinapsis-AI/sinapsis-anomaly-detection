# -*- coding: utf-8 -*-
import os

import gradio as gr
from sinapsis.webapp.agent_gradio_helper import (
    add_logo_and_title,
    css_header,
    init_image_inference,
)
from sinapsis_core.cli.run_agent_from_config import generic_agent_builder
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.utils.env_var_keys import AGENT_CONFIG_PATH, GRADIO_SHARE_APP
from sinapsis_data_readers.templates.image_readers.base_image_folder_data_loader import SUPPORTED_IMAGE_TYPES

TRAINING_CONFIG_ENV = os.environ.get("TRAINING_CONFIG")
MODEL_PATH_ENV = os.environ.get("MODEL_PATH")
TEST_DIR_ENV = os.environ.get("TEST_DIR")

DEFAULT_TRAINING_CONFIG = "packages/sinapsis_anomalib/src/sinapsis_anomalib/configs/train_export.yaml"
TRAINING_CONFIG = TRAINING_CONFIG_ENV or DEFAULT_TRAINING_CONFIG

DEFAULT_CONFIG = "packages/sinapsis_anomalib/src/sinapsis_anomalib/configs/inference/torch_demo.yml"
CONFIG_FILE = AGENT_CONFIG_PATH or DEFAULT_CONFIG

DEFAULT_MODEL_PATH = "artifacts/exported_models/weights/torch/model.pt"
MODEL_PATH = MODEL_PATH_ENV or DEFAULT_MODEL_PATH

DEFAULT_TEST_DIR = "artifacts/data/test_data"
TEST_DIR = TEST_DIR_ENV or DEFAULT_TEST_DIR


def get_examples_list() -> list[str]:
    """Inspects test directory and produce a list of paths of valid image files.

    Returns:
        list[str]: List of test image paths.
    """
    example_list = os.listdir(TEST_DIR)
    example_list = [
        os.path.join(TEST_DIR, file)
        for file in example_list
        if os.path.splitext(file)[1].lower() in SUPPORTED_IMAGE_TYPES
    ]
    return example_list


EXAMPLES_LIST = get_examples_list()


def train_model() -> None:
    """Creates and executes an agent to train and export the anomalib model used during demo inference app."""
    agent = generic_agent_builder(TRAINING_CONFIG)

    container = DataContainer()
    _ = agent(container)


def create_demo() -> gr.Blocks:
    """Creates a Gradio interface.

    Returns:
        gr.Blocks: A configured Gradio Blocks interface ready to launch.
    """
    with gr.Blocks(css=css_header(), title="Sinapsis Anomalib Inference") as demo_app:
        add_logo_and_title("Sinapsis Anomalib Inference")

        init_image_inference(CONFIG_FILE, examples=EXAMPLES_LIST)

    return demo_app


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        train_model()

    demo = create_demo()
    demo.launch(share=GRADIO_SHARE_APP)

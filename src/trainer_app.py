import sys
import yaml
from pathlib import Path

import gradio as gr
import anomalib

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.utils import TestSplitMode
from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import get_experiment_logger

from pytorch_lightning import Trainer, seed_everything

from .logger import FileLogger


class AnomalibApp:
    def __init__(self):
        self.trainer = None
        self.app = gr.Blocks()

    def train(self, image_folder, model, batch_size, val_ratio):
        logs = gr.Code(visible=True)
        infer_img = gr.File(interactive=True, visible=True)
        infer_btn = gr.Button(value="Infer", interactive=True, visible=True)

        if self.trainer is not None:
            self.logger.info("Model already trained.")
            return infer_img, infer_btn, logs, trainer

        # model_config = yaml.safe_load(open(f"./src/anomalib/src/anomalib/models/{model}/config.yaml", "r"))
        # config = model_config
        config_path = (
            Path(f"{anomalib.__file__}").parent / f"models/{model}/config.yaml"
        )
        config = get_configurable_parameters(model_name=model, config_path=config_path)
        config["dataset"] = yaml.safe_load(open("./config.yaml", "r"))

        data_config = {
            "format": "folder",
            "name": Path(image_folder).name,
            "root": Path(image_folder),
            "split_ratio": float(val_ratio),
            "train_batch_size": int(batch_size),
            "test_batch_size": int(batch_size),
        }

        config["dataset"].update(data_config)

        if config.project.get("seed") is not None:
            seed_everything(config.project.seed)

        datamodule = get_datamodule(config)
        model = get_model(config)
        experiment_logger = get_experiment_logger(config)
        callbacks = get_callbacks(config)

        trainer = Trainer(
            **config.trainer, logger=experiment_logger, callbacks=callbacks
        )
        self.logger.info("Training the model.")
        trainer.fit(model=model, datamodule=datamodule)

        self.logger.info("Loading the best model weights.")
        load_model_callback = LoadModelCallback(
            weights_path=trainer.checkpoint_callback.best_model_path
        )
        trainer.callbacks.insert(0, load_model_callback)  # pylint: disable=no-member

        if config.dataset.test_split_mode == TestSplitMode.NONE:
            self.logger.info("No test set provided. Skipping test stage.")
        else:
            self.logger.info("Testing the model.")
            trainer.test(model=model, datamodule=datamodule)

        self.trainer = trainer

        return infer_img, infer_btn, logs, trainer

    def update_options(self, config_type):
        if config_type == "Basic":
            batch_size = gr.Text(value=1, label="Batch Size", visible=True)
            val_ratio = gr.Text(value=0.2, label="Validation Ratio", visible=True)
            cust_conf = gr.File(
                label="Select custom config yaml.", interactive=False, visible=False
            )
        else:
            batch_size = gr.Text(label="Batch Size", visible=False)
            val_ratio = gr.Text(label="Validation Ratio", visible=False)
            cust_conf = gr.File(
                label="Select custom config yaml.", interactive=True, visible=True
            )

        return (batch_size, val_ratio, cust_conf)

    # Utility
    def isfloat(self, num):
        try:
            float(num)
            return True
        except ValueError:
            return False

    def change_train_btn_state(self, batch_size, val_ratio):
        train_btn = gr.Button(value="Train", interactive=False, visible=False)
        if (
            batch_size.isnumeric()
            and self.isfloat(val_ratio)
            and 0 <= float(val_ratio) < 1.0
        ):
            train_btn = gr.Button(value="Train", interactive=True, visible=True)
        return train_btn

    def read_logs(self):
        with open(".output.log", "r") as f:
            return f.read()

    def build(self):
        with self.app:
            with gr.Row():
                with gr.Column():
                    image_folder = gr.Text(label="Select training folder.")
                    train_btn = gr.Button(value="Train", interactive=False)
                with gr.Column():
                    config_type = gr.Radio(choices=["Basic", "Custom"])
                    batch_size = gr.Text(label="Batch Size", visible=False)
                    val_ratio = gr.Text(label="Validation Ratio", visible=False)
                    cust_conf = gr.File(visible=False)
            with gr.Row():
                with gr.Column():
                    infer_img = gr.File(visible=False)
                    infer_btn = gr.Button(value="Infer", visible=False)
                with gr.Column():
                    infer_preview = gr.Image(label="Detected Anomaly", visible=False)
            with gr.Row():
                logs = gr.Code(visible=True, interactive=False)
                self.app.load(self.read_logs, None, logs, every=1)

            # Update config options
            gr.on(
                [config_type.change],
                self.update_options,
                inputs=[config_type],
                outputs=[batch_size, val_ratio, cust_conf],
            )

            # Update Train button state
            gr.on(
                [batch_size.change, val_ratio.change],
                self.change_train_btn_state,
                inputs=[batch_size, val_ratio],
                outputs=[train_btn],
            )

            sys.stdout = self.logger = FileLogger(".output.log")
            # logging.basicConfig(filename='.output.log',
            #                     filemode='w',
            #                     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            #                     datefmt='%H:%M:%S',
            #                     level=logging.INFO)
            # self.logger = logging.getLogger("AnomalibTrainer")

            model = gr.Text(label="Model", value="padim", visible=False)

            # Run training on folder
            train_btn.click(
                self.train,
                inputs=[image_folder, model, batch_size, val_ratio],
                outputs=[infer_img, infer_btn, logs],
                show_progress=True,
            )

    def launch(self):
        self.build()
        # Run Gradio app
        self.app.launch()

    def shutdown(self):
        self.app.close()

import sys
import yaml
from pathlib import Path

import gradio as gr
import anomalib

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.utils import TestSplitMode
from anomalib.models import get_model, get_available_models
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import get_experiment_logger
from anomalib.deploy import Inferencer, TorchInferencer

from contextlib import redirect_stdout
from threading import Thread

from pytorch_lightning import Trainer, seed_everything

import logging


class AnomalibApp:
    def __init__(self):
        self.trainer = None
        self.app = gr.Blocks(title="Anomalib Trainer")
        self.trained = False
        self.inferencer = None

    def train_thread(self, model, config):
        with open("./.output.log", "w") as f:
            with redirect_stdout(f):

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
                trainer.callbacks.insert(
                    0, load_model_callback
                )  # pylint: disable=no-member

                if config.dataset.test_split_mode == TestSplitMode.NONE:
                    self.logger.info("No test set provided. Skipping test stage.")
                else:
                    self.logger.info("Testing the model.")
                    trainer.test(model=model, datamodule=datamodule)

        self.trained = True

    def train(self, image_folder, model, batch_size, val_ratio, epochs, cust_conf):
        logs = gr.Code(visible=True)
        train_btn = gr.Button(value="Train", interactive=False, visible=False)

        if not cust_conf:
            config_path = (
                Path(f"{anomalib.__file__}").parent / f"models/{model}/config.yaml"
            )
            config = get_configurable_parameters(
                model_name=model, config_path=config_path
            )
            config["dataset"] = yaml.safe_load(open("./config.yaml", "r"))
            config["trainer"].update(
                {"default_root_dir": "results/custom/run", "max_epochs": int(epochs)}
            )
            config["project"].update({"path": "results/custom/run"})
            config["optimization"].update({"export_mode": "torch"})

            data_config = {
                "format": "folder",
                "name": Path(image_folder).name,
                "root": Path(image_folder),
                "val_split_ratio": float(val_ratio),
                "train_batch_size": int(batch_size),
                "test_batch_size": int(batch_size),
            }

            config["dataset"].update(data_config)

            if config.project.get("seed") is not None:
                seed_everything(config.project.seed)
        else:
            config = get_configurable_parameters(
                model_name=model, config_path=cust_conf
            )

        trainer_thread = Thread(
            target=self.train_thread, args=(model, config)
        )
        trainer_thread.start()

        return logs, train_btn

    def infer(self, image):
        """Inference function, return anomaly map, score, heat map, prediction mask ans visualisation.

        Args:
            image (np.ndarray): image to compute
            inferencer (Inferencer): model inferencer

        Returns:
            tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
            heat_map, pred_mask, segmentation result.
        """
        # Perform inference for the given image.
        if self.inferencer is None:
            self.inferencer = TorchInferencer(
                path="results/custom/run/weights/torch/model.pt"
            )
        predictions = self.inferencer.predict(image=image)
        infer_preview = gr.Image(
            value=predictions.segmentations, label="Detected Anomaly", visible=True
        )
        return infer_preview

    def update_options(self, config_type):
        if config_type == "Basic":
            model = gr.Dropdown(
                label="Model", choices=get_available_models(), visible=True
            )
            batch_size = gr.Text(value=1, label="Batch Size", visible=True)
            val_ratio = gr.Text(value=0.2, label="Validation Ratio", visible=True)
            epochs = gr.Text(value=5, label="Epochs", visible=True)
            cust_conf = gr.File(
                label="Select custom config yaml.", interactive=False, visible=False
            )
        else:
            model = gr.Dropdown(label="Model", visible=False)
            batch_size = gr.Text(label="Batch Size", visible=False)
            val_ratio = gr.Text(label="Validation Ratio", visible=False)
            epochs = gr.Text(label="Epochs", visible=False)
            cust_conf = gr.File(
                label="Select custom config yaml.", interactive=True, visible=True
            )

        return (model, batch_size, val_ratio, epochs, cust_conf)

    # Utility
    def isfloat(self, num):
        try:
            float(num)
            return True
        except ValueError:
            return False

    def isint(self, num):
        try:
            int(num)
            return True
        except ValueError:
            return False

    def change_train_btn_state(self, batch_size, val_ratio, epochs):
        train_btn = gr.Button(value="Train", interactive=False, visible=False)
        if (
            batch_size.isnumeric()
            and self.isfloat(val_ratio)
            and 0 <= float(val_ratio) < 1.0
            and self.isint(epochs)
            and 0 <= int(epochs)
        ):
            train_btn = gr.Button(value="Train", interactive=True, visible=True)
        return train_btn

    def enable_infer_btn(self):
        if self.trained:
            enable = True
        else:
            enable = False

        infer_img = gr.File(
            label="Select image to infer.", interactive=True, visible=enable
        )
        infer_btn = gr.Button(value="Infer", interactive=True, visible=enable)

        return (infer_img, infer_btn)

    def read_logs(self):
        with open("./.output.log", "r") as f:
            return f.read()

    def build(self):
        with self.app:
            gr.Markdown("# Anomalib Trainer")
            with gr.Row():
                with gr.Column():
                    image_folder = gr.Text(label="Select training folder.")
                    train_btn = gr.Button(value="Train", interactive=False)
                with gr.Column():
                    config_type = gr.Radio(label="Mode", choices=["Basic", "Custom"])
                    model = gr.Dropdown(label="Model", visible=False)
                    batch_size = gr.Text(label="Batch Size", visible=False)
                    val_ratio = gr.Text(label="Validation Ratio", visible=False)
                    epochs = gr.Text(label="Epochs", visible=False)
                    cust_conf = gr.File(visible=False)
            with gr.Row():
                with gr.Column():
                    infer_img = gr.File(label="Select image to infer.", visible=True)
                    infer_btn = gr.Button(value="Infer", visible=True)
                with gr.Column():
                    infer_preview = gr.Image(label="Detected Anomaly", visible=False)
            with gr.Row():
                logs = gr.Code(visible=False, interactive=False)
                self.app.load(self.read_logs, None, logs, every=1)

            # Update config options
            gr.on(
                [config_type.change],
                self.update_options,
                inputs=[config_type],
                outputs=[model, batch_size, val_ratio, epochs, cust_conf],
            )

            # Update Train button state
            gr.on(
                [batch_size.change, val_ratio.change, epochs.change],
                self.change_train_btn_state,
                inputs=[batch_size, val_ratio, epochs],
                outputs=[train_btn],
            )

            self.logger = logging.getLogger("AnomalibTrainer")

            # Run training on folder
            train_btn.click(
                self.train,
                inputs=[image_folder, model, batch_size, val_ratio, epochs, cust_conf],
                outputs=[logs, train_btn],
                show_progress=True,
            )

            # Update infer button
            gr.on(
                [logs.change],
                self.enable_infer_btn,
                inputs=[],
                outputs=[infer_img, infer_btn],
            )

            # Run inference
            infer_btn.click(
                self.infer,
                inputs=[infer_img],
                outputs=[infer_preview],
                show_progress=True,
            )

    def launch(self):
        self.build()
        # Run Gradio app
        self.app.launch()

    def shutdown(self):
        self.app.close()

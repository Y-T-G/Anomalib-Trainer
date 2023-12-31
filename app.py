from pathlib import Path

from src.trainer_app import AnomalibApp

if __name__ == "__main__":
    base_model_dir = "model"

    # # Download Instance Segmentation model
    # model_name = "instance-segmentation-security-1040"

    # model_path = Path(f"{base_model_dir}/{model_name}.xml")
    # if not model_path.exists():
    #     model_xml_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/temp/instance-segmentation-security-1040/FP16/instance-segmentation-security-1040.xml"
    #     download_ir_model(model_xml_url, base_model_dir)

    app = AnomalibApp()

    app.launch()

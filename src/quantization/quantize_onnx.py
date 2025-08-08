from ultralytics import YOLO
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
import glob
import subprocess
from src.quantization.image_calibration import ImageCalibrationDataReader


if __name__ == "__main__":
    original_model: str = "models/yolo11n.pt"
    onnx_model: str = "models/yolo11n.onnx"
    preprocessed_model: str = "models/yolo11_pre.onnx"
    quantized_model: str = "models/yolo11n_static_quantized.onnx"

    model: YOLO = YOLO(original_model)
    model.export(format='onnx') # exports the model in '.onnx' format

    subprocess.run([
        "python", "-m", "onnxruntime.quantization.preprocess",
        "--input", onnx_model,
        "--output", preprocessed_model
    ])

    calibration_images: list[str] = sorted(glob.glob("src/quantization/calibration_data/*.jpg"))

    # Create an instance of the ImageCalibrationDataReader
    calibration_data_reader: ImageCalibrationDataReader = ImageCalibrationDataReader(calibration_images)

    quantize_static(preprocessed_model, quantized_model,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8,
                calibration_data_reader=calibration_data_reader,
                quant_format=QuantFormat.QDQ,
                nodes_to_exclude=['/model.22/Concat_3', '/model.22/Split', '/model.22/Sigmoid'
                                 '/model.22/dfl/Reshape', '/model.22/dfl/Transpose', '/model.22/dfl/Softmax', 
                                 '/model.22/dfl/conv/Conv', '/model.22/dfl/Reshape_1', '/model.22/Slice_1',
                                 '/model.22/Slice', '/model.22/Add_1', '/model.22/Sub', '/model.22/Div_1',
                                  '/model.22/Concat_4', '/model.22/Mul_2', '/model.22/Concat_5'],
                per_channel=False,
                reduce_range=True)
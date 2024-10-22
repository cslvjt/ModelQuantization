import onnxruntime
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
import numpy as np
import time
import argparse
import os
import torchvision.transforms as transforms
from PIL import Image
import sys
sys.path.append("..")
from utils.inference import get_gt_label


def _preprocess_images(dataset_path, height=720, width=1280, cnt=100):
    """Preprocesses the images in the dataset.

    Args:
        dataset_path: Path to the dataset.
        height: Height of the images.
        width: Width of the images.

    Returns:
        A list of preprocessed images.
    """
    # Read the images from the dataset.
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    images = []
    img_list = os.listdir(dataset_path)
    for img_name in img_list[:cnt]:
        img_path = os.path.join(dataset_path, img_name)
        image = Image.open(img_path).convert("RGB")
        image = trans(image).numpy()
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        images.append(image)
    return images

class DataReader(CalibrationDataReader):
    def __init__(self, model_path, dataset_path=r"D:\datasets\\imagenet_val\\ILSVRC2012_img_val"):
        self.enum_data = None
        session = onnxruntime.InferenceSession(model_path)
        self.input_name = session.get_inputs()[0].name
        (b,c,h,w) = session.get_inputs()[0].shape
        self.data_list = _preprocess_images(dataset_path, h, w, 100)
        self.datasize = len(self.data_list)
    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: data} for data in self.data_list]
            )
        return next(self.enum_data, None)
    def rewind(self):
        self.enum_data = None

def benchmark(model_path, dataset_path):
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    img_list = os.listdir(dataset_path)
    hashtable = get_gt_label()
    acc = 0
    runtime = 0
    cnt = 100
    for img_name in img_list[:cnt]:
        label = hashtable[img_name]
        img_path = os.path.join(dataset_path, img_name)
        image = Image.open(img_path).convert("RGB")
        image = trans(image).numpy()
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        start = time.perf_counter()
        out = session.run([], {input_name: image})
        pred = np.argmax(out[0])
        if pred == label:
            acc += 1
        infer_time = (time.perf_counter() - start) * 1000
        runtime += infer_time
    print(f"Accuracy: {acc/cnt:.4f}")
    print(f"Average time: {runtime/cnt:.2f} ms")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument("--output_model", required=True, help="output model")
    parser.add_argument(
        "--calibrate_dataset", default="./test_images", help="calibration data set"
    )
    parser.add_argument(
        "--quant_format",
        default=QuantFormat.QDQ,
        type=QuantFormat.from_string,
        choices=list(QuantFormat),
    )
    parser.add_argument("--per_channel", default=False, type=bool)
    args = parser.parse_args()
    return args
def main():
    args = get_args()
    input_model_path = args.input_model
    output_model_path = args.output_model
    dr = DataReader(input_model_path)
    quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=args.quant_format,
        per_channel=args.per_channel,
        weight_type=QuantType.QInt8,
    )
    print("Calibrated and quantized model saved.")

    print("benchmarking fp32 model...")
    benchmark(input_model_path, r"D:\datasets\\imagenet_val\\ILSVRC2012_img_val")

    print("benchmarking int8 model...")
    benchmark(output_model_path, r"D:\datasets\\imagenet_val\\ILSVRC2012_img_val")
if __name__ == "__main__":
    main()
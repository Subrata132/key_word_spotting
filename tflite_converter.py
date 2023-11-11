import os
import torch
from torchvision.models import mobilenet_v2
import tensorflow as tf
from model import CNNModelTiny
from constants import TrainingParams
from onnx_tf.backend import prepare
import onnx


def main():
    batch_size = 1
    onnx_model_path = 'model.onnx'
    training_params = TrainingParams()
    model = CNNModelTiny()
    model.load_state_dict(
        torch.load(os.path.join(training_params.save_path, training_params.model_name), map_location='cpu')
    )
    model.eval()
    sample_input = torch.rand((batch_size, 1, 65, 65))
    y = model(sample_input)
    torch.onnx.export(
        model,
        sample_input,
        onnx_model_path,
        verbose=False,
        input_names=['input'],
        output_names=['output'],
        opset_version=12
    )
    tf_model_path = 'model_tf'
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_model_path)
    tflite_model_path = 'model.tflite'
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    tflite_model = converter.convert()
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    main()

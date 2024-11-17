from os import walk
from ultralytics import YOLO


for (dirpath, dirname, filenames) in walk('weights'):
    for i in filenames:
        model = YOLO(f'{dirpath}/{i}')
        model.export(format="onnx")
import os

import urllib.request

weights_dir = os.path.join(os.getcwd(), "weights")

os.makedirs(weights_dir, exist_ok=True)

urls = [

    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10n.pt",

    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10s.pt",

    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10m.pt",

    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10b.pt",

    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10x.pt",

    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10l.pt"

]

for url in urls:

    file_name = os.path.join(weights_dir, os.path.basename(url))

    urllib.request.urlretrieve(url, file_name)
from flask import Flask, request
import string
import random
from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = Flask(__name__)
model = YOLO('best.pt')


@app.route('/predict', methods=['POST'])
def predict():
    images = [f[0] for f in request.files.listvalues()]
    input = list()
    output = list()
    hashes = dict()
    for index, image in enumerate(images):
        letters = string.ascii_lowercase
        hash = ''.join(random.choice(letters) for i in range(10))
        image_path = f'uploads/{hash}-{image.filename}'
        hashes[f'image{index}'] = {
            'image_name': image.filename,
            'image_path': image_path,
            'hash': hash
        }
        image.save(image_path)
        input.append(image_path)
    results = model.predict(input)
    for result in results:
        boxes = list()
        for box in result.boxes:
            boxes.append({
                'x1': int(box.xyxy[0][0]),
                'y1': int(box.xyxy[0][1]),
                'x2': int(box.xyxy[0][2]),
                'y2': int(box.xyxy[0][3]),
                's': float(box.xywhn[0][2] * box.xywhn[0][3]),
                'conf': float(box.conf),
                'cls': int(box.cls),
            })
        output.append(
            {
                'boxes': boxes
            }
        )
    for img in input:
        if os.path.exists(img):
            os.remove(img)
    return {
        'results': output,
        'speed': results[0].speed,
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8100', debug=True)

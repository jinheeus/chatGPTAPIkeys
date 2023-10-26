from flask import Flask, render_template, request
import cv2
import numpy as np

app = Flask(__name__)

# YOLO 모델과 클래스 이름 로드
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# 이미지 분석 함수
def analyze_image(image_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # YOLO에 필요한 전처리
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # 객체 탐지 후 사각형과 클래스명 표시
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite('static/analyzed_image.png', img)

# 웹 앱 라우트
@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        f = request.files['file']
        f.save('static/uploads/' + f.filename)
        analyze_image('static/uploads/' + f.filename)
        return render_template('result.html', image_name=f.filename)

if __name__ == '__main__':
    app.run(debug=True)

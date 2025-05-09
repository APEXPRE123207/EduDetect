import os
import time
import cv2
import numpy as np
from color_identification import color_identify
from shape_detection_v2 import mask_image
from test_model import predict_image
# from util import util1
from voice import speech

# Configuration (paths, thresholds, etc.)v
CONFIG = {
    "yolo_config": r'..\Resources\.cvlib\object_detection\yolo\yolov3\yolov3.cfg',
    "yolo_weights": r'..\Resources\.cvlib\object_detection\yolo\yolov3\yolov3.weights',
    "coco_names": r'..\Resources\.cvlib\object_detection\yolo\yolov3\coco.names',
    "save_path": r"..\Assets\Pics",
    "colors_csv": r'..\Resources\colors4.csv',
    "model_path": r'..\Resources\shapes_model_v1.pth'
}

# Initialize YOLO network
def initialize_yolo():
    net = cv2.dnn.readNetFromDarknet(CONFIG["yolo_config"], CONFIG["yolo_weights"])
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net

# Main Processing Loop
def main():
    # Initialize YOLO
    net = initialize_yolo()
    cap = cv2.VideoCapture(0)

    # Load class names from coco.names file
    with open(CONFIG["coco_names"], 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Get index for "person" class
    person_class = "person"
    person_idx = classes.index(person_class) if person_class in classes else None
    
    # Get YOLO output layers
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]

    # Set thresholds for object detection
    confidence_threshold = 0.1
    nms_threshold = 0.4
    last_save_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids, confidences, boxes = [], [], []
        for output in outs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold and class_id != person_idx:
                    center_x, center_y, w, h = map(int, [detection[0] * width, detection[1] * height, detection[2] * width, detection[3] * height])
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

        if len(indices) > 0:
            current_time = time.time()
            if current_time - last_save_time >= 20:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    conf_text = f"{confidences[i]:.2f}"
                    color = (0, 255, 0)
                    x_pad, y_pad = max(0, x - 50), max(0, y - 50)
                    w_pad, h_pad = min(w + 100, width - x_pad), min(h + 100, height - y_pad)

                    cv2.rectangle(frame, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), color, 2)
                    cv2.putText(frame, f"{label}: {conf_text}", (x_pad, y_pad - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    cropped_object = frame[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"{label}_{timestamp}.jpg"
                    filepath = os.path.join(CONFIG["save_path"], filename)
                    cv2.imwrite(filepath, cropped_object)

                    rgb,color=color_identify(filepath, CONFIG["colors_csv"])
                    rgb=tuple(rgb)
                    # print(f"RGB: {rgb}")
                    path=mask_image(filepath,rgb)
                    # approx=util1(path)
                    shape = predict_image(path, CONFIG["model_path"])
                    # print(approx)
                    # print(f"Saved: {filepath}")
                    print(f"Detected Shape: {shape} and Color: {color}")
                    text=f" Detected Shape: {shape}, and Color: {color}"
                    speech(text,shape,color)


                last_save_time = current_time

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

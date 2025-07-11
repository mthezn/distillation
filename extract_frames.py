import cv2
import os
from pycocotools.coco import COCO
import numpy as np
import cv2
import json
import os
def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{index:04d}.png")
        cv2.imwrite(frame_path, frame)
        index += 1

    cap.release()
    print(f"Extracted {index} frames to {output_dir}")


def draw_boxes_to_masks(json_path, output_dir, image_size=(512, 640)):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    for box in data['boxes']:
        name = box['name']  # e.g. 'slice_nr_45_needle_driver'
        corners = box['corners']

        # Estrai numero della slice dal nome (es. 45)
        slice_number = ''.join(filter(str.isdigit, name))

        # Crea maschera nera
        mask = np.zeros(image_size, dtype=np.uint8)

        # Estrai solo x, y dai punti (ignora la z=0.5)
        pts = np.array([[int(p[0]), int(p[1])] for p in corners], np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Disegna poligono pieno
        cv2.fillPoly(mask, [pts], color=255)

        # Salva la maschera
        output_path = os.path.join(output_dir, f"mask_{slice_number}.png")
        cv2.imwrite(output_path, mask)

    print(f"✅ Salvate {len(data['boxes'])} maschere in {output_dir}")


#extract_frames("cat1_test_set_public/7_fps1.mp4", "cat1_test_set_public/frames7")
draw_boxes_to_masks(
    json_path="cat1_test_set_public/7_fps1_gc.json",
    output_dir="cat1_test_set_public/masks7",
    image_size=(1024, 1024)  # altezza x larghezza!
)
import cv2
import os
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
def contains_instrument(example):
    mask = np.array(example["color_mask"])  # o "segmentation" se diverso
    return np.any((mask == 169) | (mask == 170))
# 1. Carica il dataset da Hugging Face
dataset = load_dataset("minwoosun/CholecSeg8k", trust_remote_code=True)
filtered_ds = dataset['train'].filter(contains_instrument)
# 2. Crea una cartella temporanea per i frame (opzionale)
os.makedirs("frames", exist_ok=True)

# 3. Estrai i frame dal dataset (split 'train')
frames = filtered_ds.select(range(1000))

# 4. Ordina i frame se necessario (in base a "image_id")
sorted_frames = sorted(frames, key=lambda x: x['image_id'])

# 5. Specifica dimensioni e codec video
sample_image = sorted_frames[0]["image"]
height, width = sample_image.size[1], sample_image.size[0]  # PIL image: size = (W, H)
fps = 25
video_writer = cv2.VideoWriter("cholec_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# 6. Scrivi i frame nel video
for item in tqdm(sorted_frames, desc="Scrivendo il video"):
    pil_img = item["image"]
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # Converti da PIL (RGB) a OpenCV (BGR)
    video_writer.write(cv_img)

video_writer.release()
print("✅ Video salvato come cholec_video.mp4")
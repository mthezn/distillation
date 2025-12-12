import requests
import base64
import json
import os
from pathlib import Path
from datetime import datetime

# Configurazione Roboflow
ROBOFLOW_CONFIG = {
    "api_url": "https://serverless.roboflow.com",
    "api_key": "JGNiK5a9OPkVjtopY5b3",
    "workspace": "sugvu",
    "workflow_id": "find-objects-4"
}


def analyze_image(image_path):
    """Analizza un'immagine con Roboflow API."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")

    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    url = f"{ROBOFLOW_CONFIG['api_url']}/{ROBOFLOW_CONFIG['workspace']}/workflows/{ROBOFLOW_CONFIG['workflow_id']}"
    payload = {
        "api_key": ROBOFLOW_CONFIG["api_key"],
        "inputs": {
            "image": {
                "type": "base64",
                "value": encoded_image
            }
        }
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Errore API: {response.status_code} - {response.text}")


def create_coco_format(image_folder, output_file="annotations.json", min_area=None, top_n=None):
    """
    Crea un file JSON in formato COCO con tutte le immagini e annotazioni.

    Args:
        image_folder: Cartella con le immagini
        output_file: Nome del file JSON di output
        min_area: Area minima per filtrare bbox
        top_n: Numero massimo di bbox per immagine (le più grandi)
    """
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = sorted([f for f in os.listdir(image_folder)
                          if f.lower().endswith(extensions)])

    if not image_files:
        print(f"Nessuna immagine trovata in {image_folder}")
        return

    print(f"Trovate {len(image_files)} immagini da processare\n")

    # Struttura COCO
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Traccia le categorie uniche
    category_map = {}
    category_id = 0
    annotation_id = 0

    for image_id, filename in enumerate(image_files, start=1):
        print(f"[{image_id}/{len(image_files)}] Processando: {filename}")
        image_path = os.path.join(image_folder, filename)

        try:
            # Analizza l'immagine
            result = analyze_image(image_path)

            # Estrai info immagine
            predictions = result['outputs'][0]['predictions']['predictions']
            image_info = result['outputs'][0]['predictions']['image']

            # Aggiungi info immagine
            coco_data["images"].append({
                "id": image_id,
                "width": image_info['width'],
                "height": image_info['height'],
                "file_name": filename,
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # Calcola area e filtra bbox
            bboxes_with_area = []
            for pred in predictions:
                area = pred['width'] * pred['height']
                bboxes_with_area.append({
                    "class": pred['class'],
                    "confidence": pred['confidence'],
                    "x": pred['x'],
                    "y": pred['y'],
                    "width": pred['width'],
                    "height": pred['height'],
                    "area": area
                })

            # Filtra per area minima
            if min_area is not None:
                bboxes_with_area = [b for b in bboxes_with_area if b['area'] >= min_area]

            # Ordina per area e prendi le top N
            bboxes_with_area.sort(key=lambda x: x['area'], reverse=True)
            if top_n is not None:
                bboxes_with_area = bboxes_with_area[:top_n]

            print(f"  ✓ {len(bboxes_with_area)} bbox rilevate")

            # Aggiungi annotazioni
            for bbox in bboxes_with_area:
                # Gestisci categoria
                class_name = bbox['class']
                if class_name not in category_map:
                    category_id += 1
                    category_map[class_name] = category_id
                    coco_data["categories"].append({
                        "id": category_id,
                        "name": class_name
                    })

                # Converti coordinate da centro a top-left corner (formato COCO)
                x_min = bbox['x'] - bbox['width'] / 2
                y_min = bbox['y'] - bbox['height'] / 2

                annotation_id += 1
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_map[class_name],
                    "bbox": [x_min, y_min, bbox['width'], bbox['height']],
                    "area": bbox['area'],
                    "iscrowd": 0,
                    "confidence": bbox['confidence']
                })

        except Exception as e:
            print(f"  ✗ Errore: {e}")
            # Aggiungi comunque l'immagine anche se ci sono errori
            coco_data["images"].append({
                "id": image_id,
                "width": 640,
                "height": 512,
                "file_name": filename,
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    # Salva il file JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)

    print(f"\n=== COMPLETATO ===")
    print(f"File salvato: {output_file}")
    print(f"Immagini totali: {len(coco_data['images'])}")
    print(f"Annotazioni totali: {len(coco_data['annotations'])}")
    print(f"Categorie: {len(coco_data['categories'])} -> {list(category_map.keys())}")


def create_coco_format_simple(image_folder, output_file="annotations.json", min_area=None, top_n=None):
    """
    Versione semplificata che crea il formato simile al tuo esempio.
    Ogni riga è un oggetto immagine separato da virgole.
    """
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = sorted([f for f in os.listdir(image_folder)
                          if f.lower().endswith(extensions)])

    if not image_files:
        print(f"Nessuna immagine trovata in {image_folder}")
        return

    print(f"Trovate {len(image_files)} immagini da processare\n")

    images_data = []

    for image_id, filename in enumerate(image_files, start=1):
        print(f"[{image_id}/{len(image_files)}] Processando: {filename}")
        image_path = os.path.join(image_folder, filename)

        try:
            # Analizza l'immagine
            result = analyze_image(image_path)
            image_info = result['outputs'][0]['predictions']['image']

            images_data.append({
                "id": image_id,
                "width": image_info['width'],
                "height": image_info['height'],
                "file_name": filename,
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            print(f"  ✓ Aggiunta")

        except Exception as e:
            print(f"  ✗ Errore: {e}")
            images_data.append({
                "id": image_id,
                "width": 640,
                "height": 512,
                "file_name": filename,
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    # Salva come array di oggetti (formato semplice)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(images_data, f, indent=0, ensure_ascii=False)

    print(f"\n=== COMPLETATO ===")
    print(f"File salvato: {output_file}")
    print(f"Immagini totali: {len(images_data)}")


# ============= ESECUZIONE =============

if __name__ == "__main__":
    image_folder = "Surgical Visual Understanding (SurgVU) Dataset/frames2"

    # OPZIONE 1: Formato COCO completo (con annotazioni bbox)
    # Decomment per usare:
    create_coco_format(
        image_folder=image_folder,
        output_file="Surgical Visual Understanding (SurgVU) Dataset/2_fps1_coco_new.json",
        top_n=2  # Solo le 3 bbox più grandi per immagine
     )

    # OPZIONE 2: Solo lista immagini (come il tuo esempio)
    # Decomment per usare:
    #create_coco_format_simple(
    #     image_folder=image_folder,
    #     output_file="images_list.json"
    # )

    print("=" * 60)
    print("SCRIPT PRONTO!")
    print("=" * 60)
    print("\nDue modalità disponibili:\n")
    print("1. create_coco_format() - Formato COCO completo")
    print("   Output: {images: [...], annotations: [...], categories: [...]}")
    print("   Include tutte le bbox e le loro coordinate\n")
    print("2. create_coco_format_simple() - Solo lista immagini")
    print("   Output: [{id, width, height, file_name, date_captured}, ...]")
    print("   Come il tuo esempio\n")
    print("Decomment l'opzione che preferisci e modifica image_folder!")
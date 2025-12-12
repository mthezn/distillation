from ultralytics import YOLO
from yolo import YOLOSegmentationFineTuner
from Dataset import ImageMaskDataset,InstrumentDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
# Carica modello pretrained
model = YOLO('yolov8n.pt')  # o yolov8s.pt, yolov8m.pt

# Fine-tuning diretto con il tuo DataLoader

train_transform = A.Compose([
    A.Resize(1024, 1024),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    #A.ColorJitter(p=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])


validation_transform = A.Compose([
    A.Resize(1024, 1024),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])
image_dirs_val = ["/home/mdezen/multiclass/MICCAImod/instrument_1_4_training/validation_dataset_1/left_frames"]
mask_dirs_val = ["/home/mdezen/multiclass/MICCAImod/instrument_1_4_training/validation_dataset_1/ground_truth"
                ]


image_dirs_train = [
    "/home/mdezen/multiclass/MICCAImod/instrument_1_4_training/instrument_dataset_1/left_frames",
    "/home/mdezen/multiclass/MICCAImod/instrument_1_4_training/instrument_dataset_2/left_frames",
    "/home/mdezen/multiclass/MICCAImod/instrument_1_4_training/instrument_dataset_3/left_frames",
    "/home/mdezen/multiclass/MICCAImod/instrument_1_4_training/instrument_dataset_4/left_frames",
    "/home/mdezen/multiclass/MICCAImod/instrument_5_8_training/instrument_dataset_5/left_frames",
    "/home/mdezen/multiclass/MICCAImod/instrument_5_8_training/instrument_dataset_6/left_frames",
    "/home/mdezen/multiclass/MICCAImod/instrument_5_8_training/instrument_dataset_7/left_frames",
    "/home/mdezen/multiclass/MICCAImod/instrument_5_8_training/instrument_dataset_8/left_frames",
]
mask_dirs_train = [

    "/home/mdezen/multiclass/MICCAImod/instrument_1_4_training/instrument_dataset_1/ground_truth",

    "/home/mdezen/multiclass/MICCAImod/instrument_1_4_training/instrument_dataset_2/ground_truth",
    "/home/mdezen/multiclass/MICCAImod/instrument_1_4_training/instrument_dataset_3/ground_truth",
    "/home/mdezen/multiclass/MICCAImod/instrument_1_4_training/instrument_dataset_4/ground_truth",
    "/home/mdezen/multiclass/MICCAImod/instrument_5_8_training/instrument_dataset_5/ground_truth",
    "/home/mdezen/multiclass/MICCAImod/instrument_5_8_training/instrument_dataset_6/ground_truth",

    "/home/mdezen/multiclass/MICCAImod/instrument_5_8_training/instrument_dataset_7/ground_truth",

    "/home/mdezen/multiclass/MICCAImod/instrument_5_8_training/instrument_dataset_8/ground_truth"






    #"/home/mdezen/distillation/MICCAI/instrument_1_4_training/testGT"


]
GLOBAL_CLASS_MAPPING = {
    'Large_Needle_Driver': 1,
    'Prograsp_Forceps': 2,

    'Bipolar_Forceps': 2,#cambia a 2 se vuoi uificare le forceps
    'Grasping_Retractor': 3,
    'Maryland_Bipolar_Forceps': 4,
    'Monopolar_Curved_Scissors': 5,
     #'Other': 7,

    'Vessel_Sealer': 6
}
datasetVal = InstrumentDataset(image_dirs=image_dirs_val,gt_dirs=mask_dirs_val,class_to_id=GLOBAL_CLASS_MAPPING,increase=False)
dataloaderVal = DataLoader(datasetVal,batch_size=2,shuffle=True)
print(len(datasetVal))

#dataset_cholec = CholecDataset(filtered_ds, transform=train_transform)
datasetMiccai = InstrumentDataset(image_dirs=image_dirs_train,gt_dirs=mask_dirs_train,class_to_id=GLOBAL_CLASS_MAPPING,transform=train_transform,increase=False)
print(len(datasetMiccai))
#dataset_finale = ConcatDataset([dataset_cholec, datasetMiccai])

dataloader = DataLoader(datasetMiccai,batch_size=2,shuffle=True,pin_memory=True)

print("\n2 Inizializza Fine-tuner")
print("-" * 70)


    # Crea fine-tuner
finetuner = YOLOSegmentationFineTuner(
        model_path='yolov8n-seg.pt',  # o yolov8s-seg.pt, yolov8m-seg.pt
        num_classes=6,  # Numero di classi (escluso background)
        class_names=[
            'Large_Needle_Driver',
            'Forceps',

            'Grasping_Retractor',
            'Maryland_Bipolar_Forceps',
            'Monopolar_Curved_Scissors',

            'Vessel_Sealer'
        ]
    )



# 3. Converti dataset
print("\n3 Converti Dataset in Formato YOLO")
print("-" * 70)


    # Converte automaticamente il tuo dataset
yaml_config = finetuner.prepare_yolo_dataset(
        train_loader=dataloader,
        val_loader=dataloaderVal,
        output_dir='./yolo_instrument_dataset',
        min_mask_area=100  # Ignora maschere troppo piccole
    )


#4. Training
print("\n4️⃣ Training")
print("-" * 70)


    # Fine-tuning
results = finetuner.train(
        yaml_config=yaml_config,
        epochs=25,
        imgsz=1024,
        batch=16,
        lr0=0.01,
        patience=50,
        project='runs/segment',
        name='instrument_seg'
    )


# 5. Predizione
print("\n5 Predizione")
print("-" * 70)


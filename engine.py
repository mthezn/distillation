"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

from PIL.ImageChops import logical_or
from tqdm import tqdm
import torch
import cv2

import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast


from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
from repvit_sam import SamPredictor

"""
import builtins
import inspect

old_print = builtins.print

def custom_print(*args, **kwargs):
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    old_print(f"[PRINT from {filename}:{lineno}]", *args, **kwargs)

builtins.print = custom_print
"""
def calculate_iou(mask_pred, mask_gt):
    # Ensure the inputs are NumPy arrays
    if isinstance(mask_pred, torch.Tensor):
        mask_pred = mask_pred.cpu().numpy()
    if isinstance(mask_gt, torch.Tensor):
        mask_gt = mask_gt.cpu().numpy()

    # Calculate the intersection (common pixels in both masks)
    intersection = np.logical_and(mask_pred, mask_gt).sum()

    # Calculate the union (all pixels that are 1 in at least one of the masks)
    union = np.logical_or(mask_pred, mask_gt).sum()

    # Calculate IoU (Intersection over Union)
    iou = intersection / union if union != 0 else 0  # Avoid division by zero

    return iou
def set_bn_state(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

def predict_boxes(predictor, boxes):


    masks, _, low_res = predictor.predict_torch(
        # predict_torch serve quando ho le bboxes altrimenti predictor.predict
        point_coords=None,
        point_labels=None,
        boxes=boxes,
        multimask_output=False,
    )
    return masks, _, low_res


def predict_points_boxes_manual(model, image_embedding, boxes, centroids, input_label):
    all_masks = []
    all_scores = []
    all_low_res = []
    model_device = next(model.prompt_encoder.parameters()).device


    for i in range(boxes.shape[0]):
        # Estrai singola box, punto e label
        box = boxes[i].unsqueeze(0).to(device=model_device) # [1, 4]
        point = centroids[:, i, :].unsqueeze(0).to(device=model_device) # [1, 1, 2]
        label = input_label[:, i].unsqueeze(0).to(device = model_device) # [1, 1]

        # Encode prompt: box + point
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=(point, label),
            boxes=box,
            masks=None,
        )

        # Usa mask decoder
        low_res_logits, score = model.mask_decoder(
            image_embeddings=image_embedding,          # [1, C, H', W']
            image_pe=model.prompt_encoder.get_dense_pe(),  # positional encoding
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        # Upscale maschera a risoluzione originale
        mask = model.postprocess_masks(low_res_logits, input_size=(image_embedding.shape[-2], image_embedding.shape[-1]),original_size=(1024, 1024))

        all_masks.append(mask)
        all_scores.append(score)
        all_low_res.append(low_res_logits)

    # Concatenazione dei risultati
    if all_masks == []:
        return torch.zeros((1, 1, 1024, 1024)).to(device=model_device), torch.zeros((1, 1)).to(device=model_device), torch.zeros((1, 1, 1024, 1024)).to(device=model_device)
    final_masks = torch.cat(all_masks, dim=0)  # [N, 1, H, W]
    final_scores = torch.cat(all_scores, dim=0)  # [N, 1]
    final_low_res = torch.cat(all_low_res, dim=0)

    return final_masks, final_scores, final_low_res
def predict_points_boxes(predictor,image,boxes,centroids,input_label):
    all_masks = []
    all_scores = []
    all_low_res = []
    image_array =(image[0].detach().cpu().numpy())
    image_array = np.transpose(image_array, (1, 2, 0))
    predictor.set_image(image_array)
    model_device = next(predictor.model.parameters()).device  # Assicura coerenza col modello

    for i in range(boxes.shape[0]):
        box = boxes[i].unsqueeze(0).to(model_device)  # shape: [1, 4]
        centroid = centroids[:, i, :].unsqueeze(0).to(model_device)  # shape: [1, 1, 2]
        label = input_label[:, i].unsqueeze(0).to(model_device)  # shape: [1, 1]

        masks, scores, low_res = predictor.predict_torch(
            point_coords=centroid,
            point_labels=label,
            boxes=box,
            multimask_output=False
        )

        all_masks.append(masks)
        all_scores.append(scores)
        all_low_res.append(low_res)
    if all_masks == []:
            return torch.zeros((1, 1, 1024, 1024)).to(device=model_device), torch.zeros((1, 1)).to(
                device=model_device), torch.zeros((1, 1, 1024, 1024)).to(device=model_device)
    # Concatenazione dei risultati
    final_masks = torch.cat(all_masks, dim=0)
    final_scores = torch.cat(all_scores, dim=0)
    final_low_res = torch.cat(all_low_res, dim=0)

    return final_masks, final_scores, final_low_res






def train_one_epoch(model,teacher,epoch,criterion,dataloader,optimizer,device,run):
    model.train()
    scaler = torch.amp.GradScaler()
    bar = tqdm(enumerate(dataloader),total=len(dataloader),desc =f"Epoch {epoch}")
    running_loss = 0.0
    dataset_size = 0
    epoch_loss = 0.0
    for i,(images,labels) in bar: #i->batch index, images->batch of images, labels->batch of labels
        optimizer.zero_grad()
        with torch.amp.autocast(device_type = "cuda"):

            images = images.to(device)
            if torch.isnan(images).any():
                print("NaN detected in images!")



        with torch.no_grad():

            outTeach = teacher(images)
        outStud = model(images) #in teoria posso passare n batch di immagini
        torch.cuda.empty_cache()

        if torch.isnan(outTeach).any():
            print("NaN detected in predictions stud!")
        if torch.isnan(outStud).any():
            print("NaN detected in predictions teach!")

        loss = criterion(outTeach,outStud)
        scaler.scale(loss).backward()
        #loss.backward()
        #optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        bar.set_description(f"Loss: {loss.item()}")
        batch_size = images.shape[0]
        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        #epoch_loss += loss.item()
        bar.set_postfix(Epoch = epoch,Train_loss = epoch_loss,LR = optimizer.param_groups[0]['lr'])
        run.log({"train_loss": epoch_loss, "epoch": epoch + 1, "batch": i + 1})
    return epoch_loss

def train_one_epoch_coupled(modelS,predictorS,predictorT,epoch,criterion,dataloader,optimizer,device,run):
    modelS.train()
    predictorT.eval()
    predictorS.eval()
    scaler = torch.amp.GradScaler()
    bar = tqdm(enumerate(dataloader),total=len(dataloader),desc =f"Epoch {epoch}")
    running_loss = 0.0
    dataset_size = 0

    for i,(images,labels) in bar: #i->batch index, images->batch of images, labels->batch of labels
            optimizer.zero_grad()
        #with torch.amp.autocast(device_type = "cuda"):

            images = images.to(device)
            #print(images.shape) #secondo me da rivedere i formati, forse np.array, swap dei canali prima di fare predictor .setimage
            labels = labels.to(device)
            #print(labels.shape)
            results_teach = []
            logits_teach = []
            results_stud = []
            mask = torch.zeros((1,1,1024,1024)).to(device)
            result_mask = []
            gt = []
            for image,label in zip(images,labels):
                # Convert the mask to a binary mask
                label = label.detach().cpu().numpy()
                label = label[0]
                # print("label",label)
                #print(label.shape)
                # Convert to binary mask
                label = (label > 0).astype(np.uint8)
                # Convert to binary mask
                image_array = image.cpu().numpy()

                contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
                # print("contours",contours)

                centroids = []
                input_label = []
                bbox = []
                if contours:
                    for countour in contours:
                        M = cv2.moments(countour)
                        if M["m00"] != 0:
                            centroid_x = int(M["m10"] / M["m00"])
                            centroid_y = int(M["m01"] / M["m00"])
                            centroids.append([centroid_x, centroid_y])
                            input_label.append(1)
                            x, y, w, h = cv2.boundingRect(countour)
                            bbox.append([x, y, x + w, y + h])
                centroids = np.array(centroids)
                #print(centroids)

                bbox = torch.tensor(bbox).float()
                transformed_boxes = predictorT.transform.apply_boxes_torch(bbox, images[0].shape[:2])
                #plt.figure(figsize=(10, 10))
                #plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())

                image_array = np.transpose(image_array,(1,2,0))
                predictorT.set_image(image_array)
                masks, _, low_res_teach = predictorT.predict_torch(
                    # predict_torch serve quando ho le bboxes altrimenti predictor.predict
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
                low_res_teach = modelS.postprocess_masks(low_res_teach, (1024, 1024), (1024, 1024))


                maskunion_teach = torch.zeros((1,1,1024,1024)).to(device)
                for i in range(low_res_teach.shape[0]):
                    mask = masks[i].float() #ricordarsi .foat con Bcelogits
                    logits_teach.append(low_res_teach[i])
                    maskunion_teach = torch.max(maskunion_teach, mask)

                    #print("Min:", masks[i].min().item(), "Max:", masks[i].max().item())
                    #print("Unique:", torch.unique(masks[i]))

                    #print(mask)

                    results_teach.append(mask)
                    if torch.isnan(masks[i]).any():
                        print("NaN detected in predictions teach!")


                image = image.unsqueeze(0)

                image_embeddings = modelS.image_encoder(image)  # -> dict con "image_embed"

                    # 2. Encode prompt (box, punto...)
                sparse_embeddings, dense_embeddings = modelS.prompt_encoder(
                    points=None,
                    boxes=transformed_boxes.to(device),  # shape: [1, 4]
                    masks=None
                )

                # 3. Decode final mask
                low_res_stud, _ = modelS.mask_decoder(
                    image_embeddings=image_embeddings,  # dict
                    image_pe=modelS.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )
                masks = modelS.postprocess_masks(low_res_stud, (1024,1024),(1024,1024))

                #masks = masks > modelS.mask_threshold
                #masks = masks.float()

                maskunion_stud = torch.zeros((1,1,1024,1024)).to(device)
                for i in range(low_res_stud.shape[0]):
                    mask = masks[i].float()

                    maskunion_stud = torch.max(maskunion_stud, mask.float())


                    results_stud.append(mask)
                    #print("Min:", mask.min(), "Max:", mask.max())
                    #mask = torch.logical_or(mask,masks[i])
                    #result_mask = result_mask.append(mask)
                    #print(low_res_stud[i])
                #gt = gt.append(label)


            results_teach = torch.stack(results_teach).to(device)
            logits_teach = torch.stack(logits_teach).to(device)
            results_stud = torch.stack(results_stud).to(device)
            #problem, ho outptud con numero di maschere diverso per ogni tipo di immagine, come allineare le dimensioni?
                #separo ogni maschera in modo che siano tutte 1,1,256,256?
                #print(low_res_stud.shape)
            if torch.isnan(results_teach).any():
                print("NaN detected in predictions stud!")
            if torch.isnan(results_stud).any():
                print("NaN detected in predictions teach!")
            #loss = criterion(results_stud,logits_teach,results_teach)
            #maskunion_stud.requires_grad = True
            loss = criterion(results_stud,results_teach)



            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update progress
            batch_size = images.shape[0]
            running_loss += loss.item() * batch_size
            dataset_size += batch_size
            epoch_loss = running_loss / dataset_size
            bar.set_description(f"Loss: {loss.item()}")
            run.log({"train_loss": epoch_loss, "epoch": epoch + 1, "batch": i + 1})
            bar.set_postfix(Epoch=epoch, Train_loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])

    return epoch_loss

def train_one_epoch_auto(model,student,
                         dataloader,
                         optimizer,
                         device,
                         run,
                         epoch,
                         criterion,
                         ):

    bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
    running_loss = 0.0
    dataset_size = 0
    epoch_loss = 0.0
    scaler = torch.amp.GradScaler()
    predictor = SamPredictor(model)

    for i, (images, labels) in bar:  # i->batch index, images->batch of images, labels->batch of labels



        images = images.to(device)
        labels = labels.to(device)



        results_teach = []
        logits_teach = []
        results_stud = []




        for image, label in zip(images, labels):
            # Convert the mask to a binary mask
            label = label.detach().cpu().numpy()
            label = label[0]
            label = (label > 0).astype(np.uint8)


            image_array = image.cpu().numpy()
            image = image.unsqueeze(0)
            # print(image.shape)

            # Create contours from the gt
            contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]


            centroids = []
            input_label = []
            bbox = []
            if contours:
                for countour in contours:
                    M = cv2.moments(countour)
                    if M["m00"] != 0:
                        centroid_x = int(M["m10"] / M["m00"])
                        centroid_y = int(M["m01"] / M["m00"])
                        centroids.append([centroid_x, centroid_y])
                        input_label.append(1)
                        x, y, w, h = cv2.boundingRect(countour)
                        bbox.append([x, y, x + w, y + h])
                centroids = np.array(centroids)
            # print(centroids)
            bbox = torch.tensor(bbox).float()
            centroids = torch.tensor(centroids).float().unsqueeze(0)



            original_size = tuple(map(int, images[0].shape[-2:]))



            input_label = torch.tensor(input_label, dtype=torch.int64).unsqueeze(0)
            # print(image.shape)


            #image_embedding_model = model.image_encoder(image)  # in teoria posso passare n batch di immagini

            masks_model, _, low_res = predict_points_boxes(predictor,image, bbox, centroids,
                                                           input_label) #masks_model -> binary masks, low_res -> logits

            low_res = model.postprocess_masks(low_res, (1024, 1024), (1024, 1024))
            unique, value = np.unique(low_res.detach().cpu().numpy(), return_index=True)
            #print("unique",unique)
            #print("values", value)


            logits_list = []
            maskunion = torch.zeros((1, 1024, 1024)).to(device)
            for i in range(low_res.shape[0]):
                    mask = masks_model[i].float()  # ricordarsi .foat con Bcelogits

                    maskunion = torch.logical_or(maskunion, mask)


                    logits_list.append(low_res[i]) #devo unire in unica maschera il risultato perche il mio modello teacher produce tante maschere qunati gli strumenti invece il mio modello produce una maschera per immagine


            #creo un unica maschera di logits
            union_logits = torch.full_like(logits_list[0], float('-inf'))
            for logits in logits_list:
                union_logits = torch.maximum(union_logits, logits)





            results_teach.append(union_logits)


            image_embeddings = student.image_encoder(image)  # -> dict con "image_embed"


            

            low_res_stud, _ = student.mask_decoder(
                    image_embeddings=image_embeddings,  # dict
                    image_pe=student.prompt_encoder.get_dense_pe(),


                    multimask_output=False
                ) #low_res_stud -> logits
                

            #low_res_stud = student.mask_decoder(image_embeddings)
            low_res_stud = student.postprocess_masks(low_res_stud, (1024, 1024), (1024, 1024))
            mask = low_res_stud > student.mask_threshold
            #iou = calculate_iou(mask, maskunion)
            #print("iou", iou)

                #maskunion_stud = torch.zeros((1, 1, 1024, 1024)).to(device)
            for i in range(low_res_stud.shape[0]):
                    low_res_stud_temp = low_res_stud[i].float()
                    #maskunion_stud = torch.max(maskunion_stud, mask.float())

                    results_stud.append(low_res_stud_temp)


        results_teach = torch.stack(results_teach).to(device)
        target = torch.sigmoid(results_teach.detach())
        results_stud = torch.stack(results_stud).to(device)
        unique,value = np.unique(results_teach.detach().cpu().numpy(),return_index=True)
        #print("uniqueTeach",unique)
        #print("valueTeach",value)
        unique,value = np.unique(results_stud.detach().cpu().numpy(),return_index=True)
        #print("uniqueStud",unique)


        with torch.amp.autocast(device_type="cuda"):
            loss = criterion(results_stud, results_teach)
            #print("loss", loss)

        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            print(f"Skipping step at batch {i} due to non-finite loss: {loss}")
            optimizer.zero_grad(set_to_none=True)
            continue  # salta al batch successivo

            # Update progress
        batch_size = images.shape[0]
        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        bar.set_description(f"Loss: {loss.item()}")
        run.log({"train_loss": epoch_loss, "epoch": epoch + 1, "batch": i + 1})
        bar.set_postfix(Epoch=epoch, Train_loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])
    return epoch_loss




@torch.no_grad()
def evaluate(data_loader, model,teacher, device):
    criterion = torch.nn.MSELoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)


        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            outputTeacher = teacher(images)
            loss = criterion(output, outputTeacher)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
         .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def validate_one_epoch(
    model,              # student
    teacher,            # teacher (SAM encoder)
    dataloader,         # validation DataLoader
    criterion,          # es. MSELoss o CosineSimilarity
    device,             # "cuda"
    epoch,               # epoch corrente (per logging)
    run
):
    model.eval()
    teacher.eval()

    running_loss = 0.0
    dataset_size = 0

    bar = tqdm(dataloader, desc=f"[Val] Epoch {epoch}", leave=False)

    with torch.no_grad():
        for i, (images, _) in enumerate(bar):
            images = images.to(device)

            with torch.autocast(device_type="cuda"):
                teacher_out = teacher(images)
                student_out = model(images)
                loss = criterion(student_out, teacher_out)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size
            epoch_loss = running_loss / dataset_size

            bar.set_postfix(Val_Loss=f"{epoch_loss:.4f}")
            run.log({"val_loss": epoch_loss, "epoch": epoch + 1, "batch": i + 1})

    return epoch_loss


def validate_one_epoch_coupled(
    model,              # student
    teacher,            # teacher
    dataloader,         # validation DataLoader
    criterion,          # es. MSELoss o CosineSimilarity
    device,             # "cuda"
    epoch,               # epoch corrente (per logging)
    run
):
    model.eval()
    teacher.eval()

    running_loss = 0.0
    dataset_size = 0

    bar = tqdm(dataloader, desc=f"[Val] Epoch {epoch}", leave=False)

    with torch.no_grad():
        for i, (images, labels) in enumerate(bar):
            images = images.to(device)
            labels = labels.to(device)
            results_teach = []

            results_stud = []
            for image,label in zip(images,labels):
                # Convert the mask to a binary mask
                label = label.detach().cpu().numpy()
                label = label[0]
                # print("label",label)
                # print(label.shape)
                # Convert to binary mask
                label = (label > 0).astype(np.uint8)
                # Convert to binary mask
                image_array = image.cpu().numpy()

                contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
                # print("contours",contours)

                centroids = []
                input_label = []
                bbox = []
                if contours:
                    for countour in contours:
                        M = cv2.moments(countour)
                        if M["m00"] != 0:
                            centroid_x = int(M["m10"] / M["m00"])
                            centroid_y = int(M["m01"] / M["m00"])
                            centroids.append([centroid_x, centroid_y])
                            input_label.append(1)
                            x, y, w, h = cv2.boundingRect(countour)
                            bbox.append([x, y, x + w, y + h])
                centroids = np.array(centroids)
                # print(centroids)

                bbox = torch.tensor(bbox).float()
                transformed_boxes = teacher.transform.apply_boxes_torch(bbox, images[0].shape[:2])
                # plt.figure(figsize=(10, 10))
                # plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())

                image_array = np.transpose(image_array, (1, 2, 0))
                teacher.set_image(image_array)
                masks, _, low_res_teach = teacher.predict_torch(
                    # predict_torch serve quando ho le bboxes altrimenti predictor.predict
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
                for i in range(low_res_teach.shape[0]):
                    mask = masks[i].float()
                    results_teach.append(mask)
                    if torch.isnan(low_res_teach[i]).any():
                        print("NaN detected in predictions teach!")

                # print(low_res_teach.shape)
                # print(low_res_teach)


                image = image.unsqueeze(0)

                image_embeddings = model.image_encoder(image)  # -> dict con "image_embed"

                # 2. Encode prompt (box, punto...)
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=transformed_boxes.to(device),  # shape: [1, 4]
                    masks=None
                )

                # 3. Decode final mask
                low_res_stud, _ = model.mask_decoder(
                    image_embeddings=image_embeddings,  # dict
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )
                low_res_stud = model.postprocess_masks(low_res_stud, (1024, 1024), (1024, 1024))
                for i in range(low_res_stud.shape[0]):

                    results_stud.append(low_res_stud[i])
                    # print(low_res_stud[i])

            results_teach = torch.stack(results_teach).to(device)
            results_stud = torch.stack(results_stud).to(
                device)  # problem, ho outptud con numero di maschere diverso per ogni tipo di immagine, come allineare le dimensioni?
            # separo ogni maschera in modo che siano tutte 1,1,256,256?
            # print(low_res_stud.shape)
            loss = criterion(results_stud, results_teach)



            # Update progress
            batch_size = images.shape[0]
            running_loss += loss.item() * batch_size
            dataset_size += batch_size
            epoch_loss = running_loss / dataset_size
            bar.set_postfix(Val_Loss=f"{epoch_loss:.4f}")
            run.log({"val_loss": epoch_loss, "epoch": epoch + 1, "batch": i + 1})





    return epoch_loss

def validate_one_epoch_auto(
    model,              # student
    student,            # teacher
    dataloader,         # validation DataLoader
    criterion,          # es. MSELoss o CosineSimilarity
    device,             # "cuda"
    epoch,               # epoch corrente (per logging)
    run
):
    model.eval()
    student.eval()

    running_loss = 0.0
    dataset_size = 0
    predictor = SamPredictor(model)

    bar = tqdm(enumerate(dataloader), desc=f"[Val] Epoch {epoch}", leave=False)

    with torch.no_grad():


        for i, (images, labels) in bar:  # i->batch index, images->batch of images, labels->batch of labels



            images = images.to(device)

            labels = labels.to(device)

            results_teach = []
            logits_teach = []
            results_stud = []

            for image, label in zip(images, labels):
                # Convert the label to a binary mask
                label = label.detach().cpu().numpy()
                label = label[0]
                label = (label > 0).astype(np.uint8)


                image_array = image.cpu().numpy()
                image = image.unsqueeze(0)
               # print(image.shape)

                contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
                # print("contours",contours)

                centroids = []
                input_label = []
                bbox = []
                if contours:
                    for countour in contours:
                        M = cv2.moments(countour)
                        if M["m00"] != 0:
                            centroid_x = int(M["m10"] / M["m00"])
                            centroid_y = int(M["m01"] / M["m00"])
                            centroids.append([centroid_x, centroid_y])
                            input_label.append(1)
                            x, y, w, h = cv2.boundingRect(countour)
                            bbox.append([x, y, x + w, y + h])
                centroids = np.array(centroids)
                # print(centroids)

                bbox = torch.tensor(bbox).float()
                centroids = torch.tensor(centroids).float().unsqueeze(0)




                original_size = tuple(map(int, images[0].shape[-2:]))



                input_label = torch.tensor(input_label, dtype=torch.int64).unsqueeze(0)


                image_embedding_model = model.image_encoder(image)  # in teoria posso passare n batch di immagini
                masks_model, _, low_res = predict_points_boxes(predictor, image, bbox, centroids,
                                                               input_label)
                low_res = model.postprocess_masks(low_res, (1024, 1024), (1024, 1024))

                maskunion_teach = torch.zeros(( 1, 1024, 1024)).to(device)
                for i in range(low_res.shape[0]):
                    mask = masks_model[i].float()
                                                 # ricordarsi .foat con Bcelogits
                    logits_teach.append(low_res[i])
                    maskunion_teach = torch.logical_or(maskunion_teach, mask)


                union_logits = torch.full_like(logits_teach[0], float('-inf'))
                for logits in logits_teach:
                    union_logits = torch.maximum(union_logits, logits)
                results_teach.append(union_logits)

                image_embeddings = student.image_encoder(image)

                    # 3. Decode final mask
                low_res_stud, _ = student.mask_decoder(
                    image_embeddings=image_embeddings,  # dict
                    image_pe=student.prompt_encoder.get_dense_pe(),

                    multimask_output=False
                )
                low_res_stud = student.postprocess_masks(low_res_stud, (1024, 1024), (1024, 1024)) 

                #print(image_embeddings.shape)
                #low_res_stud = student.mask_decoder(image_embeddings)



                for i in range(low_res_stud.shape[0]):
                    low_res_temp = low_res_stud[i].float()
                    results_stud.append(low_res_temp)



            results_teach = torch.stack(results_teach).to(device)
            target = torch.sigmoid(results_teach.detach()) #for BCE with logits loss
            logits_teach = torch.stack(logits_teach).to(device)
            results_stud = torch.stack(results_stud).to(device)


            loss = criterion(results_stud, results_teach)



            # Update progress
            batch_size = images.shape[0]
            running_loss += loss.item() * batch_size
            dataset_size += batch_size
            epoch_loss = running_loss / dataset_size
            bar.set_description(f"Loss: {loss.item()}")
            run.log({"val_loss": epoch_loss, "epoch": epoch + 1, "batch": i + 1})
            bar.set_postfix(Epoch=epoch, Val_loss=epoch_loss)
        return epoch_loss


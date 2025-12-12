"""
Train and eval functions used in mainDecoupled.py, mainAuto.py
"""
import math
import random
import sys
from typing import Iterable, Optional

from PIL.ImageChops import logical_or
from PIL.ImagePalette import random
from tqdm import tqdm
import torch
import cv2

import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import os
from PIL import Image
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import random
from losses import DistillationLoss
import utils
from repvit_sam import SamPredictor

from utility import save_binary_mask, calculate_iou, predict_points_boxes,predict_boxes,get_bbox_centroids

def train_one_epoch(model,
                    teacher,
                    epoch,
                    criterion,
                    dataloader,
                    optimizer,
                    device,
                    run):
    """
    Function: train_one_epoch

    Purpose:
        Trains a student model for one epoch using knowledge distillation from a fixed teacher model (e.g., SAM).
        The student is trained to mimic the teacher’s encoder output via MSE LOSS.

    Arguments:
        model (torch.nn.Module):
            The student model being trained. It can be a lighter encoder-decoder architecture.

        teacher (torch.nn.Module):
            The pre-trained teacher model (e.g., a full SAM model) used to generate target outputs. It is frozen (no gradients).

        epoch (int):
            Current training epoch number, used for logging and display.

        criterion (torch.nn.Module):
            A loss function (e.g., MSELoss) used to minimize the difference between teacher and student outputs.

        dataloader (torch.utils.data.DataLoader):
            DataLoader that provides batches of (images, masks). Labels may not be used directly if only encoder distillation is performed.

        optimizer (torch.optim.Optimizer):
            Optimizer used to update the student model's parameters.

        device (str or torch. device):
            Device on which computations are performed (e.g., 'cuda' or 'cpu').

        run (object):
            Logging object (e.g., from Weights & Biases) used to track training loss and metadata.


    Returns:
        epoch_loss (float):
            The average loss over the entire epoch, useful for monitoring training convergence.
    """


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

            outTeach = teacher(images)#[B,3,1024,1024]
        outStud = model(images)#[B,3,1024,1024]
        torch.cuda.empty_cache()

        if torch.isnan(outTeach).any():
            print("NaN detected in predictions stud!")
        if torch.isnan(outStud).any():
            print("NaN detected in predictions teach!")

        loss = criterion(outTeach,outStud)
        scaler.scale(loss).backward()

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


def train_one_epoch_fine(model,

                         dataloader,
                         optimizer,
                         device,
                         run,
                         epoch,
                         criterion
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


        results_stud = []
        label_list = []

        for image, label in zip(images, labels):
            # Convert the mask to a binary mask
            label = label.detach().cpu().numpy()
            #label = label.unsqueeze()  # Assicurati che sia 2D
            label = (label > 0).astype(np.uint8)  # Assicurati che sia 2D
            label_list.append(label)


            image_array = image.cpu().numpy()
            image = image.unsqueeze(0)  # B, C, H, W



            image_embeddings = model.image_encoder(image)  # -> dict con "image_embed"
            #low_res_stud, _ = model.mask_decoder(
             #   image_embeddings=image_embeddings,  # dict
              #  image_pe=model.prompt_encoder.get_dense_pe(),

               # multimask_output=False
            #)  # low_res_stud -> logits
            low_res_stud = model.mask_decoder(image_embeddings)  #versione per rete unet
            low_res_stud = model.postprocess_masks(low_res_stud, (1024, 1024), (1024, 1024))
            mask = low_res_stud > model.mask_threshold
            iou = calculate_iou(mask, label)



            for i in range(low_res_stud.shape[0]):
                low_res_stud_temp = low_res_stud[i].float()
                # maskunion_stud = torch.max(maskunion_stud, mask.float())

                results_stud.append(low_res_stud_temp)


        results_label = torch.stack([torch.tensor(label) for label in label_list]).to(device)
        results_stud = torch.stack(results_stud).to(device)

        loss = criterion(results_stud,results_label.float())
        # print("loss", loss)

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


def train_one_epoch_auto(model,
                         student,
                         dataloader,
                         optimizer,
                         device,
                         run,
                         epoch,
                         criterion
                         ):
    """
    Function: train_one_epoch_auto

    Purpose:
        Trains a student segmentation model via **automatic self-distillation** from a frozen **SAM teacher model**.
        This version does **not rely on manual prompts** but derives prompts (points, boxes) directly from ground truth masks.

    Arguments:
        model (SamModel or nn.Module):
            The frozen SAM model acting as the teacher, providing high-quality masks via prompt-based inference.

        student (nn.Module):
            The trainable model being optimized to replicate SAM's output. Must implement `image_encoder` and `mask_decoder`.

        dataloader (torch.utils.data.DataLoader):
            Yields batches of (images, ground truth masks). GT masks are used to generate automatic prompts (centroids, bounding boxes).

        optimizer (torch.optim.Optimizer):
            Optimizer for the student model.

        device (str or torch.device):
            Computation device (e.g., "cuda" or "cpu").

        run (object):
            Logger object (e.g., from Weights & Biases) to track training loss and epoch progress.

        epoch (int):
            Current training epoch index, used for progress reporting and logging.

        criterion (torch.nn.Module):
            Loss function used to compare student and teacher predictions (e.g., MSE or BCEWithLogitsLoss).



    Returns:
        epoch_loss (float):
            Average loss over all batches in the epoch, used for convergence monitoring.



    Use Case:
        Ideal for **automated self-supervised distillation** of a large model like SAM into a compact segmentation network.
        Allows leveraging SAM’s high-quality masks without manual annotations by bootstrapping prompts from rough GT masks.

    """

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
            label = (label > 0).astype(np.uint8)



            image_array = image.cpu().numpy()
            image = image.unsqueeze(0) #B, C, H, W

            centroids,bbox,input_label = get_bbox_centroids(label,5)


            bbox = torch.tensor(bbox).float()
            centroids = torch.tensor(centroids).float().unsqueeze(0)



            original_size = tuple(map(int, images[0].shape[-2:]))
            input_label = torch.tensor(input_label, dtype=torch.int64).unsqueeze(0)
            masks_model, _, low_res = predict_points_boxes(predictor,image,bbox,centroids,input_label) #masks_model -> binary masks, low_res -> logits

            low_res = model.postprocess_masks(low_res, (1024, 1024), (1024, 1024))


            logits_list = []
            maskunion = torch.zeros((1, 1024, 1024)).to(device)

            for i in range(low_res.shape[0]):
                    mask = masks_model[i]  # ricordarsi .foat con Bcelogits
                    maskunion = torch.logical_or(maskunion, mask)
                    logits_list.append(low_res[i]) #devo unire in unica maschera il risultato perche il mio modello teacher produce tante maschere qunati gli strumenti invece il mio modello produce una maschera per immagine

            #creo un unica maschera di logits
            union_logits = torch.full_like(logits_list[0], float('-inf'))
            for logits in logits_list:
                union_logits = torch.maximum(union_logits, logits)

            results_teach.append(maskunion)


            image_embeddings = student.image_encoder(image)  # -> dict con "image_embed"
            #low_res_stud, _ = student.mask_decoder(
                   # image_embeddings=image_embeddings,  # dict
                    #image_pe=student.prompt_encoder.get_dense_pe(),


                    #multimask_output=False
                #) #low_res_stud -> logits
                
            low_res_stud = student.mask_decoder(image_embeddings)  # versione per rete unet
            low_res_stud = student.postprocess_masks(low_res_stud, (1024, 1024), (1024, 1024))
            mask = low_res_stud > student.mask_threshold
            iou = calculate_iou(mask, maskunion)


            if iou > 0.89:
                #save_binary_mask(low_res_stud.detach().cpu().numpy() >0 , epoch, random.randint(0,100), output_dir="binary_masks_train")

                print(f"Saved binary mask for epoch {epoch}, batch {i}")


            for i in range(low_res_stud.shape[0]):
                    low_res_stud_temp = low_res_stud[i].float()
                    #maskunion_stud = torch.max(maskunion_stud, mask.float())

                    results_stud.append(low_res_stud_temp )


        results_teach = torch.stack(results_teach).to(device)
        target = torch.sigmoid(results_teach.detach())
        results_stud = torch.stack(results_stud).to(device)


        loss = criterion(results_stud, results_teach.float())
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






def validate_one_epoch(
    model,
    teacher,
    dataloader,
    criterion,
    device,
    epoch,
    run
):
    """
    Function: validate_one_epoch

    Purpose:
        Evaluates the performance of the student model during a single validation epoch
        by comparing its output against that of a frozen teacher model (e.g., SAM).
        This is typically used during knowledge distillation, where the student learns
        to mimic the teacher.The outut of SAM image encoder is compared with the output of the student image encoder.

    Arguments:
        model (torch.nn.Module):
            The student model being trained, whose predictions are evaluated here.

        teacher (torch.nn.Module):
            The pretrained and frozen teacher model (e.g., SAM) providing target outputs.
            Assumed to output logits or masks from input images.

        dataloader (torch.utils.data.DataLoader):
            Validation data loader yielding batches of images. Masks are ignored here since
            the student is compared against the teacher.

        criterion (torch.nn.Module):
            Loss function used to measure similarity between student and teacher predictions.
            Typically a regression-based loss (e.g., MSELoss, BCEWithLogitsLoss).

        device (str or torch.device):
            Target device where the computation will be performed ("cuda" or "cpu").

        epoch (int):
            Current validation epoch number, used for logging and tracking.

        run (object):
            Logging object (e.g., from Weights & Biases or other experiment tracker).
            Used to store validation metrics.

    Returns:
        epoch_loss (float):
            The average validation loss across the full validation set.


    """

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


def validate_one_epoch_fine(model,

                         dataloader,

                         device,
                         run,
                         epoch,
                         criterion
                         ):
    model.eval()


    running_loss = 0.0
    dataset_size = 0
    predictor = SamPredictor(model)

    bar = tqdm(enumerate(dataloader), desc=f"[Val] Epoch {epoch}", leave=False)

    with torch.no_grad():

        for i, (images, labels) in bar:  # i->batch index, images->batch of images, labels->batch of labels
            images = images.to(device)

            labels = labels.to(device)
            label_list = []

            results_stud = []

            for image, label in zip(images, labels):
                # Convert the label to a binary mask
                label = label.detach().cpu().numpy()

                label = (label > 0).astype(np.uint8)
                label_list.append(label)

                image_array = image.cpu().numpy()
                image = image.unsqueeze(0)



                image_embeddings = model.image_encoder(image)
                """
                low_res_stud, _ = model.mask_decoder(
                    image_embeddings=image_embeddings,  # dict
                    image_pe=model.prompt_encoder.get_dense_pe(),

                    multimask_output=False
                )"""
                low_res_stud = model.mask_decoder(image_embeddings)  # versione per rete unet
                low_res_stud = model.postprocess_masks(low_res_stud, (1024, 1024), (1024, 1024))

                for i in range(low_res_stud.shape[0]):
                    low_res_temp = low_res_stud[i].float()
                    results_stud.append(low_res_temp)

            result_label = torch.stack([torch.tensor(label) for label in label_list]).to(device)

             # for BCE with logits loss

            results_stud = torch.stack(results_stud).to(device)

            loss = criterion(results_stud, result_label.float())

            # Update progress
            batch_size = images.shape[0]
            running_loss += loss.item() * batch_size
            dataset_size += batch_size
            epoch_loss = running_loss / dataset_size
            bar.set_description(f"Loss: {loss.item()}")
            run.log({"val_loss": epoch_loss, "epoch": epoch + 1, "batch": i + 1})
            bar.set_postfix(Epoch=epoch, Val_loss=epoch_loss)

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
    """
       Function: validate_one_epoch

       Purpose:
           Evaluates the performance of the AutoSam during a single validation epoch
           by comparing its output against that of a distilled Model with prompts.
           This is typically used during **knowledge distillation**, where the student learns
           to mimic the teacher.

       Arguments:
           model (torch.nn.AutoSam):
               The **student model** being trained, whose predictions are evaluated here.

           teacher (torch.nn.Module):
              The distilled model from SAM providing target outputs.
               Assumed to output logits or masks from input images.

           dataloader (torch.utils.data.DataLoader):
               Validation data loader yielding batches of images. Masks are ignored here since
               the student is compared against the teacher.

           criterion (torch.nn.Module):
               Loss function used to measure similarity between student and teacher predictions.
               Typically a regression-based loss (e.g., MSELoss, BCEWithLogitsLoss).

           device (str or torch.device):
               Target device where the computation will be performed ("cuda" or "cpu").

           epoch (int):
               Current validation epoch number, used for logging and tracking.

           run (object):
               Logging object (e.g., from Weights & Biases or other experiment tracker).
               Used to store validation metrics.

       Returns:
           epoch_loss (float):
               The average validation loss across the full validation set.


       """
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
                label = (label > 0).astype(np.uint8)


                image_array = image.cpu().numpy()
                image = image.unsqueeze(0)

                centroids,bbox, input_label = get_bbox_centroids(label, 5)



                bbox = torch.tensor(bbox).float()
                centroids = torch.tensor(centroids).float().unsqueeze(0)

                original_size = tuple(map(int, images[0].shape[-2:]))



                input_label = torch.tensor(input_label, dtype=torch.int64).unsqueeze(0)

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

                iou = calculate_iou(maskunion_teach, low_res.detach().cpu().numpy()>0)

                if iou > 0.9:
                    #save_binary_mask((low_res.detach().cpu().numpy() > 0), epoch, random.randint(0,100), output_dir="binary_masks_validation")

                    print(f"Saved binary mask for epoch {epoch}, batch {i}")
                results_teach.append(maskunion_teach)

                image_embeddings = student.image_encoder(image)

                #low_res_stud, _ = student.mask_decoder(
                   # image_embeddings=image_embeddings,  # dict
                    #image_pe=student.prompt_encoder.get_dense_pe(),

                    #multimask_output=False
                #)

                low_res_stud = student.mask_decoder(image_embeddings)  # versione per rete unet
                low_res_stud = student.postprocess_masks(low_res_stud, (1024, 1024), (1024, 1024)) 

                for i in range(low_res_stud.shape[0]):
                    low_res_temp = low_res_stud[i].float()
                    results_stud.append(low_res_temp )


            results_teach = torch.stack(results_teach).to(device)
            target = torch.sigmoid(results_teach.detach()) #for BCE with logits loss
            logits_teach = torch.stack(logits_teach).to(device)
            results_stud = torch.stack(results_stud).to(device)


            loss = criterion(results_stud, results_teach.float())



            # Update progress
            batch_size = images.shape[0]
            running_loss += loss.item() * batch_size
            dataset_size += batch_size
            epoch_loss = running_loss / dataset_size
            bar.set_description(f"Loss: {loss.item()}")
            run.log({"val_loss": epoch_loss, "epoch": epoch + 1, "batch": i + 1})
            bar.set_postfix(Epoch=epoch, Val_loss=epoch_loss)

        return epoch_loss


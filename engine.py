"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
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
def set_bn_state(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

# def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler,
#                     clip_grad: float = 0,
#                     clip_mode: str = 'norm',
#                     model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
#                     set_training_mode=True,
#                     set_bn_eval=False,):
#     model.train(set_training_mode)
#     if set_bn_eval:
#         set_bn_state(model)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(
#         window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 100
#
#     for samples, targets in metric_logger.log_every(
#             data_loader, print_freq, header):
#         samples = samples.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)
#
#         if mixup_fn is not None:
#             samples, targets = mixup_fn(samples, targets)
#
#         with torch.cuda.amp.autocast():
#             outputs = model(samples)
#             loss = criterion(samples, outputs, targets)
#
#         loss_value = loss.item()
#
#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             sys.exit(1)
#
#         optimizer.zero_grad()
#
#         # this attribute is added by timm on one optimizer (adahessian)
#         is_second_order = hasattr(
#             optimizer, 'is_second_order') and optimizer.is_second_order
#         loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
#                     parameters=model.parameters(), create_graph=is_second_order)
#
#         torch.cuda.synchronize()
#         if model_ema is not None:
#             model_ema.update(model)
#
#         metric_logger.update(loss=loss_value)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

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
        with torch.amp.autocast(device_type = "cuda"):

            images = images.to(device)
            #print(images.shape) #secondo me da rivedere i formati, forse np.array, swap dei canali prima di fare predictor .setimage
            labels = labels.to(device)
            #print(labels.shape)
            results_teach = []
            results_stud = []
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
                for i in range(low_res_teach.shape[0]):

                    results_teach.append(low_res_teach[i])
                    if torch.isnan(low_res_teach[i]).any():
                        print("NaN detected in predictions teach!")

                #print(low_res_teach.shape)
                #print(low_res_teach)
                """
                predictorS.set_image(image)

                masks, _, low_res_stud = predictorS.predict_torch(
                    # predict_torch serve quando ho le bboxes altrimenti predictor.predict
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                ) 
                """
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


                for i in range(low_res_stud.shape[0]):

                    results_stud.append(low_res_stud[i])
                    #print(low_res_stud[i])


            results_teach = torch.stack(results_teach).to(device)
            results_stud = torch.stack(results_stud).to(device)#problem, ho outptud con numero di maschere diverso per ogni tipo di immagine, come allineare le dimensioni?
                #separo ogni maschera in modo che siano tutte 1,1,256,256?
                #print(low_res_stud.shape)
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

                    results_teach.append(low_res_teach[i])
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
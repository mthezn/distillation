
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

def validate_one_epoch_coupled(
    model,
    teacher,
    dataloader,
    criterion,
    device,
    epoch,
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
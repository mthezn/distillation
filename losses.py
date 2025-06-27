"""
Implements the knowledge distillation loss, proposed in deit
"""
import torch
from torch.nn import functional as F
from torch import nn


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, #teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        #self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, teacher_outputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        #with torch.no_grad():
            #teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(
                outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


def bce_soft_hard(student_logits, teacher_logits, T=2.0, alpha=0.7):
    # Teacher produce probabilità morbide (soft label)
    teacher_probs = torch.sigmoid(teacher_logits / T)

    # BCE tra student logits e soft label del teacher
    bce_soft = F.binary_cross_entropy_with_logits(student_logits, teacher_probs)

    # BCE supervision con logits del teacher come ground truth hard
    bce_hard = F.binary_cross_entropy_with_logits(student_logits, (teacher_logits > 0).float())

    # Loss combinata
    return alpha * bce_soft + (1 - alpha) * bce_hard

def distillation_loss(student_logits, teacher_logits, T=2.0, alpha=0.7):

    # Applica sigmoid con temperatura per ottenere probabilità "soft"
    student_probs = torch.sigmoid(student_logits / T).clamp(min=1e-6, max=1 - 1e-6)
    teacher_probs = torch.sigmoid(teacher_logits / T).clamp(min=1e-6, max=1 - 1e-6)

    # KL Divergence tra distribuzioni soft (per ogni pixel e canale)
    #kl_div = F.kl_div(torch.log(student_probs + 1e-6),teacher_probs.log(), reduction='batchmean',log_target=True) * (T**2) / student_logits.numel()
    kl_div = (teacher_probs * torch.log(teacher_probs / student_probs) +
              (1 - teacher_probs) * torch.log((1 - teacher_probs) / (1 - student_probs)))


    kl_loss = ((kl_div) * T**2 )/ student_logits.numel()
    kl_loss = kl_loss.mean()


    bce = F.binary_cross_entropy_with_logits(student_logits, teacher_probs)


    # Loss combinata
    return alpha * kl_loss + (1 - alpha) * bce

def dice_loss(student_masks, teacher_masks):
    student_probs = torch.sigmoid(student_masks)  # Se student_masks sono logits
    #student_probs  =  student_masks > 0.0
    dice_loss_total = 0
    bce_loss_total = 0
    N = teacher_masks.shape[0]  # Numero di maschere (canali)
    #print("teacher_masks.shape", teacher_masks.shape)
    #print("student_probs.shape", student_probs.shape)
    #print(N)
    for i in range(N):
        s = student_probs[i, :, :, :]  # Maschera del modello studente
        #print(s.shape)
        if( teacher_masks.ndim == 3):
            teacher_masks = teacher_masks.unsqueeze(1)
        t = teacher_masks[i, :, :, :]
        #print(t.shape)
        # Dice Loss
        intersection = (s * t).sum(dim=(1, 2))  # Somma su H, W
        union = s.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        dice = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        # BCE Loss
        bce = F.binary_cross_entropy(s, t, reduction='none').mean(dim=(1, 2))

        dice_loss_total += dice
        bce_loss_total += bce

    return 0.5 * dice_loss_total.mean() + 0.5 * bce_loss_total.mean()


def iou_loss(student_masks, teacher_masks, eps=1e-6):
    #student_probs = torch.sigmoid(student_masks)  # Se sono logits
    if teacher_masks.ndim == 3:
        teacher_masks = teacher_masks.unsqueeze(1)  # [N, 1, H, W]

    iou_losses = []

    N = teacher_masks.shape[0]
    for i in range(N):
        s = student_masks[i]  # [1, H, W]
        t = teacher_masks[i]  # [1, H, W]

        intersection = (s * t).sum(dim=(1, 2))  # somma su H, W
        union = (s + t - s * t).sum(dim=(1, 2))

        iou = (intersection + eps) / (union + eps)
        loss = 1 - iou  # per maschera
        iou_losses.append(loss)

    return torch.stack(iou_losses).mean()
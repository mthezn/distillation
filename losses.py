"""
Implements the knowledge distillation loss, proposed in deit
"""
import torch
from torch.nn import functional as F
from torch import nn


#########FINE TUNING LOSSES##########
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, 
                 dice_smooth=1.0, focal_alpha=0.25, focal_gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.focal_loss = ImprovedFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)
        
        # Combine losses with weights
        total_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss
        
        # Check for NaN values and handle them
        if torch.isnan(total_loss) or torch.isnan(dice_loss) or torch.isnan(focal_loss):
            print(f"Warning: NaN detected in loss calculation!")
            print(f"Dice loss: {dice_loss.item()}, Focal loss: {focal_loss.item()}")
            print(f"Pred stats - min: {pred.min().item():.6f}, max: {pred.max().item():.6f}")
            print(f"Target stats - min: {target.min().item():.6f}, max: {target.max().item():.6f}")
            
            # Return a fallback loss to prevent training from crashing
            return torch.tensor(1.0, device=pred.device, requires_grad=True), dice_loss, focal_loss
        
        return total_loss, dice_loss, focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, preds, targets):
        #preds = torch.sigmoid(preds)  # Ensure predictions are in [0,1]
        preds = torch.clamp(torch.sigmoid(preds), min=1e-7, max=1 - self.eps)
        preds = preds.view(-1)
        targets = targets.view(-1).float()

        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()

        # Dice coefficient calculation
        if union == 0:
            return torch.tensor(0.0, device=preds.device)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        dice_loss = 1. - dice

        return torch.clamp(dice_loss, min=0.0, max=1.0)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1.0, logits=True, reduce=True, foreground_only=False, dynamic_alpha = True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.foreground_only = foreground_only 
        self.dynamic_alpha = dynamic_alpha

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)
        
        if self.dynamic_alpha:
            # Create foreground/background masks
            foreground = (targets == 1).float()
            background = (targets == 0).float()

            fg_pixels = foreground.sum()
            bg_pixels = background.sum()
            total_pixels = fg_pixels + bg_pixels + 1e-8  # small epsilon to avoid div/0

            # If either class is missing, fallback to neutral weights (1.0)
            if fg_pixels == 0 or bg_pixels == 0:
                alpha = torch.ones_like(targets)  # fallback to no reweighting
            else:
                fg_ratio = fg_pixels / total_pixels
                bg_ratio = bg_pixels / total_pixels
                alpha = foreground * bg_ratio + background * fg_ratio
        else:
            alpha = self.alpha

        F_loss = alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.foreground_only:
            # Mask only target == 1 (foreground)
            foreground_mask = (targets == 1)
            F_loss = F_loss * foreground_mask.float()

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class ImprovedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-8):
        super(ImprovedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        probs = torch.clamp(probs, min=self.eps, max=1 - self.eps)
        
        # Ensure targets are float
        targets = targets.float()
        
        # Calculate cross entropy manually for better numerical stability
        log_probs = torch.log(probs + self.eps)
        log_one_minus_probs = torch.log(1 - probs + self.eps)
        
        # Binary cross entropy
        bce = -(targets * log_probs + (1 - targets) * log_one_minus_probs)
        
        # Calculate pt for focal term
        pt = torch.where(targets == 1, probs, 1 - probs)
        pt = torch.clamp(pt, min=self.eps, max=1 - self.eps)
        
        # Calculate alpha term
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Calculate focal loss
        focal_weight = alpha_t * torch.pow(1 - pt, self.gamma)
        focal_loss = focal_weight * bce
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0, eps=1e-7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # weight for false positives
        self.beta = beta    # weight for false negatives
        self.smooth = smooth
        self.eps = eps
    
    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = torch.clamp(preds, min=self.eps, max=1 - self.eps)
        
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1).float()
        
        # True Positives, False Positives & False Negatives
        TP = (preds_flat * targets_flat).sum()
        FP = (preds_flat * (1 - targets_flat)).sum()
        FN = ((1 - preds_flat) * targets_flat).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky
                
def create_loss_criterion_old(a=0.5, b=0.5):
    dice_loss_fn = DiceLoss(smooth=1.0)
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=1.0, logits=True, foreground_only=False, dynamic_alpha=True)

    def loss_criterion(pred, target):
        dice_loss = dice_loss_fn(pred, target)
        focal_loss = focal_loss_fn(pred, target)
        total_loss = a * dice_loss + b * focal_loss

        # Check for NaN contributions and warn
        if torch.isnan(dice_loss) or torch.isnan(focal_loss) or torch.isnan(total_loss):
            print("Warning: One of the loss contributions is NaN.")

        return total_loss, dice_loss, focal_loss

    return loss_criterion

def create_loss_criterion(dice_weight=0.5, focal_weight=0.5, use_tversky=False):
    """
    Create an improved loss criterion for segmentation.
    
    Args:
        dice_weight: Weight for dice/tversky loss
        focal_weight: Weight for focal loss
        use_tversky: If True, use Tversky loss instead of Dice loss
    
    Returns:
        Loss function that returns (total_loss, component_loss_1, component_loss_2)
    """
    if use_tversky:
        primary_loss = TverskyLoss(alpha=0.3, beta=0.7)  # Emphasizes recall (reduces false negatives)
        loss_name = "tversky"
    else:
        primary_loss = DiceLoss(smooth=1.0)
        loss_name = "dice"
    
    focal_loss = ImprovedFocalLoss(alpha=0.25, gamma=2.0)
    
    def loss_criterion(pred, target):
        loss1 = primary_loss(pred, target)
        loss2 = focal_loss(pred, target)
        
        total_loss = dice_weight * loss1 + focal_weight * loss2
        
        # Debugging information for unstable training
        if torch.isnan(total_loss) or total_loss > 10.0:  # Suspiciously high loss
            print(f"Warning: Unusual loss values detected!")
            print(f"{loss_name.capitalize()} loss: {loss1.item():.6f}")
            print(f"Focal loss: {loss2.item():.6f}")
            print(f"Total loss: {total_loss.item():.6f}")
            print(f"Pred range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
            print(f"Target range: [{target.min().item():.3f}, {target.max().item():.3f}]")
            
            # Clamp extremely high losses
            total_loss = torch.clamp(total_loss, max=5.0)
        
        return total_loss, loss1, loss2
    
    return loss_criterion

class NaNSafeLoss(nn.Module):
    """
    A wrapper that makes any loss function more robust against NaN values
    """
    def __init__(self, loss_fn, fallback_value=1.0):
        super(NaNSafeLoss, self).__init__()
        self.loss_fn = loss_fn
        self.fallback_value = fallback_value
    
    def forward(self, pred, target):
        # Check for NaN in inputs before computing loss
        if torch.isnan(pred).any():
            print(f"ERROR: NaN detected in predictions!")
            return torch.tensor(self.fallback_value, device=pred.device, requires_grad=True)
        
        if torch.isinf(pred).any():
            print(f"ERROR: Inf detected in predictions!")
            pred = torch.clamp(pred, min=-10, max=10)
        
        loss = self.loss_fn(pred, target)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"ERROR: NaN/Inf in loss computation!")
            return torch.tensor(self.fallback_value, device=pred.device, requires_grad=True)
        
        return loss

class RobustTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0, eps=1e-7):
        super(RobustTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.eps = eps
    
    def forward(self, preds, targets):
        # Early NaN detection
        if torch.isnan(preds).any():
            print("NaN detected in predictions before sigmoid!")
            return torch.tensor(1.0, device=preds.device, requires_grad=True)
        
        # Clamp input logits to prevent extreme values
        preds = torch.clamp(preds, min=-10, max=10)
        
        # Apply sigmoid with numerical stability
        preds = torch.sigmoid(preds)
        
        # Additional clamping after sigmoid
        preds = torch.clamp(preds, min=self.eps, max=1 - self.eps)
        
        # Check for NaN after sigmoid
        if torch.isnan(preds).any():
            print("NaN detected after sigmoid!")
            return torch.tensor(1.0, device=preds.device, requires_grad=True)
        
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1).float()
        
        # Calculate components with numerical stability
        TP = (preds_flat * targets_flat).sum()
        FP = (preds_flat * (1 - targets_flat)).sum()
        FN = ((1 - preds_flat) * targets_flat).sum()
        
        # Check for NaN in components
        if torch.isnan(TP) or torch.isnan(FP) or torch.isnan(FN):
            print("NaN in Tversky components!")
            return torch.tensor(1.0, device=preds.device, requires_grad=True)
        
        denominator = TP + self.alpha * FP + self.beta * FN + self.smooth
        
        # Ensure denominator is not zero or NaN
        if denominator <= self.eps or torch.isnan(denominator):
            print("Invalid denominator in Tversky!")
            return torch.tensor(0.5, device=preds.device, requires_grad=True)
        
        tversky = (TP + self.smooth) / denominator
        loss = 1 - tversky
        
        # Final NaN check
        if torch.isnan(loss):
            print("NaN in final Tversky loss!")
            return torch.tensor(1.0, device=preds.device, requires_grad=True)
        
        return torch.clamp(loss, min=0.0, max=2.0)

class RobustFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-8):
        super(RobustFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, inputs, targets):
        # Early NaN detection
        if torch.isnan(inputs).any():
            print("NaN detected in focal loss inputs!")
            return torch.tensor(1.0, device=inputs.device, requires_grad=True)
        
        # Clamp input logits
        inputs = torch.clamp(inputs, min=-10, max=10)
        
        # Use BCE with logits for better numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none'
        )
        
        # Check for NaN in BCE
        if torch.isnan(bce_loss).any():
            print("NaN in BCE computation!")
            return torch.tensor(1.0, device=inputs.device, requires_grad=True)
        
        # Calculate pt more robustly
        probs = torch.sigmoid(inputs)
        probs = torch.clamp(probs, min=self.eps, max=1 - self.eps)
        
        pt = torch.where(targets == 1, probs, 1 - probs)
        pt = torch.clamp(pt, min=self.eps, max=1 - self.eps)
        
        # Calculate alpha term
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Calculate focal weight with clamping
        focal_weight = alpha_t * torch.pow(1 - pt, self.gamma)
        focal_weight = torch.clamp(focal_weight, max=100.0)  # Prevent extreme weights
        
        focal_loss = focal_weight * bce_loss
        
        # Final checks
        if torch.isnan(focal_loss).any():
            print("NaN in final focal loss!")
            return torch.tensor(1.0, device=inputs.device, requires_grad=True)
        
        return torch.mean(focal_loss)

def check_model_for_nans(model, input_tensor):
    """
    Debug function to find where NaNs are introduced in the model
    """
    print("=== NaN Debugging ===")
    print(f"Input stats: min={input_tensor.min():.6f}, max={input_tensor.max():.6f}, has_nan={torch.isnan(input_tensor).any()}")
    
    # Hook function to check each layer's output
    def nan_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                has_nan = torch.isnan(output).any()
                has_inf = torch.isinf(output).any()
                if has_nan or has_inf:
                    print(f"Layer {name}: NaN={has_nan}, Inf={has_inf}")
                    print(f"  Output stats: min={output.min():.6f}, max={output.max():.6f}")
                else:
                    print(f"Layer {name}: OK (min={output.min():.6f}, max={output.max():.6f})")
        return hook
    
    # Register hooks on all modules
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            hook = module.register_forward_hook(nan_hook(name))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    print("=== End NaN Debug ===")
    return output

def create_robust_loss_criterion(dice_weight=0.6, focal_weight=0.4):
    """
    Create a robust loss criterion that handles NaN gracefully
    """
    tversky_loss = RobustTverskyLoss(alpha=0.3, beta=0.7)
    focal_loss = RobustFocalLoss(alpha=0.25, gamma=2.0)
    
    def loss_criterion(pred, target):
        # Pre-checks
        if torch.isnan(pred).any():
            print("CRITICAL: NaN in predictions before loss computation!")
            print("This suggests the issue is in your model, not the loss function.")
            return (torch.tensor(10.0, device=pred.device, requires_grad=True), 
                   torch.tensor(5.0, device=pred.device, requires_grad=True),
                   torch.tensor(5.0, device=pred.device, requires_grad=True))
        
        # Compute individual losses
        tversky_val = tversky_loss(pred, target)
        focal_val = focal_loss(pred, target)
        
        # Combine losses
        total_loss = dice_weight * tversky_val + focal_weight * focal_val
        
        return total_loss, tversky_val, focal_val
    
    return loss_criterion

def gradient_clipping_hook(model, max_norm=1.0):
    """
    Add gradient clipping to prevent exploding gradients
    """
    def clip_grad_hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            torch.nn.utils.clip_grad_norm_([grad_output[0]], max_norm)
    
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            module.register_backward_hook(clip_grad_hook)
        
##################################################################################
class MultiClassCEIoULoss(nn.Module):
    """
    Combined CrossEntropy + Soft IoU Loss for multiclass segmentation.
    """
    def __init__(self, ce_weight=0.5, iou_weight=0.5, smooth=1.0):
        super(MultiClassCEIoULoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.iou_loss = MultiClassIoULoss(smooth=smooth)
        self.iou_weight = iou_weight
        self.ce_weight = ce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (N, C, H, W) - raw logits
            targets: Tensor of shape (N, H, W) - integer labels [0, C-1]
        Returns:
            Combined loss value.
        """
        ce = self.ce_loss(logits, targets)
        iou = self.iou_loss(logits, targets)

        return self.ce_weight * ce + self.iou_weight * iou

class MultiClassIoULoss(nn.Module):
    """
    Differentiable IoU Loss for multi-class segmentation.
    Supports soft IoU over probabilistic predictions.
    """
    def __init__(self, smooth=1.0):
        super(MultiClassIoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (N, C, H, W) - raw logits
            targets: Tensor of shape (N, H, W) - integer class labels
        Returns:
            Scalar IoU loss (1 - mean IoU over all classes).
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)               # (N, C, H, W)
        targets_onehot = F.one_hot(targets, num_classes=num_classes)  # (N, H, W, C)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()   # (N, C, H, W)

        intersection = (probs * targets_onehot).sum(dim=(2, 3))       # (N, C)
        union = (probs + targets_onehot - probs * targets_onehot).sum(dim=(2, 3))  # (N, C)
        iou = (intersection + self.smooth) / (union + self.smooth)    # (N, C)

        return 1 - iou.mean()

class IoULoss(nn.Module):
    """
    IoU (Jaccard) Loss for binary segmentation.
    """
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (N, 1, H, W) — raw model outputs
            targets: Tensor of shape (N, 1, H, W) or (N, H, W) — binary ground truth (0 or 1)
        Returns:
            IoU loss value.
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors
        probs = probs.view(-1)
        targets = targets.view(-1).float()

        # Compute intersection and union
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum() - intersection

        # IoU calculation with smoothing
        iou = (intersection + self.smooth) / (union + self.smooth)

        # Return IoU loss
        return 1 - iou

class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        alpha: balancing factor for positive vs negative samples
        gamma: focusing parameter (higher = focus more on hard examples)
        """
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        probs = torch.sigmoid(preds)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

                        
class BCEIoULoss(nn.Module):
    """
    Combined BCE + IoU Loss for stable binary segmentation training.
    """
    def __init__(self, bce_weight=0.5, iou_weight=0.5, smooth=1.0):
        super(BCEIoULoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.iou_loss = IoULoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (N, 1, H, W)
            targets: Tensor of shape (N, 1, H, W) or (N, H, W)
        Returns:
            Combined loss value.
        """
        # Ensure target has shape (N, 1, H, W)
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        bce = self.bce_loss(logits, targets.float())
        iou = self.iou_loss(logits, targets)
        return self.bce_weight * bce + self.iou_weight * iou
        

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
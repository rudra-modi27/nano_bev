import torch
import torch.nn.functional as F

# ==========================================
# 1. EVALUATION METRIC: Occupancy IoU
# ==========================================
def calculate_iou(pred_logits, target_grid, threshold=0.0):
    if target_grid.dim() == 3:
        target_grid = target_grid.unsqueeze(1)
        
    preds = (pred_logits >= threshold).float()
    targets = (target_grid >= 0.5).float()
    
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

# ==========================================
# 2. RESOLUTION-AWARE DISTANCE MASK
# ==========================================
def create_distance_weight_mask(grid_size=(400, 400), device='cuda'):
    H, W = grid_size
    center_y, center_x = H // 2, W // 2
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device), 
        torch.arange(W, device=device), 
        indexing='ij'
    )
    distances = torch.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_dist = torch.sqrt(torch.tensor((H/2)**2 + (W/2)**2, device=device))
    
    weights = 1.0 - (distances / (max_dist * 1.5))
    weights = torch.clamp(weights, min=0.5) 
    return weights.unsqueeze(0).unsqueeze(0)

# ==========================================
# 3. ADVANCED FOCAL + DICE LOSS
# ==========================================
def distance_weighted_focal_dice_loss(pred_logits, target_grid, alpha=0.95, gamma=2.0):
    """
    Combined Loss: Focal Loss + Dice Loss + Distance Weighting.
    alpha=0.95 heavily penalizes missing a car.
    gamma=2.0 aggressively ignores the "easy" empty road pixels.
    """
    if target_grid.dim() == 3:
        target_grid = target_grid.unsqueeze(1)
        
    pred_probs = torch.sigmoid(pred_logits)
    
    # --- 1. FOCAL LOSS ---
    # Standard unreduced BCE
    bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target_grid, reduction='none')
    
    # Calculate probabilities of the correct class (p_t)
    p_t = pred_probs * target_grid + (1 - pred_probs) * (1 - target_grid)
    
    # Apply alpha (class balancing) and gamma (focusing)
    alpha_factor = target_grid * alpha + (1 - target_grid) * (1 - alpha)
    modulating_factor = (1.0 - p_t) ** gamma
    
    focal_loss = alpha_factor * modulating_factor * bce_loss
    
    # Apply your Distance Mask
    H, W = pred_logits.shape[2], pred_logits.shape[3]
    weight_mask = create_distance_weight_mask(grid_size=(H, W), device=pred_logits.device)
    weighted_focal = (focal_loss * weight_mask).mean()
    
    # --- 2. SOFT DICE LOSS ---
    intersection = (pred_probs * target_grid).sum(dim=(2, 3))
    union = pred_probs.sum(dim=(2, 3)) + target_grid.sum(dim=(2, 3))
    
    dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)
    dice_loss = 1.0 - dice_score.mean()
    
    # --- 3. COMBINE ---
    # Because focal loss shrinks the background penalty so much, 
    # we double the weight of the Dice Loss to ensure bounding box shapes stay sharp.
    return weighted_focal + (dice_loss * 2.0)
    
    # Paste at the bottom of utils.py
def depth_distillation_loss(student_logits, teacher_depth):
    """
    Forces the student's 41 depth bins to mimic the Teacher's relative depth map.
    """
    # 1. Convert Student's 41 bins into a single expected normalized depth [0, 1]
    probs = F.softmax(student_logits, dim=1)
    bins = torch.linspace(0, 1, probs.shape[1], device=probs.device).view(1, -1, 1, 1)
    student_depth = (probs * bins).sum(dim=1, keepdim=True) # Shape: [B*6, 1, H, W]
    
    # 2. Resize Teacher's high-res depth to match the Student's smaller feature map
    teacher_depth = teacher_depth.unsqueeze(1) 
    teacher_depth_resized = F.interpolate(teacher_depth, size=student_depth.shape[2:], mode='bilinear', align_corners=False)
    
    # 3. Normalize Teacher's depth to [0, 1] per image (Min-Max Scaling)
    min_val = teacher_depth_resized.view(teacher_depth_resized.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
    max_val = teacher_depth_resized.view(teacher_depth_resized.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    teacher_depth_norm = (teacher_depth_resized - min_val) / (max_val - min_val + 1e-6)
    
    # 4. Calculate the difference
    return F.l1_loss(student_depth, teacher_depth_norm)
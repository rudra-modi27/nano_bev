import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import time
from nuscenes.nuscenes import NuScenes

# Import your custom modules
from dataset import OptimizedNuScenesDataset
from models import BEVOccupancyModel
from utils import calculate_iou, distance_weighted_focal_dice_loss, depth_distillation_loss

# ==========================================
# CONFIGURATION - THE LABORATORY
# ==========================================
DATAROOT = '/media/rudra-modi/shared data/Coding/Projects/MAHE/data/nuscenes/'
PREVIOUS_WEIGHTS = 'weights.pth' 
BATCH_SIZE = 1 # CRITICAL: Keeps your 4GB VRAM safe
EPOCHS = 5
LEARNING_RATE = 2e-4 # Slightly higher to jumpstart the new Depth Head

def train_model():
    print("Initializing Phase 4: MiDaS Knowledge Distillation...")
    
    # 1. SETUP DATASET
    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
    all_scenes = [s['token'] for s in nusc.scene]
    train_scenes = all_scenes[:8] 
    
    train_dataset = OptimizedNuScenesDataset(nusc, train_scenes)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    # 2. SETUP MODEL
    model = BEVOccupancyModel(unfreeze_backbone=False).cuda()
    
    # ==========================================
    # 3. THE "BRAIN TRANSPLANT" (Smart Loader)
    # ==========================================
    print(f"Attempting Hybrid Load from '{PREVIOUS_WEIGHTS}'...")
    try:
        checkpoint = torch.load(PREVIOUS_WEIGHTS)
        model_dict = model.state_dict()
        
        # Filter out the old projector and head, KEEP ONLY the ConvNeXt backbone
        pretrained_dict = {k: v for k, v in checkpoint.items() 
                           if k in model_dict and 'backbone' in k}
                           
        # Update the new model's dictionary with the old vision weights
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
        print(f"SUCCESS: Transplanted {len(pretrained_dict)} vision layers.")
        print("The new Depth Projector is blank and ready to learn physics.")
    except Exception as e:
        print(f"⚠️ Warning: Could not perform hybrid load: {e}")
        print("Starting entirely from scratch...")

    # ==========================================
    # 4. SUMMON THE TEACHER (MiDaS)
    # ==========================================
    print("Summoning MiDaS Teacher from the cloud...")
    # Using MiDaS_small because it's tiny and fits in 4GB VRAM
    teacher = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).cuda()
    teacher.eval() # CRITICAL: Freezes the teacher
    for param in teacher.parameters():
        param.requires_grad = False
    print("Teacher ready.")

    # 5. OPTIMIZER & SCALER
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler('cuda')

    # 6. TRAINING LOOP
    print(f"\n🔥 Starting Distillation Training... (Total Epochs: {EPOCHS})\n")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_iou = 0.0
        
        print(f"=== Epoch {epoch+1}/{EPOCHS} ===")
        
        for step, (images, intrinsics, extrinsics, occupancy_gt) in enumerate(train_loader):
            start_time = time.time()
            
            # Move to 4GB VRAM
            images = images.cuda()
            occupancy_gt = occupancy_gt.cuda()
            
            # The 6 cameras are packed. Unpack them for the Teacher.
            B, num_cams, C, H, W = images.shape
            flat_images = images.view(B * num_cams, C, H, W)
            
            optimizer.zero_grad(set_to_none=True)
            
            # --- THE TEACHER'S FORWARD PASS (VRAM Shield active) ---
            with torch.no_grad():
                with autocast('cuda', dtype=torch.float16):
                    # MiDaS looks at the images and creates perfect depth maps
                    midas_depth = teacher(flat_images) 
            
            # --- THE STUDENT'S FORWARD PASS ---
            with autocast('cuda', dtype=torch.float16):
                # Unpack the returned tuple from the tweaked models.py
                occupancy_pred, student_depth_logits = model(images)
                
                # Calculate the standard BEV Map loss
                map_loss = distance_weighted_focal_dice_loss(occupancy_pred, occupancy_gt)
                
                # Calculate the new Distillation Loss
                distill_loss = depth_distillation_loss(student_depth_logits, midas_depth)
                
                # Combine them (Weight the distillation heavily so the student prioritizes it)
                loss = map_loss + (distill_loss * 5.0) 
            
            # Mixed Precision Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Metrics (Only grade the BEV map for the printout IoU)
            iou = calculate_iou(occupancy_pred, occupancy_gt, threshold=0.0)
            total_loss += loss.item()
            total_iou += iou.item()
            
            step_time = time.time() - start_time
            
            if step % 20 == 0:
                print(f"  Step {step:03d} | Total Loss: {loss.item():.4f} (Distill: {distill_loss.item():.4f}) | IoU: {iou.item():.4f} | {step_time:.2f}s")
                
        # Epoch Summary
        avg_loss = total_loss / len(train_loader)
        avg_iou = total_iou / len(train_loader)
        print(f"\nEpoch {epoch+1} Done -> Avg Loss: {avg_loss:.4f} | Avg IoU: {avg_iou:.4f}\n")
        
        # Save checkpoint after every epoch
        torch.save(model.state_dict(), 'LSS_BEV_MODEL_400.pth')

    print("Distillation Training Complete! Weights saved as 'LSS_BEV_MODEL_400.pth'.")

if __name__ == "__main__":
    train_model()
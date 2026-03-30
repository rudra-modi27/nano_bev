import torch
import time
from torch.utils.data import DataLoader
from torch.amp import autocast
from nuscenes.nuscenes import NuScenes

# Import your custom modules
from dataset import OptimizedNuScenesDataset
from models import BEVOccupancyModel

# ==========================================
# CONFIGURATION - THE COMPREHENSIVE EXAM
# ==========================================
DATAROOT = '/path/to/nuscenes/' 
WEIGHTS = 'LSS_BEV_MODEL_400.pth' # FIXED: Must use LSS weights for this architecture
BATCH_SIZE = 1 
CONFIDENCE_THRESHOLD = 0.15 

# ==========================================
# METRICS ENGINE
# ==========================================
def calculate_comprehensive_metrics(pred_probs, target_grid, threshold=0.5):
    if target_grid.dim() == 3:
        target_grid = target_grid.unsqueeze(1)
        
    preds = (pred_probs >= threshold).float()
    targets = (target_grid >= 0.5).float()
    
    TP = (preds * targets).sum(dim=(2, 3)) 
    FP = (preds * (1 - targets)).sum(dim=(2, 3)) 
    FN = ((1 - preds) * targets).sum(dim=(2, 3)) 
    TN = ((1 - preds) * (1 - targets)).sum(dim=(2, 3)) 
    
    iou = (TP + 1e-6) / (TP + FP + FN + 1e-6)
    precision = (TP + 1e-6) / (TP + FP + 1e-6)
    recall = (TP + 1e-6) / (TP + FN + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    accuracy = (TP + TN + 1e-6) / (TP + FP + FN + TN + 1e-6)
    
    return {
        'iou': iou.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1_score': f1_score.mean().item(),
        'accuracy': accuracy.mean().item()
    }

def evaluate_model():
    print("Initializing Comprehensive Validation Protocol...")
    
    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
    all_scenes = [s['token'] for s in nusc.scene]
    val_scenes = all_scenes[8:] 
    
    val_dataset = OptimizedNuScenesDataset(nusc, val_scenes)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print("Loading ULTIMATE 400-Res Weights...")
    model = BEVOccupancyModel(unfreeze_backbone=True).cuda()
    
    try:
        model.load_state_dict(torch.load(WEIGHTS))
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval() 

    print(f"\nStarting Inference on {len(val_dataset)} frames...\n")
    
    total_time = 0.0
    cumulative_metrics = {'iou': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0}
    
    with torch.no_grad():
        for step, (images, intrinsics, extrinsics, occupancy_gt) in enumerate(val_loader):
            start_time = time.time()
            images, occupancy_gt = images.cuda(), occupancy_gt.cuda()
            
            with autocast('cuda', dtype=torch.float16):
                pred_logits, _ = model(images)
                pred_probs = torch.sigmoid(pred_logits)
                
            metrics = calculate_comprehensive_metrics(pred_probs, occupancy_gt, threshold=CONFIDENCE_THRESHOLD)
            
            step_time = time.time() - start_time
            total_time += step_time
            
            for key in cumulative_metrics:
                cumulative_metrics[key] += metrics[key]
            
            if step % 10 == 0:
                print(f"  Processed Frame {step:03d}/{len(val_dataset)} | IoU: {metrics['iou']:.4f} | F1: {metrics['f1_score']:.4f} | {step_time:.3f}s")

    avg_metrics = {k: v / len(val_loader) for k, v in cumulative_metrics.items()}
    avg_fps = len(val_loader) / total_time
    
    print("\n" + "="*50)
    print("FINAL VALIDATION DASHBOARD")
    print("="*50)
    print(f"Model Resolution  : 400x400")
    print(f"Frames Tested     : {len(val_dataset)}")
    print(f"Speed             : {avg_fps:.2f} FPS")
    print("-"  * 50)
    print(f"BEV IoU           : {avg_metrics['iou']:.4f}  (Primary Map Quality)")
    print(f"F1 Score          : {avg_metrics['f1_score']:.4f}  (Balanced Harmony)")
    print(f"Precision         : {avg_metrics['precision']:.4f}  (When it guesses 'Car', is it right?)")
    print(f"Recall            : {avg_metrics['recall']:.4f}  (Of all real cars, how many did it find?)")
    print(f"Accuracy          : {avg_metrics['accuracy']:.4f}  (Warning: Inflated by empty road pixels)")
    print("="*50)

if __name__ == "__main__":
    evaluate_model()

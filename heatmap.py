import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from nuscenes.nuscenes import NuScenes

# Import your custom modules
from dataset import OptimizedNuScenesDataset
from models import BEVOccupancyModel

# --- CONFIG ---
DATAROOT = '/media/rudra-modi/shared data/Coding/Projects/MAHE/data/nuscenes/'
WEIGHTS = 'LSS_BEV_MODEL_400.pth' # FIXED: Pointing to your new LSS weights

def generate_heatmap_video():
    print("Preparing 'Honest' LSS Heatmap Video...")
    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
    
    scene_token = nusc.scene[0]['token']
    dataset = OptimizedNuScenesDataset(nusc, [scene_token])
    
    model = BEVOccupancyModel(unfreeze_backbone=False).cuda()
    try:
        model.load_state_dict(torch.load(WEIGHTS))
        print("LSS Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('honest_lss_drive.mp4', fourcc, 10.0, (1600, 800))

    print(f"Processing {len(dataset)} frames...")
    
    with torch.no_grad():
        for i in range(len(dataset)):
            images, _, _, gt = dataset[i] 
            images = images.unsqueeze(0).cuda()
            
            # FIXED: Unpack the tuple here!
            logits, _ = model(images)
            pred = torch.sigmoid(logits)[0, 0].cpu().numpy()
            gt_img = gt[0].cpu().numpy()

            # --- RENDER: AI HEATMAP (The pure truth) ---
            pred_color = (cm.magma(pred)[:, :, :3] * 255).astype(np.uint8)
            pred_color = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
            
            for x in range(0, 400, 20):
                cv2.line(pred_color, (x, 0), (x, 400), (40, 40, 40), 1)
                cv2.line(pred_color, (0, x), (400, x), (40, 40, 40), 1)

            # --- RENDER: AI BOXES (Strict Threshold) ---
            binary_mask = (pred > 0.70).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 2 and h > 2: 
                    cv2.rectangle(pred_color, (x, y), (x + w, y + h), (255, 255, 255), 1)

            # --- RENDER: GROUND TRUTH ---
            gt_color = np.zeros((400, 400, 3), dtype=np.uint8) + 20
            gt_binary = (gt_img > 0.5).astype(np.uint8) * 255
            gt_contours, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in gt_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(gt_color, (x, y), (x + w, y + h), (0, 255, 0), -1)

            # --- EGO CAR ---
            for img in [pred_color, gt_color]:
                cv2.rectangle(img, (196, 192), (204, 208), (255, 0, 0), -1)

            # --- COMBINE ---
            combined = np.hstack((gt_color, pred_color))
            upscaled = cv2.resize(combined, (1600, 800), interpolation=cv2.INTER_NEAREST)
            
            cv2.putText(upscaled, "GROUND TRUTH", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(upscaled, "LSS HEATMAP + STRICT BOXES", (850, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            out.write(upscaled)
            
    out.release()
    print("Honest Video saved as 'honest_lss_drive.mp4'!")

if __name__ == "__main__":
    generate_heatmap_video()

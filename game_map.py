import torch
import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from dataset import OptimizedNuScenesDataset
from models import BEVOccupancyModel

# --- CONFIG ---
DATAROOT = '/media/rudra-modi/shared data/Coding/Projects/MAHE/data/nuscenes/'
WEIGHTS = 'LSS_BEV_MODEL_400.pth' 

def generate_game_map_video():
    print("Preparing Pure High-Res Grid-Box Video...")
    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
    
    scene_token = nusc.scene[0]['token']
    dataset = OptimizedNuScenesDataset(nusc, [scene_token])
    
    model = BEVOccupancyModel(unfreeze_backbone=False).cuda()
    model.load_state_dict(torch.load(WEIGHTS))
    model.eval()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('pure_grid_drive.mp4', fourcc, 10.0, (1600, 800))

    print(f"Processing {len(dataset)} frames...")
    
    with torch.no_grad():
        for i in range(len(dataset)):
            images, _, _, gt = dataset[i]
            images = images.unsqueeze(0).cuda()
            
            # FIXED: Unpack the tuple here!
            logits, _ = model(images)
            pred = torch.sigmoid(logits)[0, 0].cpu().numpy()
            gt_img = gt[0].cpu().numpy()

            # --- RENDER LOGIC: AI PREDICTION ---
            binary_mask = (pred > 0.45).astype(np.uint8) * 255
            
            kernel = np.ones((5,5), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            pred_color = np.zeros((400, 400, 3), dtype=np.uint8) + 30 
            for x in range(0, 400, 20):
                cv2.line(pred_color, (x, 0), (x, 400), (50, 50, 50), 1)
                cv2.line(pred_color, (0, x), (400, x), (50, 50, 50), 1)

            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 3 and h > 3: 
                    cv2.rectangle(pred_color, (x, y), (x + w, y + h), (0, 165, 255), -1)
                    cv2.rectangle(pred_color, (x, y), (x + w, y + h), (255, 255, 255), 1)

            # --- RENDER LOGIC: GROUND TRUTH ---
            gt_color = np.zeros((400, 400, 3), dtype=np.uint8) + 20
            gt_binary = (gt_img > 0.5).astype(np.uint8) * 255
            gt_contours, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in gt_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(gt_color, (x, y), (x + w, y + h), (0, 255, 0), -1)

            # --- DRAW EGO CAR ---
            for img in [pred_color, gt_color]:
                cv2.rectangle(img, (196, 192), (204, 208), (255, 0, 0), -1)

            # --- COMBINE AND SCALE ---
            combined = np.hstack((gt_color, pred_color))
            upscaled = cv2.resize(combined, (1600, 800), interpolation=cv2.INTER_NEAREST)
            
            cv2.putText(upscaled, "GROUND TRUTH (400 RES)", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(upscaled, "AI GRID MAP (RAW 400)", (850, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            out.write(upscaled)
            
    out.release()
    print("Video saved as 'pure_grid_drive.mp4'!")

if __name__ == "__main__":
    generate_game_map_video()
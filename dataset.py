import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

class OptimizedNuScenesDataset(Dataset):
    def __init__(self, nusc, scene_tokens, target_size=(256, 704)):
        self.nusc = nusc
        self.target_size = target_size
        
        # Order matters!
        self.cams = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        
        self.samples = []
        for scene_token in scene_tokens:
            scene = self.nusc.get('scene', scene_token)
            current_sample_token = scene['first_sample_token']
            
            while current_sample_token != '':
                self.samples.append(current_sample_token)
                sample = self.nusc.get('sample', current_sample_token)
                current_sample_token = sample['next']
                
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def _generate_occupancy_grid(self, sample, grid_size=400, resolution=0.25):
        grid_img = Image.new('L', (grid_size, grid_size), 0)
        draw = ImageDraw.Draw(grid_img)
        
        cam_front_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        ego_pose = self.nusc.get('ego_pose', cam_front_data['ego_pose_token'])
        
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            if 'vehicle' not in ann['category_name'] and 'human' not in ann['category_name']:
                continue
                
            box = self.nusc.get_box(ann['token'])
            box.translate(-np.array(ego_pose['translation']))
            box.rotate(Quaternion(ego_pose['rotation']).inverse)
            
            corners = box.bottom_corners() 
            x_coords, y_coords = corners[0, :], corners[1, :] 
            
            pixel_y = (-x_coords / resolution) + (grid_size / 2) 
            pixel_x = (-y_coords / resolution) + (grid_size / 2) 
            
            polygon_pts = [(int(x), int(y)) for x, y in zip(pixel_x, pixel_y)]
            draw.polygon(polygon_pts, fill=255)
            
        occupancy_numpy = np.array(grid_img, dtype=np.float32) / 255.0
        return torch.from_numpy(occupancy_numpy).unsqueeze(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_token = self.samples[idx]
        sample = self.nusc.get('sample', sample_token)
        
        images = []
        intrinsics = []
        extrinsics = []
        
        for cam in self.cams:
            cam_data = self.nusc.get('sample_data', sample['data'][cam])
            
            # 1. IMAGE EXTRACTION
            img_path = self.nusc.get_sample_data_path(cam_data['token'])
            img = Image.open(img_path).convert('RGB')
            width, height = img.size
            img = img.crop((0, 350, width, height)) 
            img = img.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
            
            img_tensor = self.normalize(self.color_jitter(transforms.ToTensor()(img)))
            images.append(img_tensor)

            # 2. CALIBRATION EXTRACTION (The new LSS requirements)
            cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            
            # Intrinsics: 3x3 matrix representing focal length and optical center
            intrinsic = np.array(cs_record['camera_intrinsic'], dtype=np.float32)
            
            # Extrinsics: 4x4 matrix representing rotation and translation from the car's center
            translation = np.array(cs_record['translation'], dtype=np.float32)
            rotation = Quaternion(cs_record['rotation']).rotation_matrix.astype(np.float32)
            
            extrinsic = np.eye(4, dtype=np.float32)
            extrinsic[:3, :3] = rotation
            extrinsic[:3, 3] = translation

            intrinsics.append(torch.from_numpy(intrinsic))
            extrinsics.append(torch.from_numpy(extrinsic))

        images_tensor = torch.stack(images)
        intrinsics_tensor = torch.stack(intrinsics)
        extrinsics_tensor = torch.stack(extrinsics)
        real_occupancy_gt = self._generate_occupancy_grid(sample)

        # Notice we are now returning 4 items!
        return images_tensor, intrinsics_tensor, extrinsics_tensor, real_occupancy_gt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from scipy.ndimage import map_coordinates, gaussian_filter

# 탄성 변형 함수
def elastic_transform(image, label, alpha=100, sigma=10, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    # 데이터 타입 확인 및 변환
    image = image.astype(np.float32)
    label = label.astype(np.float32)
    
    shape = image.shape
    
    # 변위 필드 생성
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
    
    # 좌표 매핑
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # 이미지: 선형 보간 (order=1)
    transformed_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    
    # 라벨: 최근접 이웃 (order=0) - 이진 마스크에 적합
    transformed_label = map_coordinates(label, indices, order=0, mode='reflect').reshape(shape)
    
    return transformed_image, transformed_label

# Dataset 클래스 정의
class ISBI_Loader(Dataset):
    def __init__(self, image_dir, label_dir, augment=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.augment = augment
        
        # npy 파일 목록 가져오기
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 이미지와 레이블 파일 경로 가져오기
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx])

        # 이미지와 레이블 데이터 로드
        image = np.load(image_path)
        label = np.load(label_path)

        # 데이터 타입 및 범위 확인 후 정규화
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
            
        if label.max() > 1.0:
            label = label.astype(np.float32) / 255.0
        else:
            label = label.astype(np.float32)

        # 증강 적용
        if self.augment:
            # 1. 탄성 변형
            if np.random.rand() < 0.5:
                image, label = elastic_transform(image, label)

            # 2. 회전
            if np.random.rand() < 0.5:
                angle = np.random.randint(-30, 30)
                # float32 (0-1) → uint8 (0-255) 변환
                image_uint8 = (image * 255).astype(np.uint8)
                label_uint8 = (label * 255).astype(np.uint8)
                # PIL Image로 변환 후 회전
                image_rotated = Image.fromarray(image_uint8).rotate(angle, resample=Image.BILINEAR)
                label_rotated = Image.fromarray(label_uint8).rotate(angle, resample=Image.NEAREST)
                # 다시 float32 (0-1)로 변환
                image = np.array(image_rotated).astype(np.float32) / 255.0
                label = np.array(label_rotated).astype(np.float32) / 255.0

            # 3. 좌우 반전
            if np.random.rand() < 0.5:
                image = np.fliplr(image)
                label = np.fliplr(label)

            # 4. 상하 반전
            if np.random.rand() < 0.5:
                image = np.flipud(image)
                label = np.flipud(label)

            # 5. 밝기/대비 조절 
            if np.random.rand() < 0.5:
                # 밝기 조절
                brightness_factor = np.random.uniform(0.8, 1.2)
                image = image * brightness_factor
                image = np.clip(image, 0, 1) # 0-1 범위로 클리핑

                # 대비 조절
                contrast_factor = np.random.uniform(0.8, 1.2)
                mean = image.mean()
                image = (image - mean) * contrast_factor + mean
                image = np.clip(image, 0, 1) # 0-1 범위로 클리핑
        
        # 채널 차원 추가
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        # Tensor로 변환
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        # 이미지와 레이블 반환
        return image, label

# data loader 반환 함수
def get_dataloader(data_dir, batch_size=4, shuffle=True, num_workers=4):

    train_loader = DataLoader(
        ISBI_Loader(os.path.join(data_dir, 'train/images'), os.path.join(data_dir, 'train/labels'), augment=True),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    
    val_loader = DataLoader(
        ISBI_Loader(os.path.join(data_dir, 'val/images'), os.path.join(data_dir, 'val/labels'), augment=False),
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    test_loader = DataLoader(
        ISBI_Loader(os.path.join(data_dir, 'test/images'), os.path.join(data_dir, 'test/labels'), augment=False),
        batch_size=1, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
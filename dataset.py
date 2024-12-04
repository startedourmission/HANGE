import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import logging

class UnpairedDataset(Dataset):
    def __init__(self, content_dir, style_dir, image_size=256, mode='train'):
        self.content_dir = Path(content_dir)
        self.style_dir = Path(style_dir)
        
        # 이미지 파일 확장자 정의
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # 이미지 경로 리스트 생성
        self.content_paths = self._get_valid_images(self.content_dir)
        self.style_paths = self._get_valid_images(self.style_dir)
        
        if len(self.content_paths) == 0 or len(self.style_paths) == 0:
            raise RuntimeError('No valid images found in one or both directories')
            
        # 기본 전처리 설정
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),  # 데이터 증강
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:  # test mode
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        logging.info(f'Found {len(self.content_paths)} content images and {len(self.style_paths)} style images')

    def _get_valid_images(self, dir_path):
        """유효한 이미지 파일만 필터링하여 반환"""
        image_paths = []
        for ext in self.valid_extensions:
            image_paths.extend(list(dir_path.glob(f'*{ext}')))
        return sorted(image_paths)

    def _load_image(self, path):
        """이미지 로드 및 에러 처리"""
        try:
            img = Image.open(path).convert('RGB')
            return img
        except (IOError, OSError) as e:
            logging.error(f'Error loading image {path}: {e}')
            return None

    def __getitem__(self, index):
        """인덱스에 해당하는 content, style 이미지 쌍 반환"""
        content_path = self.content_paths[index % len(self.content_paths)]
        # 스타일 이미지는 랜덤하게 선택 (unpaired dataset)
        style_index = torch.randint(0, len(self.style_paths), (1,)).item()
        style_path = self.style_paths[style_index]

        content_img = self._load_image(content_path)
        style_img = self._load_image(style_path)

        # 이미지 로드 실패 시 다른 이미지로 대체
        if content_img is None:
            logging.warning(f'Failed to load content image {content_path}, using alternative')
            content_img = self._load_image(self.content_paths[0])  # 첫 번째 이미지로 대체
        if style_img is None:
            logging.warning(f'Failed to load style image {style_path}, using alternative')
            style_img = self._load_image(self.style_paths[0])  # 첫 번째 이미지로 대체

        # 전처리 적용
        content_img = self.transform(content_img)
        style_img = self.transform(style_img)

        return content_img, style_img

    def __len__(self):
        return len(self.content_paths)

# 데이터셋 사용 예시
if __name__ == '__main__':
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 코드
    dataset = UnpairedDataset(
        content_dir='path/to/content/images',
        style_dir='path/to/style/images',
        image_size=256,
        mode='train'
    )
    
    # 첫 번째 배치 가져오기
    content_img, style_img = dataset[0]
    print(f'Content image shape: {content_img.shape}')
    print(f'Style image shape: {style_img.shape}')
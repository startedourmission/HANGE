수묵화 스타일 변환을 위한 딥러닝 모델

## 프로젝트 개요
본 프로젝트는 CycleGAN을 기반으로 하되, self-attention 메커니즘을 추가하고 단방향 변환에 최적화된 이미지 스타일 변환 모델을 구현했습니다. 일반 이미지를 수묵화 스타일로 변환하는 것을 목표로 합니다.

## 주요 특징
- Self-attention 메커니즘 도입으로 전역적 특징 포착
- 단방향 변환 구조로 모델 경량화
- Early stopping 및 best checkpoint 저장 기능
- 유연한 attention 위치 설정 (early/middle/late) - 개발중 

## 모델 구조
- Generator: Conv-DownSample-ResBlock-Attention-UpSample 구조
- Discriminator: 70x70 PatchGAN 구조
- Attention: Self-attention 메커니즘 (Query/Key/Value projection)

## 설치 및 실행

### 요구사항
```
torch>=1.7.0
torchvision>=0.8.0
Pillow
numpy
```

### 학습 실행
```bash
python train.py \
    --content_dir [content_image_path] \
    --style_dir [style_image_path] \
    --batch_size 2 \
    --num_epochs 800 \
    --attention_position middle
```

### 추론 실행
```bash
python inference.py \
    --checkpoint [checkpoint_path] \
    --input_dir [input_image_path] \
    --output_dir [output_image_path] \
    --attention_position middle
```

## 프로젝트 구조
```
├── model.py          # 모델 구현
├── dataset.py        # 데이터셋 처리
├── train.py          # 학습 코드
├── inference.py      # 추론 코드
├── checkpoints/      # 체크포인트 저장 폴더
└── README.md
```

## 주의사항
- 학습 데이터는 256x256 크기로 자동 조정됩니다
- 충분한 GPU 메모리가 필요합니다
- 최적의 결과를 위해 800 에폭 이상의 학습을 권장합니다

## Reference
- CycleGAN

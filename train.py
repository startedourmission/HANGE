import torch
from model import create_model, GANLoss, train_step
from torch.utils.data import DataLoader
from dataset import UnpairedDataset
from torchvision import transforms
from pathlib import Path
import argparse
import glob
import os

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, save_path='checkpoints/best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss, model_dict):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model_dict)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model_dict)
            self.counter = 0
            
    def save_checkpoint(self, model_dict):
        torch.save(model_dict, self.save_path)
    
def get_latest_checkpoint(save_dir):
    """가장 최근의 체크포인트 파일을 찾습니다."""
    checkpoint_files = glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth'))
    if not checkpoint_files:
        return None
    
    # 파일명에서 에폭 번호를 추출하여 가장 큰 번호의 파일을 찾습니다
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return latest_checkpoint

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention_position', type=str, default='middle', choices=['early', 'middle', 'late'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--content_dir', type=str, required=True, help='content images directory')
    parser.add_argument('--style_dir', type=str, required=True, help='style images directory')
    parser.add_argument('--image_size', type=int, default=256, help='image size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--resume', default=False, action='store_true', help='resume training from latest checkpoint')
    return parser.parse_args()

def main():
    args = get_args()
    
    # 체크포인트 디렉토리 생성
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Early stopping 초기화
    early_stopping = EarlyStopping(
        patience=15,
        save_path=save_dir / 'best_model.pth'
    )
    
    # 이동 평균을 위한 변수
    running_G_loss = 0.0
    beta = 0.9
    start_epoch = 0  


    
    # 데이터셋 & 데이터로더 설정
    dataset = UnpairedDataset(
        content_dir=args.content_dir,
        style_dir=args.style_dir,
        image_size=args.image_size,
        mode='train'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # 모델 생성
    models = create_model(attention_position=args.attention_position)
    netG, netD = models['G'], models['D']
    
    # GPU 사용 가능시 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netG = netG.to(device)
    netD = netD.to(device)
    
    # Loss 설정
    criterionGAN = GANLoss().to(device)
    
    # Optimizer 설정
    optimG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    
    latest_checkpoint = False
    if args.resume:
        latest_checkpoint = get_latest_checkpoint(args.save_dir)
        
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)

        # 모델 및 옵티마이저 상태 복원
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimG.load_state_dict(checkpoint['optimG_state_dict'])
        optimD.load_state_dict(checkpoint['optimD_state_dict'])

        # 기타 학습 상태 복원
        start_epoch = checkpoint['epoch'] + 1
        running_G_loss = checkpoint['G_loss']

        print(f"Resumed from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting from scratch.")
            
    # 학습 루프
    for epoch in range(start_epoch, args.num_epochs):
        netG.train()
        netD.train()
        
        epoch_G_losses = []
        
        for i, (real_A, real_B) in enumerate(dataloader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            losses = train_step(real_A, real_B, netG, netD, criterionGAN, optimG, optimD)
            epoch_G_losses.append(losses['loss_G'])
            
            # 로깅
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{args.num_epochs}], Step [{i}], '
                      f'D_loss: {losses["loss_D"]:.4f}, G_loss: {losses["loss_G"]:.4f}')
        
        # 에폭의 평균 Generator loss 계산
        avg_G_loss = sum(epoch_G_losses) / len(epoch_G_losses)
        
        # 이동 평균 업데이트
        running_G_loss = beta * running_G_loss + (1 - beta) * avg_G_loss
        
        # 현재 상태 저장
        checkpoint = {
            'epoch': epoch,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optimG_state_dict': optimG.state_dict(),
            'optimD_state_dict': optimD.state_dict(),
            'G_loss': running_G_loss
        }
        
        # Early stopping 체크
        early_stopping(running_G_loss, checkpoint)
        
        # 주기적 체크포인트 저장
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        # Early Stopping
#         if early_stopping.early_stop:
#             print("Early stopping triggered")
#             break
            
        print(f'Epoch [{epoch}/{args.num_epochs}] '
              f'Avg G_loss: {avg_G_loss:.4f}, '
              f'Running G_loss: {running_G_loss:.4f}')
        
if __name__ == '__main__':
    main()
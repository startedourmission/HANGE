import torch
import argparse
from torchvision import transforms
from PIL import Image
from pathlib import Path
from model import Generator
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='path to generator checkpoint')
    parser.add_argument('--input_dir', type=str, required=True, help='directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='directory to save output images')
    parser.add_argument('--attention_position', type=str, default='middle', choices=['early', 'middle', 'late'])
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--gpu', action='store_true', help='use GPU if available')
    return parser.parse_args()

def load_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    return img.unsqueeze(0)  # Add batch dimension

def save_image(tensor, output_path):
    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    # Convert to PIL Image and save
    transforms.ToPILImage()(tensor).save(output_path)

def main():
    args = get_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = Generator(attention_position=args.attention_position)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['netG_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Process images
    input_dir = Path(args.input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    with torch.no_grad():  # No need to track gradients for inference
        for img_path in input_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                print(f'Processing {img_path.name}...')
                
                # Load and preprocess image
                try:
                    img = load_image(img_path, args.image_size)
                    img = img.to(device)
                    
                    # Generate output
                    output = model(img)
                    
                    # Save output
                    output_path = output_dir / f'styled_{img_path.name}'
                    save_image(output[0], output_path)  # [0] to remove batch dimension
                    
                except Exception as e:
                    print(f'Error processing {img_path.name}: {e}')
                    continue

if __name__ == '__main__':
    main()
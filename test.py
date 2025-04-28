import torch
import csv
from tqdm import tqdm
import yaml
from pathlib import Path
from types import SimpleNamespace

from data.dataset import get_dataset
from data.transforms import get_transform
from models.infuse_model import INFUSEModel
from utils.episode import FewShotEpisodeSampler
from trainer import evaluate

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataset
    test_transform = get_transform(backbone=args.image_backbone, dataset=args.dataset, split='test')
    test_dataset = get_dataset(args.dataset, split='test', transform=test_transform, root=args.data_root)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_sampler=FewShotEpisodeSampler(test_dataset, args.num_test_episodes, args.n_way, args.k_shot, args.q_query),
        num_workers=4,
        collate_fn=lambda batch: [
            torch.stack([item[0] for item in batch]),
            torch.tensor([item[1] for item in batch])
        ]
    )

    # Load model
    model = INFUSEModel(args)
    checkpoint = torch.load(args.test_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    # Test
    test_loss, test_acc, test_ci95 = evaluate(args, model, test_loader, criterion, device)
    
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f} ± {test_ci95:.2f}%")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/test_config.yaml", help="YAML config file path")
    args, remaining_args = parser.parse_known_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    args = SimpleNamespace(**config)

    with open(Path(__file__).parent / "configs/image_encoder.yaml") as f:
        args.image_encoder_config = yaml.safe_load(f)
    
    with open(Path(__file__).parent / "configs/text_encoder.yaml") as f:
        args.text_encoder_config = yaml.safe_load(f)

    with open(Path(__file__).parent / "configs/evg.yaml") as f:
        args.evg_config = yaml.safe_load(f)

    main(args)
# main.py

import torch
from torch import nn, optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import wandb

from data.dataset import get_dataset
from data.transforms import get_transform
from models.infuse_model import INFUSEModel
from trainer import train_one_epoch, evaluate
from utils.episode import FewShotEpisodeSamplerTrain, FewShotEpisodeSampler
from utils.logger import log_metrics

def main(args):
    # Initialize wandb
    wandb.init(project="infuse", config=args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    backbone = args.image_backbone
    dataset_name = args.dataset
    
    train_transform = get_transform(backbone=backbone, dataset=dataset_name, split='train')
    test_transform = get_transform(backbone=backbone, dataset=dataset_name, split='test')
    args.train_transform = train_transform
    args.test_transform = test_transform
    
    wandb.config.update({
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "q_query": args.q_query,
        "num_episodes": args.num_episodes,
        "num_val_episodes": args.num_val_episodes,
        "epochs": args.epochs,
        "lr": args.lr,
        "feature_dim": args.feature_dim,
        "metric": args.metric,
        "classifier_temperature": args.classifier_temperature,
        "image_backbone": args.image_backbone,
        "text_backbone": args.text_backbone
    })
        
    train_dataset = get_dataset(args.dataset, split='train', transform=args.train_transform, root=args.data_root)
    val_dataset = get_dataset(args.dataset, split='val', transform=args.test_transform, root=args.data_root)

    # Batch-based training
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_sampler=FewShotEpisodeSamplerTrain(train_dataset, args.batch_size),
    #     num_workers=4,
    #     collate_fn=lambda batch: [
    #         torch.stack([item[0] for item in batch]),
    #         torch.tensor([item[1] for item in batch])
    #     ]
    # )
    
    # Episode-based training
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=FewShotEpisodeSampler(train_dataset, args.num_episodes, args.n_way, args.k_shot, args.q_query),
        num_workers=4,
        collate_fn=lambda batch: [
            torch.stack([item[0] for item in batch]),
            torch.tensor([item[1] for item in batch])
        ]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=FewShotEpisodeSampler(val_dataset, args.num_val_episodes, args.n_way, args.k_shot, args.q_query),
        num_workers=4,
        collate_fn=lambda batch: [
            torch.stack([item[0] for item in batch]),
            torch.tensor([item[1] for item in batch])
        ]
    )

    args.evg_config["input_dim"] = args.text_encoder_config["projection_dim"]
    args.text_encoder_config["encoder_name"] = args.text_backbone
    
    model = INFUSEModel(args).to(device)
    
    num_total = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_total}")
    print(f"Trainable parameters: {num_trainable}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    max_acc = 0.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}]")
        train_loss, train_acc, train_ci95 = train_one_epoch(args, model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, ci95 = evaluate(args, model, val_loader, criterion, device)

        if val_acc > max_acc:
            max_acc = val_acc
            max_ci95 = ci95
            best_epoch = epoch           
            torch.save(model.state_dict(), f"checkpoints/infuse_best.pth")

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}±{train_ci95:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}±{ci95:.2f}% | Max Acc: {max_acc+max_ci95:.2f}% (Epoch {best_epoch})")

        log_metrics(epoch=epoch,
                    train_loss=train_loss, train_acc=train_acc,
                    val_loss=val_loss, val_acc=val_acc)
        
        if epoch == args.epochs:
            torch.save(model.state_dict(), f"checkpoints/infuse_last.pth")

if __name__ == "__main__":
    import argparse
    import yaml
    from pathlib import Path
    from types import SimpleNamespace

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/run_config.yaml", help="YAML config file path")
    
    args, remaining_args = parser.parse_known_args()
    
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        args = SimpleNamespace(**config)
    else:
        parser.add_argument("--dataset", type=str, default="fc100")
        parser.add_argument("--data_root", type=str, default="/root/INFUSE/database")
        parser.add_argument("--train_transform", default=None)  # transform 정의 (yaml 등등)
        parser.add_argument("--test_transform", default=None)

        parser.add_argument("--n_way", type=int, default=5)
        parser.add_argument("--k_shot", type=int, default=1)
        parser.add_argument("--q_query", type=int, default=15)
        parser.add_argument("--num_episodes", type=int, default=100)
        parser.add_argument("--num_val_episodes", type=int, default=100)

        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--feature_dim", type=int, default=640)  # Should match EVG output_dim
        parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
        parser.add_argument("--classifier_temperature", type=float, default=1.0)
        
        parser.add_argument("--image_backbone", type=str, default="resnet")
        parser.add_argument("--image_encoder_ckpt", type=str, default=None)
        parser.add_argument("--text_backbone", type=str, default="bert")

        args = parser.parse_args()

    # Load configs from YAML files
    with open(Path(__file__).parent / "configs/image_encoder.yaml") as f:
        args.image_encoder_config = yaml.safe_load(f)
    
    with open(Path(__file__).parent / "configs/text_encoder.yaml") as f:
        args.text_encoder_config = yaml.safe_load(f)

    with open(Path(__file__).parent / "configs/evg.yaml") as f:
        args.evg_config = yaml.safe_load(f)
        
    

    main(args)

import torch
from tqdm import tqdm
from utils.metrics import AverageMeter, accuracy
import numpy as np

def train_one_epoch(args, model, dataloader, optimizer, criterion, device):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    episode_accs = []

    for data, labels in tqdm(dataloader, desc="Training"):
        data, label = data.to(device), labels.to(device)
        
        ##########
        # Episode-based training
        ##########
        n_way = args.n_way
        k_shot = args.k_shot
        q_query = args.q_query
        
        total_per_class = k_shot + q_query
        support_indices = []
        query_indices = []
        
        for i in range(n_way):
            start_idx = i * total_per_class
            end_idx = start_idx + k_shot
            support_indices.extend(range(start_idx, end_idx))
            query_indices.extend(range(end_idx, start_idx + total_per_class))
        
        # Create support and query sets
        support_images = data[support_indices]
        support_labels = label[support_indices]
        query_images = data[query_indices]
        query_labels = label[query_indices]
        
        # Create class names
        unique_classes = torch.unique(support_labels).tolist()
        class_names = [dataloader.dataset.idx2label[c] for c in unique_classes]

        # Move to device
        support_images, support_labels = support_images.to(device), support_labels.to(device)
        query_images, query_labels = query_images.to(device), query_labels.to(device)
        
        unique_classes = torch.unique(support_labels)
        label_map = {cls.item(): i for i, cls in enumerate(unique_classes)}
        support_labels = torch.tensor([label_map[cls.item()] for cls in support_labels], device=support_labels.device)
        query_labels = torch.tensor([label_map[cls.item()] for cls in query_labels], device=query_labels.device)

        # Forward
        logits = model(
            support_images=support_images,
            support_labels=support_labels,
            query_images=query_images,
            class_names=class_names
        )
        
        loss = criterion(logits, query_labels)
        acc = accuracy(logits, query_labels)
        
        ##########
        # End of episode-based training
        ##########
        
        ##########
        # Batch-based training
        ##########
        
        # logits = model(
        #     support_images=data,
        #     support_labels=label,
        #     query_images=data,
        #     class_names=[dataloader.dataset.idx2label[c] for c in label.tolist()]
        # )
        
        # loss = criterion(logits, label)
        # acc = accuracy(logits, label)

        ##########
        # End of batch-based training
        ##########
        
        loss_meter.update(loss.item())
        acc_meter.update(acc)
        episode_accs.append(acc)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    mean_acc = np.mean(episode_accs)
    std_acc = np.std(episode_accs)
    ci95 = 1.96 * std_acc / np.sqrt(len(episode_accs))

    return loss_meter.avg, mean_acc, ci95

def evaluate(args, model, dataloader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    episode_accs = []

    with torch.no_grad():
        for episode_data in tqdm(dataloader, desc="Evaluating"):
            data, label = episode_data
        
            n_way = args.n_way
            k_shot = args.k_shot
            q_query = args.q_query
            
            total_per_class = k_shot + q_query
            support_indices = []
            query_indices = []
            
            for i in range(n_way):
                start_idx = i * total_per_class
                end_idx = start_idx + k_shot
                support_indices.extend(range(start_idx, end_idx))
                query_indices.extend(range(end_idx, start_idx + total_per_class))
            
            # Create support and query sets
            support_images = data[support_indices]
            support_labels = label[support_indices]
            query_images = data[query_indices]
            query_labels = label[query_indices]
            
            # Create class names
            unique_classes = torch.unique(support_labels).tolist()
            class_names = [dataloader.dataset.idx2label[c] for c in unique_classes]

            # Move to device
            support_images, support_labels = support_images.to(device), support_labels.to(device)
            query_images, query_labels = query_images.to(device), query_labels.to(device)
            
            unique_classes = torch.unique(support_labels)
            label_map = {cls.item(): i for i, cls in enumerate(unique_classes)}
            support_labels = torch.tensor([label_map[cls.item()] for cls in support_labels], device=support_labels.device)
            query_labels = torch.tensor([label_map[cls.item()] for cls in query_labels], device=query_labels.device)

            logits = model(
                support_images=support_images,
                support_labels=support_labels,
                query_images=query_images,
                class_names=class_names
            )

            loss = criterion(logits, query_labels)
            acc = accuracy(logits, query_labels)

            loss_meter.update(loss.item())
            acc_meter.update(acc)
            episode_accs.append(acc)
            
    mean_acc = np.mean(episode_accs)
    std_acc = np.std(episode_accs)
    ci95 = 1.96 * std_acc / np.sqrt(len(episode_accs))

    return loss_meter.avg, mean_acc, ci95
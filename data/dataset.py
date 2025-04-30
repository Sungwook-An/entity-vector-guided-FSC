import os
import torch
from glob import glob

from data.fc100 import FC100
from data.cifar_fs import CIFAR_FS
from data.miniimagenet import MiniImageNet
from data.tieredimagenet import TieredImageNet

def get_dataset(name, root, split='train', transform=None, split_file=None):
    """
    Dataset dispatcher function

    Args:
        name (str): Dataset name, one of ['fc100', 'cifar-fs', 'miniimagenet', 'tieredimagenet']
        root (str): Root directory (e.g., './database')
        split (str): One of ['train', 'val', 'test']
        transform (callable): Transform function
        split_file (str or None): Optional class split list

    Returns:
        A dataset instance (subclass of FewShotDataset)
    """

    name = name.lower()

    if name == 'fc100':
        return FC100(root=root, split=split, transform=transform)
    elif name == 'cifar-fs':
        return CIFAR_FS(root=root, split=split, transform=transform)
    elif name == 'miniimagenet':
        return MiniImageNet(root=root, split=split, transform=transform, split_file=split_file)
    elif name == 'tieredimagenet':
        return TieredImageNet(root=root, split=split, transform=transform, split_file=split_file)
    else:
        raise ValueError(f"Unsupported dataset name: {name}")

def validate_dataset(name, root, split, expected_classes=None, min_images_per_class=1):
    """
    주어진 root와 split에 따라 데이터셋이 올바르게 구성되어 있는지 검증합니다.

    Args:
        root (str): 데이터셋 루트 경로
        split (str): 데이터 분할 ('train', 'val', 'test')
        expected_classes (int, optional): 기대하는 클래스 수
        min_images_per_class (int, optional): 각 클래스별 최소 이미지 수

    Raises:
        FileNotFoundError: 경로가 존재하지 않거나 클래스 폴더가 없음
        ValueError: 클래스 수 또는 이미지 수 부족 시 경고
    """
    if name == 'fc100':
        split_path = os.path.join(root, 'FC100', split)
    elif name == 'cifar-fs':
        split_path = os.path.join(root, 'CIFAR-FS', split)
    elif name == 'miniimagenet':
        split_path = os.path.join(root, 'MiniImageNet', split)
    elif name == 'tieredimagenet':
        split_path = os.path.join(root, 'TieredImageNet', split)
    else:
        raise ValueError(f"지원하지 않는 데이터셋 이름: {name}")

    if not os.path.isdir(split_path):
        raise FileNotFoundError(f"[검증 실패] split 경로가 존재하지 않음: {split_path}")

    class_dirs = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    if len(class_dirs) == 0:
        raise FileNotFoundError(f"[검증 실패] {split_path} 내에 클래스 디렉토리가 없습니다.")

    if expected_classes is not None and len(class_dirs) < expected_classes:
        raise ValueError(f"[경고] 클래스 수가 부족합니다: {len(class_dirs)}개 (예상: {expected_classes})")

    for cls in class_dirs:
        cls_path = os.path.join(split_path, cls)
        image_files = glob(os.path.join(cls_path, "*"))
        if len(image_files) < min_images_per_class:
            raise ValueError(f"[경고] '{cls}' 클래스에 이미지가 부족합니다: {len(image_files)}개")

    print(f"[검증 성공] '{split}' split 검증 완료. 총 {len(class_dirs)}개 클래스 존재.")

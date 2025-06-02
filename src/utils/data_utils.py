import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32):
    """
    Create data loaders for training and testing

    Args:
        data_dir (str): Path to the dataset directory
        batch_size (int): Batch size for the data loaders

    Returns: 
        tuple: (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.toTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(
        root=f'{data_dir}/train',
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=f'{data_dir}/test',
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torchvision

import torch.distributed as dist
from tqdm import tqdm
    

def main():

    # Distributed training settings
    dist.init_process_group(backend='nccl')

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    world_size = dist.get_world_size()

    # Hyper-parameters configurations
    num_epochs = 5
    num_classes = 100
    batch_size = 100
    learning_rate = 0.001

    # CIFAR-100 dataset
    train_dataset = torchvision.datasets.CIFAR100(root='./data/',
                                                 train=True,
                                                 transform=torchvision.transforms.ToTensor(),
                                                 download=True) 
    
    test_dataset = torchvision.datasets.CIFAR100(root='./data/',
                                                train=False,
                                                transform=torchvision.transforms.ToTensor())   

    # Split the dataset into train and validation with 80:20 ratio
    torch.manual_seed(42)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Distributed sampler
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)

    # Data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              sampler=sampler)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    # Define the model - ResNet-50
    model = torchvision.models.resnet50(weights=None, num_classes=num_classes).cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)

        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()

            # Forward pass
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
        
            if (i+1) % 50 == 0 and local_rank == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
        # Validate the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.cuda()
                labels = labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            if local_rank == 0:
                print('Accuracy of the model on the 10000 validation images: {} %'.format(100 * correct / total))

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if local_rank == 0:
            print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    

if __name__ == '__main__':
    main()
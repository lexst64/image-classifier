"""
To start training the model, create a directory "training_data" with the train data and
run "train.py" script with an arbitrary positional argument (for example, on Windows, "python.exe train.py model").

This script loads, splits, augments the data, and then trains, evaluates, tests, and saves the model.
It also logs the process to the standard output and tensorboard log files.
"""

import sys

import torch
import torch.nn
import torch.utils
import torch.utils.data
import tqdm
import numpy as np
import torchvision.transforms.v2 as transforms
import os
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from architecture import MyCNN
from dataset import ImagesDataset
from typing import Optional


device = 'cuda'
seed = 123456789
torch.manual_seed(seed)
np.random.seed(seed)


def train_network(
        network: torch.nn.Module,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        n_epochs: int,
        batch_size: int,
        lr: float,
        show_progress: bool,
        early_stopping_threshold: int,
        writer: SummaryWriter,
):
    network = network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    min_eval_loss = float('+Infinity')
    min_eval_loss_epoch = 0

    for epoch in tqdm.tqdm(range(n_epochs), disable=(show_progress == False), desc='epoch', leave=False):
        network.train()

        train_accuracy = 0
        train_loss = 0

        for input_tensor, target_tensor, *rest in tqdm.tqdm(train_dataloader, disable=(show_progress == False),
                                                            desc='train', leave=False):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            output = network(input_tensor).to(device)
            loss = loss_fn(output, target_tensor)

            train_loss += loss.item() / len(train_dataloader)
            train_accuracy += torch.sum((output.argmax(dim=1) == target_tensor)).item() / len(train_dataset)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        network.eval()

        eval_accuracy = 0
        eval_loss = 0

        with torch.no_grad():
            for input_tensor, target_tensor, *rest in tqdm.tqdm(eval_dataloader, disable=(show_progress == False),
                                                                desc='eval', leave=False):
                input_tensor = input_tensor.to(device)
                target_tensor = target_tensor.to(device)

                output = network(input_tensor).to(device)
                loss = loss_fn(output, target_tensor)

                eval_loss += loss.item() / len(eval_dataloader)
                eval_accuracy += torch.sum((output.argmax(dim=1) == target_tensor)).item() / len(eval_dataset)

        writer.add_scalar('Loss/eval', eval_loss, epoch)
        writer.add_scalar('Accuracy/eval', eval_accuracy, epoch)

        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            min_eval_loss_epoch = epoch
            torch.save(network.state_dict(), "model.pth")
        else:
            if (epoch - min_eval_loss_epoch) >= early_stopping_threshold:
                print(f'training stopped early after {epoch + 1} epochs')
                break


class TransformedImagesDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__()

        self._transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomInvert(p=0.5),
            transforms.RandomChoice([
                transforms.RandomRotation(degrees=(-180, 180)),
                transforms.RandomPerspective(0.2, 0.5),
            ]),
            transforms.GaussianBlur((3, 3)),
            transforms.RandomAdjustSharpness(2, p=0.5),
            transforms.RandomAutocontrast(p=0.5),
        ])
        self._images = []

        n_augmentations = 9

        for image, *rest in tqdm.tqdm(dataset, desc='augmentation', leave=False):
            self._images.append((image, *rest))

            for _ in range(n_augmentations):
                self._images.append((self._transforms(image), *rest))

    def __getitem__(self, index: int) -> tuple:
        return self._images[index]

    def __len__(self) -> int:
        return len(self._images)


def load_datasets(data_path: str, train_size: float, eval_size: float, test_size: float) -> tuple[
    Dataset, Dataset, Dataset]:
    image_dataset = ImagesDataset(data_path, 100, 100)
    train_dataset, eval_dataset, test_dataset = torch.utils.data.random_split(image_dataset,
                                                                              [train_size, eval_size, test_size])
    return train_dataset, eval_dataset, test_dataset


def test_network(network: torch.nn.Module,
                 test_dataset: Dataset,
                 batch_size: int,
                 show_progress: bool,
                 writer: Optional[SummaryWriter] = None):
    network = network.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loss = 0
    test_accuracy = 0

    with torch.no_grad():
        network.eval()

        for input_tensor, target_tensor, *rest in tqdm.tqdm(test_dataloader, disable=(show_progress == False),
                                                            desc='test best model:', leave=False):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            output = network(input_tensor).to(device)
            loss = loss_fn(output, target_tensor)

            test_loss += loss.item() / len(test_dataloader)
            test_accuracy += torch.sum((output.argmax(dim=1) == target_tensor)).item() / len(test_dataset)

    print(f'Test loss: {test_loss}\nTest accuracy: {test_accuracy}')
    if writer:
        writer.add_text('Test/loss', str(test_loss))
        writer.add_text('Test/accuracy', str(test_accuracy))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('logname', type=str)
    args = parser.parse_args()

    train_dataset, eval_dataset, test_dataset = load_datasets(
        data_path='training_data',
        train_size=0.90,
        eval_size=0.05,
        test_size=0.05,
    )

    # ######
    # network = MyCNN()
    # network.load_state_dict(torch.load('model_final.pth'))
    # test_network(
    #     network=network,
    #     test_dataset=test_dataset,
    #     batch_size=32,
    #     show_progress=True,
    # )
    #
    # sys.exit(0)
    # ######

    train_dataset_filename = 'train_dataset_dump.pickle'
    if os.path.exists(train_dataset_filename):
        with open(train_dataset_filename, 'rb') as f:
            print('loading dumped augmented train dataset...')
            train_dataset = pickle.load(f)
            print('finished')
    else:
        train_dataset = TransformedImagesDataset(train_dataset)
        with open(train_dataset_filename, 'wb') as f:
            print('dumping augmented train dataset...')
            pickle.dump(train_dataset, f)
            print('finished')

    model = MyCNN()
    writer = SummaryWriter(f'runs/{args.logname}')

    try:
        train_network(
            network=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            n_epochs=20,
            batch_size=32,
            lr=0.0001,
            early_stopping_threshold=5,
            show_progress=True,
            writer=writer,
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'model_final.pth')
    except Exception as e:
        torch.save(model.state_dict(), 'model_final.pth')
        raise e

    torch.save(model.state_dict(), 'model_final.pth')

    network = MyCNN()
    network.load_state_dict(torch.load('model.pth'))
    # test the best performing model
    test_network(
        network=network,
        test_dataset=test_dataset,
        batch_size=48,
        show_progress=True,
        writer=writer,
    )
    sys.exit(0)

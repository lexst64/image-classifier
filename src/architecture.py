import torch
import torch.nn as nn


seed = 123456789
torch.manual_seed(seed)


class MyCNN(nn.Module):
	def __init__(self) -> None:
		super().__init__()

		self._sequence = nn.Sequential(
			nn.Conv2d(1, 64, 3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),

			nn.Conv2d(64, 64, 3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2),

			nn.Conv2d(64, 128, 3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),

			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2),

			nn.Conv2d(128, 256, 3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),

			nn.Conv2d(256, 256, 3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),

			nn.Conv2d(256, 256, 3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2),

			nn.Conv2d(256, 512, 3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),

			nn.Conv2d(512, 512, 3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),

			nn.Conv2d(512, 512, 3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),

			nn.Conv2d(512, 512, 3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2),

			nn.Flatten(),

			nn.Dropout(p=0.5),
			nn.LazyLinear(out_features=2048),
			nn.ReLU(),

			nn.Dropout(p=0.5),
			nn.Linear(in_features=2048, out_features=2048),
			nn.ReLU(),

			nn.Linear(in_features=2048, out_features=20),
		)
		
	def forward(self, input_images: torch.Tensor) -> torch.Tensor:
		return self._sequence(input_images)


model = MyCNN()

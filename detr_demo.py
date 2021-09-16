import  torch
import torch.nn as nn
from torchvision.models import resnet50

class DETR(nn.Module):

	def __init__(self):
		self.backbone = resnet50()
		def self.backbone.fc

		self.conv = nn.Conv2d(2048,256, kernel_size=1)

		self.transformer = nn.Transformer(256, 8, 6,6)
		self.classes = nn.Linear(256,92)
		self.boxes = nn.Linear(256,4)

		self.query = torch.zeros((100,256))

	def forward(self, x):
		x = self.backbone(x)
		x = self.conv(x)
		x = self.transformer( self.pos+x.flatten(2).permute(2,0,1), self.query.unsqueeze(1)).transpose(0,1)
		x1 = self.classes(x)
		x2 = self.boxes(x)







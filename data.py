### Upsampling
lis = []
from PIL import Image
import torch
import os
from torchvision import transforms
transform = transforms.Compose([
   transforms.RandomChoice([
   transforms.RandomHorizontalFlip(),
   transforms.RandomRotation(degrees=45),
   transforms.RandomGrayscale(p=0.2),
   ])
])
import random

for folder in os.listdir("train"):
  lis.append(len(os.listdir("train/" + folder)))
max_file = max(lis)
print("sum is", sum(lis))
print("max number of files are: ", max_file)
count = 0
for folder in os.listdir("train"):
  if (len(os.listdir("train/" + folder)) < max_file and len(os.listdir("train/" + folder)) > 0):
    prev = os.listdir("train/" + folder)
    i = 0
    while (len(os.listdir("train/" + folder)) < max_file):
      idx = random.randint(0,len(prev)-1)
      file = prev[idx]
      im = Image.open("train/" + folder + "/" + file)
      trans = transform(im)
      trans.save("train/" + folder + "/" + file[:-4] + str(i + i*7 + i*9) + file[-4:])
      i+=1

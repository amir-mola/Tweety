import os
import torchvision.models as models
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision import transforms as T
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from PIL import Image
import csv
from tqdm import tqdm
torch.manual_seed(12)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.chdir("birds21sp/birds/")

def train(net, dataloader,val_loader, epochs=1, lr=0.01, momentum=0.9, decay=0.0005, verbose=1):
  net.to(device)
  losses = []
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
  print("Starting training")
  for epoch in range(epochs):
    print("Epoch {0}".format(epoch))
    sum_loss = 0.0
    if(epoch % 2 == 1):
      torch.save(model.state_dict(), 'trained-dense-{}.pth'.format(epoch))
    net.train()
    for i, batch in enumerate(tqdm(dataloader,position=0, leave=True)):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = batch[0].to(device), batch[1].to(device)
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize 
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()  # autograd magic, computes all the partial derivatives
      optimizer.step() # takes a step in gradient direction
      # print statistics
      losses.append(loss.item())
      sum_loss += loss.item()
      if i % 100 == 99:    # print every 100 mini-batches
          if verbose:
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, sum_loss / 100))
          sum_loss = 0.0
  return losses

def get_data_loaders(data_dir, batch_size):
  transforms_train = transforms.Compose([
   transforms.Resize((256, 256)),
   transforms.CenterCrop(224),
   transforms.RandomChoice([
   transforms.RandomRotation(degrees=45),
   transforms.RandomGrayscale(p=0.3),
   ]),
   transforms.ToTensor(),
  ])

  all_data = datasets.ImageFolder(data_dir, transform=transforms_train)
  train_loader = DataLoader(all_data, batch_size=batch_size, shuffle=True)
  return train_loader, all_data.classes

print("Getting train loader")
train_loader, label_classes = get_data_loaders("train/", 64)


print("loading the weights")


model = models.densenet161(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 555)


print("Starting to train")
losses = train(model, train_loader, None, epochs=15)
print("Finished training")
torch.save(model.state_dict(), 'trained-dense-15.pth')


def test_accuracy(model, file):
  model.to(device)
  model.eval()
  with torch.no_grad():
    transform = transforms.Compose([
          transforms.Resize((256, 256)),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
    ])
    im = Image.open(file)
    x = transform(im).unsqueeze(0).to(device)
    pred = model(x)
    _, predicted = torch.max(pred.data, 1)
    return label_classes[predicted]
w = open('densenet.csv', mode='w')
fieldnames = ['path', 'class']

writer = csv.DictWriter(w, fieldnames=fieldnames)
writer.writeheader()
with  open('sample.csv' ,mode='r') as csv_file:
  csv_reader = csv.DictReader(csv_file)
  line_count = 0
  for row in csv_reader:
    if line_count == 0:
      print(f'Column names are {", ".join(row)}')
      line_count += 1
    pred = test_accuracy(model, "test/0/" + row["path"][5:])
    writer.writerow({'path': row["path"], 'class': pred})
    w.flush()
    line_count += 1
    if (line_count % 100 == 99):
      print("Processed ", line_count, " lines so far")
  print(f'Processed {line_count} lines.')


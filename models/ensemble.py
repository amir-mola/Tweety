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

def get_data_loaders():
  transform_train = transforms.Compose([
   transforms.Resize((256, 256)),
   transforms.CenterCrop(224),
   transforms.RandomChoice([
   transforms.RandomHorizontalFlip(),
   transforms.RandomRotation(degrees=45),
   transforms.RandomGrayscale(p=0.3),
   ]),
   transforms.ToTensor(),
  ])

  transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
  ])
  import multiprocessing
  kwargs = {'num_workers': multiprocessing.cpu_count(),'pin_memory': True}

  trainset = torchvision.datasets.ImageFolder(root='train/', transform=transform_train)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True,**kwargs)

  testset = torchvision.datasets.ImageFolder(root='test/', transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, **kwargs)
  classes = open("names.txt").read().strip().split("\n")
  class_to_idx = trainset.class_to_idx
  idx_to_class = {int(v): int(k) for k, v in class_to_idx.items()}
  idx_to_name = {k: classes[v] for k,v in idx_to_class.items()}
  return {'train': trainloader, 'test': testloader, 'to_class': idx_to_class, 'to_name':idx_to_name}

def predict(net1, net2, net3, net4, dataloader, data, ofname):
    out = open(ofname, 'w')
    out.write("path,class\n")
    net1.to(device)
    net1.eval()
    net2.to(device)
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader, 0):
            if i%100 == 0:
                print(i)
            images, labels = images.to(device), labels.to(device)
            outputs = torch.div(net1(images) + net2(images) + net3(images) + net4(images), 4)
            _, predicted = torch.max(outputs.data, 1)
            fname, _ = dataloader.dataset.samples[i]
            out.write("test/{},{}\n".format(fname.split('/')[-1], data['to_class'][predicted.item()]))
    out.close()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    os.chdir("birds21sp/birds/")

    model1 = models.densenet161(pretrained=True)
    model1.classifier = nn.Linear(model1.classifier.in_features, 555)
    model1.to(device)
    model1 = nn.DataParallel(model1)
    model1.load_state_dict(torch.load('trained-dense-15.pth'))

    model2 = models.resnet152(pretrained=True)
    model2.fc = nn.Linear(model2.fc.in_features, 555)
    model2 = nn.DataParallel(model2)
    model2.to(device)
    model2.load_state_dict(torch.load('trained-resnet-15.pth'))

    model3 = models.resnext101_32x8d(pretrained=True)
    model3.fc = nn.Linear(model3.fc.in_features, 555)
    model3 = nn.DataParallel(model3)
    model3.to(device)
    model3.load_state_dict(torch.load('trained-next-15.pth'))
    
    model4 = models.vgg19_bn(pretrained=True)
    model4.classifier[6] = nn.Linear(model4.classifier[6].in_features, 555)
    model4.to(device)
    model4 = nn.DataParallel(model4)
    model4.load_state_dict(torch.load('trained-vgg-15.pth'))


    data = get_data_loaders()
    predict(model1, model2, model3, model4, data['test'], data, "combo.csv")

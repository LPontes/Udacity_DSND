# Import libraries
import argparse
import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import time
import json
'''
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu
'''
parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('--arch', default = "vgg19")
parser.add_argument('--learning_rate', type = float, default = 0.001)
parser.add_argument('--hidden_units', type = int, default = 4096 )
parser.add_argument('--epochs', type = int, default = 8)
parser.add_argument('--gpu', action = 'store_true', default = False)
parser.add_argument('--save_directory')

args = parser.parse_args()
data_dir = args.data_dir
device = 'cpu'

if args.arch:
    arch = args.arch
if args.hidden_units:
    hidden_units = args.hidden_units
if args.learning_rate:
    learning_rate = args.learning_rate
if args.epochs:
    epochs = args.epochs
if args.gpu:        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
#Load and transform images
def load_datasets(train_dir = data_dir + '/train', valid_dir = data_dir + '/valid'):
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomRotation(30),                                     
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])
                                     ])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])
                                     ])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data= datasets.ImageFolder(valid_dir, transform=valid_transforms)

    #Define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size= 64, shuffle = True)  
    validloader = torch.utils.data.DataLoader(valid_data, batch_size= 64) 
  
    return trainloader, validloader, train_data

trainloader, validloader, train_data = load_datasets()

#Create model
def create_model(arch=arch,hidden_units =hidden_units,lr=learning_rate):
    '''Create a NN model'''
    #pre-trained model
    model = getattr(models,arch)(pretrained = True)
    in_features = model.classifier[0].in_features
        
    #Freeze features parameters
    for param in model.parameters():
        param.requires_grad = False
    
    #Define classifier
    classifier = nn.Sequential(
        nn.Linear(in_features,hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.5),        
        nn.Linear(hidden_units,102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    model.to(device)

    return model, criterion, optimizer

model, criterion, optimizer = create_model()

def train_model(model, criterion, optimizer,epochs):
    steps = 0
    running_loss = 0
    t0 = time.time()

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch+1, epochs))
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)        
            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item()
        test_loss = 0
        accuracy = 0    
        model.eval()    
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                
                test_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Train loss: {running_loss/len(trainloader):.3f}.. ")
        print(f"Validation loss: {test_loss/len(validloader):.3f}.. "
             f"Validation accuracy: {accuracy/len(validloader):.3f}")
                 
        running_loss = 0
        model.train()

    time_elapsed = time.time() - t0
    print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
        
    return model

#Run the function to train the model
model_trained = train_model(model, criterion, optimizer,epochs)

def save_checkpoint(model_trained):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': model.classifier[0].in_features,
                  'hidden_units': hidden_units,
                  'output_size': 102,
                  'arch' : arch,
                  'learning_rate': learning_rate,
                  'batch_size':64,
                  'epochs':epochs,
                  'optimizer':optimizer.state_dict(),
                  'class_to_idx':model.class_to_idx,
                  'classifier_state_dict': model.classifier.state_dict()}
    torch.save(checkpoint, 'checkpoint.pth')

save_checkpoint(model_trained)
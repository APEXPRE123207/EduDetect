import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import time

is_cuda = torch.cuda.is_available()

# Define transformations with data augmentation
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

"""It is advised to distribute the data into 70-30 ratio for training and validation respectively."""
"""You can change it and try it out also""" 
# Load datasets
train = ImageFolder("Location of training data stored", train_transform)
valid = ImageFolder("Location of Validating data stored", valid_transform)




#the model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5) #this should be (1,10,kernel_size=5) as only b&w pics is being trained
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3380, 50)
        self.fc2 = nn.Linear(50, 17) #This should be 6 not 17, cause only 6 shape classes are there

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)
model = Net()
if is_cuda:
    model.cuda()

#optimizer and scheduler
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training and validation function
def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    else:
        model.eval()
    running_loss = 0.0
    running_correct = 0
    print(f"\n{phase} - Processing batches:")
    for batch_idx, (data, target) in enumerate(data_loader):
        print(f"  Batch {batch_idx+1}/{len(data_loader)}", end='\r')
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        running_loss += loss.item()
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()
    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)
    print(f'{phase} loss is {loss} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)} = {accuracy}')
    return loss, accuracy



# Main training loop
if __name__ == '__main__':
    train_data_loader = DataLoader(train, batch_size=32, num_workers=3, shuffle=True)
    valid_data_loader = DataLoader(valid, batch_size=32, num_workers=3, shuffle=False)
    print("Checking data loader...")
    for i, batch in enumerate(train_data_loader):
        print(f"Loaded batch {i+1} with {len(batch[1])} samples")
        if i == 2:  # Check first 3 batches
         break
    print("Data loading check completed!\n")
    train_losses , train_accuracy = [],[]
    val_losses , val_accuracy = [],[]
    num_epochs = 50  # Total number of epochs
    
    for epoch in range(1, num_epochs+1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        print('-' * 10)
        

        start_time = time.time()
        
        print(f'Starting Training Phase...')
        epoch_loss, epoch_accuracy = fit(epoch, model, train_data_loader, phase='training')
        
        print(f'\nStarting Validation Phase...')
        val_epoch_loss , val_epoch_accuracy = fit(epoch, model, valid_data_loader, phase='validation')
        
        # Calculate epoch duration
        epoch_mins, epoch_secs = divmod(time.time()-start_time, 60)
        
        # Store metrics
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
        
        print(f'\nEpoch {epoch} completed in {epoch_mins:.0f}m {epoch_secs:.0f}s')
        print('=' * 50)  

    print(f'\nTraining completed for all {num_epochs} epochs!')
    print('Plotting results...')

    plt.plot(range(1, len(train_losses)+1), train_losses, 'bo', label='training loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, 'r', label='validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.show()

    torch.save(model.state_dict(), "shapes_model.pth")
    print("Model saved as shapes_model_v2.pth Successfully")
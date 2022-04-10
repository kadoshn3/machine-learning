import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np

plt.close('all')

# number of samples
batch_size = 20
# learning rate
lr = .001
# number of epochs
n_epochs = 10

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = torchvision.datasets.FashionMNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = torchvision.datasets.FashionMNIST(root='data', train=False,
                                  download=True, transform=transform)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


## NN architecture
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.nodes = 100
        self.layer_1 = nn.Linear(28*28, self.nodes)
        self.layer_2 = nn.Linear(self.nodes, self.nodes)
        self.layer_3 = nn.Linear(self.nodes, 10)
        
    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28*28)
        # add hidden layer, with relu activation function
        x = nn.functional.relu(self.layer_1(x))
        return x
    
# Initialize the NN
model = NeuralNet()
    
# Loss function - Cross Entropy Loss
criterion = nn.CrossEntropyLoss()

# Optimizer Adam
optimizer = optim.Adam(model.parameters(), lr=lr)

####### Train the model #######
model.train()
for epoch in range(n_epochs):
    # Monitor training loss
    train_loss = 0.0
    
    for data, target in train_loader:
        # Clear the gradients
        optimizer.zero_grad()
        # Forward function in Neural Net Class
        output = model(data)
        # Loss Computation
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)
        
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)

    print('Epoch: {} \tTraining Accuracy: {:.2f}%'.format(epoch+1, 100-train_loss))

####### Test the model #######
# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # prep model for *evaluation*

for data, target in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('\nTesting Accuracy: {:.2f}\n'.format(100-test_loss))

for i in range(10):
    #if class_total[i] > 0:
    print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

##### Visualize data ######
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()

# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds = torch.max(output, 1)
# prep images for display
images = images.numpy()

# plot the images in the batch, along with predicted and true labels
fig = plt.figure()
for idx in np.arange(36):
    ax = fig.add_subplot(6, 6, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())),
                 color=("green" if preds[idx]==labels[idx] else "red"))

plt.show()
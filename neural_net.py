from torch import nn, load, optim, randperm, device, save, no_grad, utils, float32
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

from pathlib import Path
from math import sqrt
from time import time

from plot_losses import plot_losses

num_workers = 0
batch_size = 256

gpu = device('cuda:0')
data_dir = 'data'
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
train_dataset = ImageFolder(root=data_dir+'/train', transform=transform)
test_dataset = ImageFolder(root=data_dir+'/validation', transform=transform)

train_iter = DataLoader(dataset=train_dataset, batch_size=1027, shuffle=True, num_workers=num_workers)
test_iter = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, num_workers=num_workers)

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

TESTING_HYPERPARAM = 'learning_rate'
TEST_RUNS = 1
for test_run in range(1, TEST_RUNS + 1):
    # Hyperparameters #
    #-----------------#
    # NUM_FEATURES = data['train'][0].size()[0] - 1
    # NUM_HIDDEN = 40
    # NUM_NEURONS_LAYER = [NUM_FEATURES] + [1024] * NUM_HIDDEN
    #DROPOUT_PROBS = [0 + 0.001 * test_run] + [0.2 + 0.01 * test_run] * (NUM_HIDDEN - 1)
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 200
    WEIGHT_DECAY = 0
    #-----------------#

    sequential_list = [
        nn.Conv2d(3, 5, kernel_size=7, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=8, stride=8),

        nn.Flatten(),
        nn.Linear(5 * 16 * 16, 100),
        nn.ReLU(),
        nn.Linear(100, 1),
        nn.Sigmoid()
    ]
    # for i in range(len(NUM_NEURONS_LAYER) - 1):
    #     sequential_list.append(nn.Linear(NUM_NEURONS_LAYER[i], NUM_NEURONS_LAYER[i + 1]))
    #     sequential_list.append(nn.ReLU())
    #     #sequential_list.append(nn.Dropout(DROPOUT_PROBS[i]))
    # sequential_list.append(nn.Linear(NUM_NEURONS_LAYER[-1], 1)) 
    # sequential_list.append(nn.Sigmoid())
    net = nn.Sequential(*sequential_list)
    net.to(gpu)
    net.apply(init_weights)

    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.SGD(params=net.parameters(), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    
    epoch_loss = []
    net.train()

    
    for X, Y in train_iter:
        X, Y = X.to(gpu), Y.to(gpu)
        for epoch in range(NUM_EPOCHS):
            print('\rEpoch: ' + str(epoch + 1) + ' / ' + str(NUM_EPOCHS), end = '')

            optimizer.zero_grad()
            Y_hat = net(X)
            loss = criterion(Y_hat.squeeze().to(float32), Y.squeeze().to(float32))
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss)
    print()

    for epoch in range(NUM_EPOCHS):
        epoch_loss[epoch] = epoch_loss[epoch].item()

    net.eval()

    with no_grad():
        correct_predictions = 0
        total_predictions = 0
        losses = []
        for X, Y in train_iter:
            X, Y = X.to(gpu), Y.to(gpu)
            Y_hat = net(X)
            losses.append(criterion(Y_hat.squeeze().to(float32), Y.squeeze().to(float32)))
            predictions = (Y_hat.squeeze() > 0.5) == Y.squeeze()
            correct_predictions += predictions.sum()
            total_predictions += predictions.numel()
    
    training_loss = sum(losses) / len(losses)
    training_accuracy = correct_predictions / total_predictions
    print('Training set loss: ' + str(training_loss.item()))
    print('Training set accuracy: ' + str(training_accuracy.item()))

    with no_grad():
        correct_predictions = 0
        total_predictions = 0
        losses = []
        for X, Y in test_iter:
            X, Y = X.to(gpu), Y.to(gpu)
            Y_hat = net(X)
            losses.append(criterion(Y_hat.squeeze().to(float32), Y.squeeze().to(float32)))
            predictions = (Y_hat.squeeze() > 0.5) == Y.squeeze()
            correct_predictions += predictions.sum()
            total_predictions += predictions.numel()

    validation_loss = sum(losses) / len(losses)
    validation_accuracy = correct_predictions / total_predictions
    print('Validation set loss: ' + str(validation_loss.item()))
    print('Validation set accuracy: ' + str(validation_accuracy.item()))

    with open('tuning_results/' + TESTING_HYPERPARAM + '/accuracies_' + str(test_run), 'a') as accuracies:
        accuracies.write(
                    '-' * 20 +
                    '\nLEARNING_RATE = ' + str(LEARNING_RATE) +
                    '\nNUM_EPOCHS = ' + str(NUM_EPOCHS) +
                    '\nWEIGHT_DECAY = ' + str(WEIGHT_DECAY) +
                    #'\nDROPOUT_PROBS = ' + str(DROPOUT_PROBS) +
                    '\n' + 
                    '\nTraining set loss:\n' + str(training_loss) +
                    '\nTraining set accuracy:\n' + str(training_accuracy) +
                    '\n' +
                    '\nValidation set loss:\n' + str(validation_loss) +
                    '\nValidation set accuracy:\n' + str(validation_accuracy) +
                    '\n' +
                    '-' * 20 +
                    '\n'
        )

    #save(net, "mymodel.pt")
    # fig_path = "tuning_results/graphs/BCELoss" + str(test_run) + ".png"
    # plot_losses(epoch_loss, fig_path)

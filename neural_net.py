from torch import nn, load, optim, randperm, device, save, no_grad
from pathlib import Path
from math import sqrt
from time import time

from torch.nn.modules.activation import ReLU
from torch.nn.modules.dropout import Dropout
from plot_losses import plot_losses

data_dir = 'data'
data_filename = 'data.pt'
data = load(Path(__file__).parent.absolute() / data_dir / data_filename)

num_train_examples = data['train'].size()[0]
num_val_examples = data['validation'].size()[0]

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

TESTING_HYPERPARAM = 'learning_rate'
TEST_RUNS = 1
for test_run in range(1, TEST_RUNS + 1):
    # Hyperparameters #
    #-----------------#
    NUM_FEATURES = data['train'][0].size()[0] - 1
    NUM_HIDDEN = 40
    NUM_NEURONS_LAYER = [NUM_FEATURES] + [1024] * NUM_HIDDEN
    #DROPOUT_PROBS = [0 + 0.001 * test_run] + [0.2 + 0.01 * test_run] * (NUM_HIDDEN - 1)
    LEARNING_RATE = 0.02
    NUM_EPOCHS = 1000
    INIT_MEAN, INIT_STD = 0, 1
    WEIGHT_DECAY = 0.002
    #-----------------#

    sequential_list = []
    for i in range(len(NUM_NEURONS_LAYER) - 1):
        sequential_list.append(nn.Linear(NUM_NEURONS_LAYER[i], NUM_NEURONS_LAYER[i + 1]))
        sequential_list.append(nn.ReLU())
        #sequential_list.append(nn.Dropout(DROPOUT_PROBS[i]))
    sequential_list.append(nn.Linear(NUM_NEURONS_LAYER[-1], 1)) 
    sequential_list.append(nn.Sigmoid())
    net = nn.Sequential(*sequential_list)
    net.to(device('cuda:0'))
    net.apply(init_weights)

    criterion = nn.BCELoss(reduction='mean')
    trainer = optim.SGD(params=net.parameters(), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY)

    X_train = data['train'][:, 0:-1].to(device('cuda:0'))
    Y_train = data['train'][:, -1].reshape(num_train_examples).to(device('cuda:0'))

    X_val = data['validation'][:, 0:-1].to(device('cuda:0'))
    Y_val = data['validation'][:, -1].reshape(num_val_examples).to(device('cuda:0'))

    epoch_loss = []
    for epoch in range(NUM_EPOCHS):
        print('\rEpoch: ' + str(epoch + 1) + ' / ' + str(NUM_EPOCHS), end = '')
        net.eval()
        with no_grad():
            Y_hat_val = net(X_val)
            validation_loss = criterion(Y_hat_val.reshape(num_val_examples), Y_val)
        net.train()
        Y_hat = net(X_train)
        training_loss = criterion(Y_hat.reshape(num_train_examples), Y_train)
        epoch_loss.append([training_loss, validation_loss])
        trainer.zero_grad()
        training_loss.backward()
        trainer.step()
    print()

    for epoch in range(NUM_EPOCHS):
        for i in range(2):
            epoch_loss[epoch][i] = epoch_loss[epoch][i].item()

    net.eval()
    training_loss = epoch_loss[-1][0]
    training_accuracy = (((net(X_train) > 0.5).reshape(num_train_examples) == Y_train).sum() / num_train_examples).item()
    print('Training set loss: ' + str(training_loss))
    print('Training set accuracy: ' + str(training_accuracy))

    validation_loss = epoch_loss[-1][1]
    validation_accuracy = (((net(X_val) > 0.5).reshape(num_val_examples) == Y_val).sum() / num_val_examples).item()
    print('Validation set loss: ' + str(validation_loss))
    print('Validation set accuracy: ' + str(validation_accuracy))

    with open('tuning_results/' + TESTING_HYPERPARAM + '/accuracies_' + str(test_run), 'a') as accuracies:
        accuracies.write(
                    '-' * 20 +
                    '\nNUM_FEATURES = ' + str(NUM_FEATURES) + str(' (3x') + str((int(sqrt(NUM_FEATURES / 3)))) + 'x' +  str((int(sqrt(NUM_FEATURES / 3))))  + ')' +
                    '\nNUM_NEURONS_LAYER = ' + str(NUM_NEURONS_LAYER) +
                    '\nLEARNING_RATE = ' + str(LEARNING_RATE) +
                    '\nNUM_EPOCHS = ' + str(NUM_EPOCHS) +
                    '\nINIT_MEAN, INIT_STD = ' + str(INIT_MEAN) + ' ' + str(INIT_STD) +
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

    fig_path = "tuning_results/graphs/BCELoss" + str(test_run) + ".png"
    plot_losses(epoch_loss, fig_path)

    del X_train, Y_train, X_val, Y_val, net, Y_hat, Y_hat_val, epoch_loss
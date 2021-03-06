from torch import nn, load, optim, randperm, device
from pathlib import Path
from math import sqrt

data_dir = 'data'
data_filename = 'data.pt'
data = load(Path(__file__).parent.absolute() / data_dir / data_filename)

# Hyperparameters #
#-----------------#
NUM_FEATURES = data['train'][0].size()[0] - 1
NUM_NEURONS_LAYER = [4000, 1000, 500, 1]
LEARNING_RATE = 0.0002
NUM_EPOCHS = 200
#-----------------#

data['train'] = data['train'].to(device('cuda:0'))
rand_indx = randperm(data['train'].size()[0])
data['train'] = data['train'][rand_indx]

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.1)
        nn.init.constant_(m.bias, 0)

net = nn.Sequential(
    nn.Linear(NUM_FEATURES, NUM_NEURONS_LAYER[0]),
    nn.ReLU(),
    nn.Linear(NUM_NEURONS_LAYER[0], NUM_NEURONS_LAYER[1]),
    nn.ReLU(),
    nn.Linear(NUM_NEURONS_LAYER[1], NUM_NEURONS_LAYER[2]),
    nn.ReLU(),
    nn.Linear(NUM_NEURONS_LAYER[2], NUM_NEURONS_LAYER[3]),
    nn.Sigmoid()
)

net.to(device('cuda:0'))
net.apply(init_weights)
criterion = nn.MSELoss()
trainer = optim.SGD(params=net.parameters(), lr=LEARNING_RATE)
print('Currently in epoch ', end = '')
for epoch in range(NUM_EPOCHS):
    print(str(epoch + 1), end = ' ')
    for image in data['train']:
        X = image[0:-1]
        y = image[-1]
        y_hat = net(X)
        loss = criterion(y_hat, y)
        trainer.zero_grad()
        loss.backward()
        trainer.step()

total_correct = 0
for image in data['train']:
    x = image[0:-1]
    y = image[-1]
    if int(0.5 + net(x)) == y:
        total_correct += 1
print('Training set accuracy:')
print(total_correct, data['train'].size()[0])
accuracy_train = total_correct / data['train'].size()[0]
print(accuracy_train)

total_correct = 0
for image in data['validation']:
    x = image[0:-1]
    y = image[-1]
    if int(0.5 + net.to(device('cpu'))(x)) == y:
        total_correct += 1
print('Validation set accuracy:')
print(total_correct, data['validation'].size()[0])
accuracy_val = total_correct / data['validation'].size()[0]
print(accuracy_val)

with open('accuracies', 'a') as accuracies:
    accuracies.write(
                '-' * 20 +
                '\nNUM_FEATURES = ' + str(NUM_FEATURES) + str(' (3x') + str((int(sqrt(NUM_FEATURES / 3)))) + 'x' +  str((int(sqrt(NUM_FEATURES / 3))))  + ')' +
                '\nNUM_NEURONS_LAYER = ' + str(NUM_NEURONS_LAYER) +
                '\nLEARNING_RATE = ' + str(LEARNING_RATE) +
                '\nNUM_EPOCHS = ' + str(NUM_EPOCHS) +
                '\n' + 
                '\nTraining set accuracy: ' + str(accuracy_train) +
                '\nValidation set accuracy: ' + str(accuracy_val) +
                '\n' +
                '-' * 20 +
                '\n'
    )

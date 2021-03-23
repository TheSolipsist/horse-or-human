from matplotlib import pyplot as plt

def plot_losses(epoch_loss, fig_path):
    train_loss = [losses[0] for losses in epoch_loss]
    val_loss = [losses[1] for losses in epoch_loss]
    x_axis_indices = list(range(1, len(epoch_loss) + 1))
    x_axis_ticks = list(range(1, len(epoch_loss) + 1, (len(epoch_loss) + 1) // 7))
    plt.ylim(bottom=0, top=max(max(train_loss), max(val_loss))+1)
    plt.plot(x_axis_indices, train_loss, 'r', label = 'Training set')
    plt.plot(x_axis_indices, val_loss, 'b', label='Validation set')
    plt.xticks(x_axis_ticks)
    plt.xlabel('Epoch')
    plt.ylabel('BCELoss')
    plt.title('Loss at each epoch')
    plt.legend()
    plt.savefig(fig_path)
    plt.show()
    plt.close()
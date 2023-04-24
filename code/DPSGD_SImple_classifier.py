import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

if __name__ == '__main__':

    train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    batch_size = 100
    n_iters = 6000
    epochs = n_iters / (len(train_dataset) / batch_size)
    input_dim = 784
    output_dim = 10
    lr_rate = 0.1
    clip_value = 0.01  # This is the maximum value that a gradient component can have
    noise_scale = 8  # This is the scale of the added noise. The larger the scale, the less privacy is preserved

    np.random.seed(42)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = LogisticRegression(input_dim, output_dim)
    criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)

    iter = 0

    for epoch in range(int(epochs)):
        for i, (images, labels) in enumerate(train_loader):

            all_per_sample_gradients = [] # will have len = batch_size

            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Clip gradients to limit their sensitivity
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            # Add Gaussian noise to the clipped gradients
            for param in model.parameters():
                per_sample_gradient = param.grad / batch_size
                noise = torch.randn_like(per_sample_gradient) * (noise_scale * clip_value)
                param.grad = per_sample_gradient + noise

            optimizer.step()

            iter += 1
            if iter % 500 == 0:
                # calculate Accuracy
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = Variable(images.view(-1, 28 * 28))
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    # for gpu, bring the predicted and labels back to cpu fro python operations to work
                    correct += (predicted == labels).sum()
                accuracy = 100

                accuracy = 100 * correct / total
                print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))

                 # calculate differential privacy
            
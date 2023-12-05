"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""
import time

import minitorch


def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        # TODO: Implement for Task 2.5.
        h1 = self.layer1.forward(x).relu()
        h2 = self.layer2.forward(h1).relu()
        return self.layer3.forward(h2).sigmoid()
        # raise NotImplementedError("Need to implement for Task 2.5")


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        # TODO: Implement for Task 2.5.
        batch_size, in_size = x.shape
        return (
            self.weights.value.view(1, in_size, self.out_size)
            * x.view(batch_size, in_size, 1)
        ).sum(1).view(batch_size, self.out_size) + self.bias.value.view(self.out_size)
        # raise NotImplementedError("Need to implement for Task 2.5")


def default_log_fn(epoch, total_loss, correct, losses, time_taken):
    print(
        "Epoch ", epoch, ": loss ", total_loss, "correct", correct, "| time", time_taken
    )


def time_fn(epoch, start_time):
    return time.time() - start_time


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []

        total_epoch_time = 0.0
        totol_time = 0.0
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            start_time = time.time()
            t0 = time.time()

            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            curr_epoch_time = time_fn(epoch, t0)
            total_epoch_time += curr_epoch_time
            losses.append(total_loss)
            totol_time += time.time() - start_time
            start_time = time.time()

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses, totol_time / 10)
                totol_time = 0.0
        print(
            f"Average time per epoch {total_epoch_time/max_epochs} (for {max_epochs} epochs)"
        )


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)

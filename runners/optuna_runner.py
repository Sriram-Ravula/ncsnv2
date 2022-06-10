import torch
import torch.optim as optim
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch


def train_mnist(config):
   use_cuda = torch.cuda.is_available()
   device = torch.device("cuda" if use_cuda else "cpu")
   train_loader, test_loader = get_data_loaders()
   model = ConvNet().to(device)

   optimizer = optim.SGD(
       model.parameters(), lr=config["lr"], momentum=config["momentum"])

   for i in range(20):
       train(model, optimizer, train_loader, device)
       acc = test(model, test_loader, device)
       tune.report(mean_accuracy=acc)


import time
start = time.time()
analysis = tune.run(
   train_mnist,
   config={
       "lr": tune.loguniform(1e-4, 1e-2),
       "momentum": tune.uniform(0.1, 0.9),
   },
   metric="mean_accuracy",
   mode="max",
   search_alg=OptunaSearch(),
   num_samples=10)
taken = time.time() - start
print(f"Time taken: {taken:.2f} seconds.")
print(f"Best config: {analysis.best_config}")
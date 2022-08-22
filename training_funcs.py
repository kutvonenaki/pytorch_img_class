from torchvision import datasets, models, transforms
import torch.nn as nn


def make_train_step(model, optimizer, loss_fn):
  def train_step(x,y):
    #make prediction
    yhat = model(x)
    #enter train mode
    model.train()
    #compute loss
    loss = loss_fn(yhat,y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    #optimizer.cleargrads()

    return loss
  return train_step



def make_model(device):

    model = models.resnet18(pretrained=True)

    #freeze all params
    for params in model.parameters():
        params.requires_grad_ = False

    #add a new final layer
    nr_filters = model.fc.in_features  #number of input features of last layer
    model.fc = nn.Linear(nr_filters, 1)

    model = model.to(device)

    return model
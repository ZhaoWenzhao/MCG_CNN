'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import torch
import torch.nn as nn


def train_xent(model, optimizer, loader, device=torch.device('cuda')):
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

##24
def train_xent_nsteps(model, optimizer, loader, device=torch.device('cuda'), accumulation_steps=1):
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        #optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        if (batch_idx+1) % accumulation_steps == 0:             # Wait for several backward steps
            optimizer.step()
            optimizer.zero_grad() ##24
##
def test_acc_nsteps(model, loader, device=torch.device('cuda'), accumulation_steps=1):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = 0 #24
            for i in range(accumulation_steps):
                output += model(data)
            output = output/accumulation_steps
            pred = output.argmax(1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()

    accuracy /= len(loader.dataset)
    return accuracy

##

def test_acc(model, loader, device=torch.device('cuda')):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()

    accuracy /= len(loader.dataset)
    return accuracy

def train_xentV2(model, optimizer, loader, device=torch.device('cuda')):
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output,f_loss = model(data)
        #print(f_loss)
        loss = criterion(output, target)
        loss = loss+f_loss*-0.1
        loss.backward()
        optimizer.step()


def test_accV2(model, loader, device=torch.device('cuda')):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output,f_loss = model(data)
            pred = output.argmax(1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()

    accuracy /= len(loader.dataset)
    return accuracy
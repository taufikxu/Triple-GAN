import torch
import numpy as np
import library.inputs as inputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_classifier(netC):
    loss_func = torch.nn.CrossEntropyLoss()
    testloader = inputs.get_data_iter_test()
    correct = 0
    total = 0
    loss_list = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = netC(images)
            _, predicted = torch.max(outputs.data, 1)
            loss_list.append(loss_func(outputs, labels).item())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total, correct, np.mean(loss_list)

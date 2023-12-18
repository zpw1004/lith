import torch
def train_model(model, criterion, optimizer, data_loader, device):
    model.train()
    for input_data, target in data_loader:
        input_data, target = input_data, target.to(device)
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, x_data, y_data, device):
    model.eval()
    with torch.no_grad():
        if device:
            x_data, y_data = x_data, y_data.to(device)
        output = model(x_data)
        _, predicted = torch.max(output, 1)
        correct = (predicted == y_data).sum().item()
        total = y_data.size(0)
        accuracy = correct / total
    return accuracy, predicted

def save_predictions(predictions, file_path):
    with open(file_path, "w") as file:
        for prediction in predictions:
            file.write(f"{prediction.item()}\n")

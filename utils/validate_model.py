import torch

def validate_model(model, data_loader, criterion):
    """
    Validate the model performance on the validation set.

    ## Args:
        - model (torch.nn.Module): The model to validate.
        - data_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        - criterion (torch.nn.Module): Loss function.

    ## Returns:
        - tuple: Validation loss and accuracy.
    """
    model.eval()
    running_loss, correct_val = 0.0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            
            if labels.ndim == 1:
                _, predicted = outputs.max(1)
                correct_val += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(data_loader)
    val_acc = correct_val / len(data_loader.dataset)

    return val_loss, val_acc

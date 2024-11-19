import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC
)

def final_validation(model, data_loader, criterion, num_classes):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []
    
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes, average="macro").to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average="macro").to(device)
    auc_metric = MulticlassAUROC(num_classes=num_classes).to(device)
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            probs = F.softmax(outputs, dim=1)
            
            all_probs.append(probs)
            all_targets.append(targets)
    
    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)
    

    accuracy = accuracy_metric(torch.argmax(all_probs, dim=1), all_targets)
    f1_score = f1_metric(torch.argmax(all_probs, dim=1), all_targets)
    precision = precision_metric(torch.argmax(all_probs, dim=1), all_targets)
    recall = recall_metric(torch.argmax(all_probs, dim=1), all_targets)
    auc = auc_metric(all_probs, all_targets)
    
    avg_loss = total_loss / len(data_loader)
    
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy.item():.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc.item():.4f}")

import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize


def final_validation(config, model, data_loader, criterion, num_classes, class_names, neptune_logger=None):
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
    all_preds = torch.argmax(all_probs, dim=1)
    
    avg_loss = total_loss / len(data_loader)
    
    metrics = {
        'avg_test_loss': avg_loss,
        'accuracy': 100 * accuracy_metric(all_preds, all_targets),
        'f1_score': 100 * f1_metric(all_preds, all_targets),
        'precision': 100 * precision_metric(all_preds, all_targets),
        'recall': 100 * recall_metric(all_preds, all_targets),
        'auc': 100 * auc_metric(all_probs, all_targets)
    }
    
    print(f"Avg Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy'].item():.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")


    plot_confusion_matrix(config, all_targets, all_preds, class_names, neptune_logger=neptune_logger)
    

    plot_roc_curves(config, all_probs, all_targets, class_names, neptune_logger=neptune_logger)

    if neptune_logger is not None:
        neptune_logger.run["final_metrics"] = {
            "avg_test_loss": avg_loss,
            "accuracy": metrics["accuracy"].item(),
            "f1_score": metrics["f1_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "auc": metrics["auc"]
        }

def plot_roc_curves(config, all_probs, all_targets, class_names, neptune_logger):

    all_targets_binarized = label_binarize(all_targets.cpu().numpy(), classes=range(len(class_names)))
    all_probs = all_probs.cpu().numpy()


    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(all_targets_binarized[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    fpr["micro"], tpr["micro"], _ = roc_curve(all_targets_binarized.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(class_names)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(class_names)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"], label=f'micro-average ROC curve (AUC: {roc_auc["micro"]:.2f})')
    plt.plot(fpr["macro"], tpr["macro"], label=f'macro-average ROC curve (AUC: {roc_auc["macro"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid()

    plt.savefig("roc_curve.png")
    if neptune_logger is not None:
        neptune_logger.run["visualizations/roc_curve"].upload("roc_curve.png")
    else:
        plt.show()

    
    
def plot_confusion_matrix(config, all_targets, all_preds, class_names, neptune_logger):
    cm = confusion_matrix(all_targets.cpu(), all_preds.cpu())
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()

    # Save and log the plot
    plt.savefig("confusion_matrix.png")
    if neptune_logger is not None:
        neptune_logger.run["visualizations/confusion_matrix"].upload("confusion_matrix.png")
    else:    
        plt.show()
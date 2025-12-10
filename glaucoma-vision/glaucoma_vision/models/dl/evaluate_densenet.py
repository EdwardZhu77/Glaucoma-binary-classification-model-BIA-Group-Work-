import os
from glaucoma_vision.utils.dl_utils import load_dl_data, load_dl_model, DEVICE
from glaucoma_vision.utils.evaluation import (calculate_metrics, collect_dl_predictions)

def evaluate_densenet(
    model_path: str,
    csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    test_loader = load_dl_data(csv_path, model_type="densenet", test_size=test_size, random_state=random_state)
    model = load_dl_model(model_path, model_type="densenet")
    y_true, y_pred, y_scores = collect_dl_predictions(model, test_loader, model_type="densenet")
    metrics = calculate_metrics(y_true, y_pred, y_scores)
    
    print("\n" + "="*50)
    print("DENSENET (IMAGE ONLY) METRICS")
    print("="*50)
    print(f"Glaucoma_Negative F1 score : {metrics['negative_f1']:.4f}")
    print(f"Glaucoma_Positive F1 score : {metrics['positive_f1']:.4f}")
    print(f"Accuracy                     : {metrics['accuracy']:.4f}")
    print(f"AUROC Score                  : {metrics['auroc']:.4f}")
    print(f"AUPRC Score                  : {metrics['auprc']:.4f}")
    
    print("\n" + "-"*50)
    print("DenseNet Confusion Matrix (TN, FP, FN, TP)")
    print("-"*50)
    tn = metrics['confusion_matrix']['TN']
    fp = metrics['confusion_matrix']['FP']
    fn = metrics['confusion_matrix']['FN']
    tp = metrics['confusion_matrix']['TP']
    print(f"                Predicted Negative  Predicted Positive")
    print(f"Actual Negative        {tn:<10}           {fp:<10}")
    print(f"Actual Positive        {fn:<10}           {tp:<10}")
    print(f"\nConfusion Matrix Values -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("="*50 + "\n")
    
    return metrics

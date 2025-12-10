import os
from glaucoma_vision.utils.dl_utils import load_dl_data, load_dl_model, DEVICE
from glaucoma_vision.utils.evaluation import (
    calculate_metrics, plot_evaluation_metrics, save_evaluation_results,
    collect_dl_predictions, plot_gradcam
)

def evaluate_densenet(
    model_path: str,
    csv_path: str,
    save_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    test_loader = load_dl_data(csv_path, model_type="densenet", test_size=test_size, random_state=random_state)
    model = load_dl_model(model_path, model_type="densenet")
    y_true, y_pred, y_scores = collect_dl_predictions(model, test_loader, model_type="densenet")
    metrics = calculate_metrics(y_true, y_pred, y_scores)

  
    # print the results
    print("\n" + "="*50)
    print("DENSENET (IMAGE ONLY) METRICS")
    print("="*50)
    print(f"Glaucoma_Negative F1 score : {metrics['negative_f1']:.4f}")
    print(f"Glaucoma_Positive F1 score : {metrics['positive_f1']:.4f}")
    print(f"Accuracy                     : {metrics['accuracy']:.4f}")
    print(f"AUROC Score                  : {metrics['auroc']:.4f}")
    print(f"AUPRC Score                  : {metrics['auprc']:.4f}")
    print("-" * 50)
    print(f"Confusion Matrix -> TP: {metrics['confusion_matrix']['TP']}, TN: {metrics['confusion_matrix']['TN']}, FP: {metrics['confusion_matrix']['FP']}, FN: {metrics['confusion_matrix']['FN']}")
    print("="*50 + "\n")
    
    # visualize
    plot_path = os.path.join(save_dir, "densenet_evaluation_plots.png")
    plot_evaluation_metrics(y_true, y_scores, y_pred, plot_path)
    
    # Grad-CAM visulazie
    target_layer = model.features.denseblock4.denselayer16.conv2
    plot_gradcam(model, test_loader, target_layer, save_dir, model_type="densenet")
    
    # save the results
    save_evaluation_results(metrics, save_dir, "densenet")
    return metrics

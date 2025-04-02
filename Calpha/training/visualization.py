import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import warnings # Import warnings module

def plot_error_distribution(hard_label, pred, threshold, save_path, epoch=None):
    """Plot distribution of TP, FP, TN, FN."""
    try:
        with torch.no_grad():
            pred = torch.sigmoid(pred)
            y_pred = (pred > threshold).float()

            tp = (y_pred * hard_label).sum().item()
            fp = (y_pred * (1 - hard_label)).sum().item()
            tn = ((1 - y_pred) * (1 - hard_label)).sum().item()
            fn = ((1 - y_pred) * hard_label).sum().item()

        plt.figure(figsize=(10, 6))
        bars = plt.bar(['TP', 'FP', 'TN', 'FN'], [tp, fp, tn, fn],
                      color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'])

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom')

        title = f'Error Distribution (Threshold={threshold:.2f})'
        if epoch is not None:
            title += f' - Epoch {epoch}'
        plt.title(title)
        plt.ylabel('Count')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error in plot_error_distribution: {e}")
        plt.close() # Ensure plot is closed on error

def plot_label_densities(hard_label, pred, threshold, save_path, epoch=None):
    """Plot density of predictions for positive and negative labels."""
    try:
        with torch.no_grad():
            pred = torch.sigmoid(pred).cpu() # Move to CPU earlier
            hard_label_cpu = hard_label.cpu() # Move labels to CPU as well
            pos_preds = pred[hard_label_cpu > 0.5].numpy().flatten()
            neg_preds = pred[hard_label_cpu <= 0.5].numpy().flatten()

        plt.figure(figsize=(10, 6))
        if pos_preds.size > 0:
            sns.kdeplot(pos_preds, color='blue', label='Positive', alpha=0.5, fill=True, bw_adjust=0.5)
        else:
            print(f"Warning plot_label_densities (Epoch {epoch}): No positive samples found.")
        if neg_preds.size > 0:
            sns.kdeplot(neg_preds, color='red', label='Negative', alpha=0.5, fill=True, bw_adjust=0.5)
        else:
             print(f"Warning plot_label_densities (Epoch {epoch}): No negative samples found.")

        plt.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.2f})')

        title = 'Prediction Densities'
        if epoch is not None:
            title += f' - Epoch {epoch}'
        plt.title(title)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error in plot_label_densities: {e}")
        plt.close() # Ensure plot is closed on error

def plot_confusion_matrix(hard_label, pred, threshold, save_path, epoch=None):
    """Plot confusion matrix."""
    try:
        with torch.no_grad():
            pred = torch.sigmoid(pred)
            y_pred = (pred > threshold).float()

            tp = (y_pred * hard_label).sum().item()
            fp = (y_pred * (1 - hard_label)).sum().item()
            tn = ((1 - y_pred) * (1 - hard_label)).sum().item()
            fn = ((1 - y_pred) * hard_label).sum().item()

        cm = np.array([[tn, fp], [fn, tp]])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                   xticklabels=['Pred 0', 'Pred 1'],
                   yticklabels=['True 0', 'True 1'])

        title = f'Confusion Matrix (Threshold={threshold:.2f})'
        if epoch is not None:
            title += f' - Epoch {epoch}'
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error in plot_confusion_matrix: {e}")
        plt.close() # Ensure plot is closed on error

def plot_metrics_vs_threshold(hard_label, pred, thresholds, save_path, epoch=None):
    """Plot precision, recall, and F1 vs threshold."""
    try:
        with torch.no_grad():
            pred = torch.sigmoid(pred)
            precisions = []
            recalls = []
            f1s = []

            for thresh in thresholds:
                y_pred = (pred > thresh).float()
                tp = (y_pred * hard_label).sum().item()
                fp = (y_pred * (1 - hard_label)).sum().item()
                fn = ((1 - y_pred) * hard_label).sum().item()

                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6)

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label='Precision', marker='.')
        plt.plot(thresholds, recalls, label='Recall', marker='.')
        plt.plot(thresholds, f1s, label='F1', marker='.')

        title = 'Metrics vs Threshold'
        if epoch is not None:
            title += f' - Epoch {epoch}'
        plt.title(title)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error in plot_metrics_vs_threshold: {e}")
        plt.close() # Ensure plot is closed on error

def plot_std_diff_vs_f1(std_diffs, f1_scores, save_path, epoch=None):
    """Plot std difference (positive - negative) vs F1 score as a box plot."""
    try:
        std_diffs = np.array(std_diffs)
        f1_scores = np.array(f1_scores)

        if std_diffs.size == 0 or f1_scores.size == 0 or std_diffs.size != f1_scores.size:
            print(f"Warning plot_std_diff_vs_f1 (Epoch {epoch}): Invalid or empty input data. Skipping plot.")
            plt.figure(figsize=(12, 6))
            plt.title(f'Std Diff vs F1 Score (No Data) - Epoch {epoch}' if epoch else 'Std Diff vs F1 Score (No Data)')
            plt.xlabel('Std Positive - Std Negative (binned)')
            plt.ylabel('F1 Score')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            return

        plt.figure(figsize=(12, 6))
        min_diff, max_diff = np.min(std_diffs), np.max(std_diffs)

        if np.isclose(min_diff, max_diff):
             bins = np.array([min_diff - 0.1, max_diff + 0.1])
        else:
            quantiles = np.linspace(0, 1, min(11, len(np.unique(std_diffs))+1))
            if len(quantiles) < 2:
                bins = np.array([min_diff - 0.1, max_diff + 0.1])
            else:
                bins = np.quantile(std_diffs, quantiles)
                bins = np.unique(bins)
        if len(bins) < 2: 
             bins = np.array([min_diff - 0.1, max_diff + 0.1])

        digitized = np.digitize(std_diffs, bins, right=False)

        boxplot_data = []
        bin_labels = []
        for i in range(1, len(bins)):
            if np.isclose(bins[i], max_diff) and i == len(bins) -1:
                 mask = (std_diffs >= bins[i-1]) & (std_diffs <= bins[i])
            else:
                 mask = (std_diffs >= bins[i-1]) & (std_diffs < bins[i])
            
            bin_f1_scores = f1_scores[mask]
            if bin_f1_scores.size > 0:
                boxplot_data.append(bin_f1_scores)
                bin_labels.append(f"{bins[i-1]:.2f} to {bins[i]:.2f}")

        if not boxplot_data:
             print(f"Warning plot_std_diff_vs_f1 (Epoch {epoch}): No data points fell into the defined bins. Plotting empty axes.")
             title = 'Std Diff (Positive - Negative) vs F1 Score (No Binned Data)'
        else:
            # Filter out potential empty arrays from boxplot_data if any edge cases were missed
            boxplot_data_filtered = [d for d in boxplot_data if len(d) > 0]
            if not boxplot_data_filtered:
                 print(f"Warning plot_std_diff_vs_f1 (Epoch {epoch}): All bins resulted in empty data arrays after filtering. Plotting empty axes.")
                 title = 'Std Diff (Positive - Negative) vs F1 Score (No Valid Binned Data)'
            else:
                 # Check if labels match filtered data length
                 if len(bin_labels) != len(boxplot_data_filtered):
                     # This case shouldn't happen with current logic, but as a safeguard:
                     print(f"Warning plot_std_diff_vs_f1 (Epoch {epoch}): Label count mismatch. Using indices as labels.")
                     plt.boxplot(boxplot_data_filtered)
                 else:
                     plt.boxplot(boxplot_data_filtered, labels=bin_labels)
                 title = 'Std Diff (Positive - Negative) vs F1 Score'

        if epoch is not None:
            title += f' - Epoch {epoch}'
        plt.title(title)
        plt.xlabel('Std Positive - Std Negative (binned)')
        plt.ylabel('F1 Score')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error in plot_std_diff_vs_f1: {e}")
        plt.close() # Ensure plot is closed on error

def plot_proportion_vs_f1(proportion_positives, f1_scores, save_path, epoch=None):
    """Plot proportion of positive samples vs F1 score as a box plot."""
    try:
        proportion_positives = np.array(proportion_positives)
        f1_scores = np.array(f1_scores)

        if proportion_positives.size == 0 or f1_scores.size == 0 or proportion_positives.size != f1_scores.size:
            print(f"Warning plot_proportion_vs_f1 (Epoch {epoch}): Invalid or empty input data. Skipping plot.")
            plt.figure(figsize=(12, 6))
            plt.title(f'Proportion Positive vs F1 Score (No Data) - Epoch {epoch}' if epoch else 'Proportion Positive vs F1 Score (No Data)')
            plt.xlabel('Proportion Positive (binned)')
            plt.ylabel('F1 Score')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            return

        plt.figure(figsize=(12, 6))
        bins = np.linspace(0, 1, 11)
        digitized = np.digitize(proportion_positives, bins, right=False)

        boxplot_data = []
        bin_labels = []

        for i in range(1, len(bins)):
            if np.isclose(bins[i], 1.0) and i == len(bins) - 1 :
                mask = (proportion_positives >= bins[i-1]) & (proportion_positives <= bins[i])
            else:
                mask = (proportion_positives >= bins[i-1]) & (proportion_positives < bins[i])
                
            bin_f1_scores = f1_scores[mask]
            if bin_f1_scores.size > 0:
                boxplot_data.append(bin_f1_scores)
                bin_labels.append(f"{bins[i-1]:.1f}-{bins[i]:.1f}")

        if not boxplot_data:
             print(f"Warning plot_proportion_vs_f1 (Epoch {epoch}): No data points fell into the defined bins. Plotting empty axes.")
             title = 'Proportion Positive vs F1 Score (No Binned Data)'
        else:
             # Filter out potential empty arrays from boxplot_data if any edge cases were missed
            boxplot_data_filtered = [d for d in boxplot_data if len(d) > 0]
            if not boxplot_data_filtered:
                 print(f"Warning plot_proportion_vs_f1 (Epoch {epoch}): All bins resulted in empty data arrays after filtering. Plotting empty axes.")
                 title = 'Proportion Positive vs F1 Score (No Valid Binned Data)'
            else:
                 # Check if labels match filtered data length
                 if len(bin_labels) != len(boxplot_data_filtered):
                     # This case shouldn't happen with current logic, but as a safeguard:
                     print(f"Warning plot_proportion_vs_f1 (Epoch {epoch}): Label count mismatch. Using indices as labels.")
                     plt.boxplot(boxplot_data_filtered)
                 else:
                    plt.boxplot(boxplot_data_filtered, labels=bin_labels)
                 title = 'Proportion Positive vs F1 Score'

        if epoch is not None:
            title += f' - Epoch {epoch}'
        plt.title(title)
        plt.xlabel('Proportion Positive (binned)')
        plt.ylabel('F1 Score')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error in plot_proportion_vs_f1: {e}")
        plt.close()


def plot_misclassified_soft_labels(fn_soft_labels, fp_soft_labels, save_path, epoch=None):
    """Plot density of soft labels for misclassified samples."""
    try:
        fn_soft_labels_np = np.array(fn_soft_labels).flatten()
        fp_soft_labels_np = np.array(fp_soft_labels).flatten()

        plt.figure(figsize=(10, 6))
        plot_legend = False
        if fn_soft_labels_np.size > 0:
            sns.kdeplot(fn_soft_labels_np, color='orange', label='False Negatives (True Label > 0.5)', alpha=0.5, fill=True, bw_adjust=0.5)
            plot_legend = True
        else:
            print(f"Warning plot_misclassified_soft_labels (Epoch {epoch}): No False Negative soft labels to plot.")

        if fp_soft_labels_np.size > 0:
            sns.kdeplot(fp_soft_labels_np, color='purple', label='False Positives (True Label <= 0.5)', alpha=0.5, fill=True, bw_adjust=0.5)
            plot_legend = True
        else:
            print(f"Warning plot_misclassified_soft_labels (Epoch {epoch}): No False Positive soft labels to plot.")

        title = 'Soft Label Density for Misclassified Voxels'
        if epoch is not None:
            title += f' - Epoch {epoch}'
        plt.title(title)
        plt.xlabel('Soft Label Value')
        plt.ylabel('Density')
        if plot_legend:
            plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error in plot_misclassified_soft_labels: {e}")
        plt.close()
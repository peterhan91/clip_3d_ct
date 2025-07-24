"""
Comprehensive Evaluation Suite for 3D CLIP Model
Advanced metrics, visualizations, and analysis tools
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, matthews_corrcoef, confusion_matrix,
    classification_report
)
from sklearn.utils import resample
from scipy import stats

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import evaluate
from train import setup_validation
from simple_tokenizer import SimpleTokenizer


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation suite for 3D CLIP model
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        output_dir: str = 'evaluation_results',
        use_amp: bool = True
    ):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp
        self.tokenizer = SimpleTokenizer()
        
        # Medical findings for evaluation
        self.medical_findings = [
            'Cardiomegaly', 'Hiatal hernia', 'Medical material',
            'Arterial wall calcification', 'Coronary artery wall calcification',
            'Pericardial effusion', 'Emphysema', 'Atelectasis', 'Lung nodule',
            'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',
            'Mosaic attenuation pattern', 'Peribronchial thickening',
            'Consolidation', 'Bronchiectasis', 'Interlobular septal thickening',
            'Lymphadenopathy'
        ]
        
        # Templates for zero-shot evaluation
        self.templates = [
            "A chest CT showing {}",
            "Chest CT findings consistent with {}",
            "CT scan demonstrating {}",
            "Radiological evidence of {}",
            "CT imaging revealing {}",
            "Thoracic CT with {}",
            "Chest computed tomography showing {}",
            "CT chest findings of {}"
        ]
    
    def tokenize(self, texts):
        """Tokenize text using SimpleTokenizer"""
        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), 77, dtype=torch.long)  # Assume context_length=77
        
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > 77:
                tokens = tokens[:77]
                tokens[76] = eot_token
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result
    
    def evaluate_comprehensive(
        self,
        test_loader: DataLoader,
        test_labels: pd.DataFrame,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation with multiple metrics and analysis
        """
        print("Starting comprehensive evaluation...")
        
        # Get model predictions
        print("Computing model predictions...")
        predictions, probabilities, ground_truth = self._get_predictions(
            test_loader, test_labels
        )
        
        # Compute basic metrics
        print("Computing basic metrics...")
        basic_metrics = self._compute_basic_metrics(
            ground_truth, predictions, probabilities
        )
        
        # Bootstrap confidence intervals
        print("Computing confidence intervals...")
        ci_metrics = self._compute_confidence_intervals(
            ground_truth, probabilities, bootstrap_samples, confidence_level
        )
        
        # Per-finding analysis
        print("Computing per-finding analysis...")
        finding_analysis = self._analyze_per_finding(
            ground_truth, predictions, probabilities
        )
        
        # Error analysis
        print("Performing error analysis...")
        error_analysis = self._analyze_errors(
            ground_truth, predictions, probabilities, test_labels
        )
        
        # Statistical tests
        print("Running statistical tests...")
        statistical_tests = self._run_statistical_tests(
            ground_truth, probabilities
        )
        
        # Calibration analysis
        print("Analyzing model calibration...")
        calibration_analysis = self._analyze_calibration(
            ground_truth, probabilities
        )
        
        # Compile comprehensive results
        comprehensive_results = {
            'basic_metrics': basic_metrics,
            'confidence_intervals': ci_metrics,
            'per_finding_analysis': finding_analysis,
            'error_analysis': error_analysis,
            'statistical_tests': statistical_tests,
            'calibration_analysis': calibration_analysis,
            'metadata': {
                'num_samples': len(ground_truth),
                'num_findings': len(self.medical_findings),
                'bootstrap_samples': bootstrap_samples,
                'confidence_level': confidence_level
            }
        }
        
        # Generate visualizations
        print("Generating visualizations...")
        self._generate_visualizations(
            ground_truth, predictions, probabilities, comprehensive_results
        )
        
        # Save results
        if save_results:
            self._save_results(comprehensive_results)
        
        # Print summary
        self._print_summary(comprehensive_results)
        
        return comprehensive_results
    
    def _get_predictions(
        self,
        test_loader: DataLoader,
        test_labels: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions and probabilities"""
        
        self.model.eval()
        all_probabilities = []
        all_ground_truth = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                images = data['img'].to(self.device)
                batch_size = images.size(0)
                
                batch_probs = []
                batch_gt = []
                
                # Get ground truth for this batch
                start_idx = batch_idx * test_loader.batch_size
                end_idx = start_idx + batch_size
                batch_labels = test_labels.iloc[start_idx:end_idx]
                
                for finding in self.medical_findings:
                    # Prepare templates
                    pos_texts = [template.format(finding.lower()) for template in self.templates]
                    neg_texts = [template.format(f"no {finding.lower()}") for template in self.templates]
                    
                    # Get embeddings
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        # Image embeddings
                        image_features = self.model.encode_image(images)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        
                        # Text embeddings
                        pos_tokens = self.tokenize(pos_texts).to(self.device)
                        neg_tokens = self.tokenize(neg_texts).to(self.device)
                        
                        pos_features = self.model.encode_text(pos_tokens)
                        neg_features = self.model.encode_text(neg_tokens)
                        
                        pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
                        neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
                        
                        # Average template embeddings
                        pos_features = pos_features.mean(dim=0, keepdim=True)
                        neg_features = neg_features.mean(dim=0, keepdim=True)
                        
                        # Compute similarities
                        pos_sim = (image_features @ pos_features.T).squeeze(-1)
                        neg_sim = (image_features @ neg_features.T).squeeze(-1)
                        
                        # Softmax probabilities
                        logits = torch.stack([neg_sim, pos_sim], dim=1)
                        probs = torch.softmax(logits, dim=1)[:, 1]  # Positive class probability
                    
                    batch_probs.append(probs.cpu().numpy())
                    
                    # Ground truth
                    if finding in batch_labels.columns:
                        gt = batch_labels[finding].values.astype(float)
                    else:
                        gt = np.zeros(batch_size)  # Default to negative if not found
                    
                    batch_gt.append(gt)
                
                # Stack findings for this batch
                batch_probs = np.stack(batch_probs, axis=1)  # (batch_size, num_findings)
                batch_gt = np.stack(batch_gt, axis=1)  # (batch_size, num_findings)
                
                all_probabilities.append(batch_probs)
                all_ground_truth.append(batch_gt)
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
        
        # Concatenate all batches
        probabilities = np.concatenate(all_probabilities, axis=0)
        ground_truth = np.concatenate(all_ground_truth, axis=0)
        
        # Convert probabilities to predictions (threshold = 0.5)
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities, ground_truth
    
    def _compute_basic_metrics(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict[str, float]:
        """Compute basic evaluation metrics"""
        
        metrics = {}
        
        # Overall metrics (micro-averaged)
        gt_flat = ground_truth.flatten()
        pred_flat = predictions.flatten()
        prob_flat = probabilities.flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(gt_flat) | np.isnan(prob_flat))
        gt_flat = gt_flat[valid_mask]
        pred_flat = pred_flat[valid_mask]
        prob_flat = prob_flat[valid_mask]
        
        metrics['overall_auc'] = roc_auc_score(gt_flat, prob_flat)
        metrics['overall_accuracy'] = accuracy_score(gt_flat, pred_flat)
        metrics['overall_f1'] = f1_score(gt_flat, pred_flat)
        metrics['overall_mcc'] = matthews_corrcoef(gt_flat, pred_flat)
        metrics['overall_ap'] = average_precision_score(gt_flat, prob_flat)
        
        # Per-finding metrics
        finding_aucs = []
        finding_accuracies = []
        finding_f1s = []
        finding_aps = []
        
        for i, finding in enumerate(self.medical_findings):
            gt_finding = ground_truth[:, i]
            pred_finding = predictions[:, i]
            prob_finding = probabilities[:, i]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(gt_finding) | np.isnan(prob_finding))
            gt_finding = gt_finding[valid_mask]
            pred_finding = pred_finding[valid_mask]
            prob_finding = prob_finding[valid_mask]
            
            if len(np.unique(gt_finding)) > 1 and len(gt_finding) > 0:
                auc = roc_auc_score(gt_finding, prob_finding)
                acc = accuracy_score(gt_finding, pred_finding)
                f1 = f1_score(gt_finding, pred_finding)
                ap = average_precision_score(gt_finding, prob_finding)
                
                finding_aucs.append(auc)
                finding_accuracies.append(acc)
                finding_f1s.append(f1)
                finding_aps.append(ap)
                
                metrics[f'{finding}_auc'] = auc
                metrics[f'{finding}_accuracy'] = acc
                metrics[f'{finding}_f1'] = f1
                metrics[f'{finding}_ap'] = ap
        
        # Mean metrics across findings
        metrics['mean_auc'] = np.mean(finding_aucs)
        metrics['mean_accuracy'] = np.mean(finding_accuracies)
        metrics['mean_f1'] = np.mean(finding_f1s)
        metrics['mean_ap'] = np.mean(finding_aps)
        
        return metrics
    
    def _compute_confidence_intervals(
        self,
        ground_truth: np.ndarray,
        probabilities: np.ndarray,
        n_bootstrap: int,
        confidence_level: float
    ) -> Dict[str, Dict[str, float]]:
        """Compute bootstrap confidence intervals"""
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_results = {}
        
        # Overall AUC confidence interval
        overall_aucs = []
        gt_flat = ground_truth.flatten()
        prob_flat = probabilities.flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(gt_flat) | np.isnan(prob_flat))
        gt_flat = gt_flat[valid_mask]
        prob_flat = prob_flat[valid_mask]
        
        n_samples = len(gt_flat)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            gt_boot = gt_flat[indices]
            prob_boot = prob_flat[indices]
            
            if len(np.unique(gt_boot)) > 1:
                auc_boot = roc_auc_score(gt_boot, prob_boot)
                overall_aucs.append(auc_boot)
        
        ci_results['overall_auc'] = {
            'mean': np.mean(overall_aucs),
            'lower': np.percentile(overall_aucs, lower_percentile),
            'upper': np.percentile(overall_aucs, upper_percentile),
            'std': np.std(overall_aucs)
        }
        
        return ci_results
    
    def _analyze_per_finding(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """Detailed per-finding analysis"""
        
        finding_analysis = {}
        
        for i, finding in enumerate(self.medical_findings):
            gt_finding = ground_truth[:, i]
            pred_finding = predictions[:, i]
            prob_finding = probabilities[:, i]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(gt_finding) | np.isnan(prob_finding))
            gt_finding = gt_finding[valid_mask]
            pred_finding = pred_finding[valid_mask]
            prob_finding = prob_finding[valid_mask]
            
            analysis = {
                'n_samples': len(gt_finding),
                'prevalence': np.mean(gt_finding) if len(gt_finding) > 0 else 0,
                'n_positive': int(np.sum(gt_finding)),
                'n_negative': int(len(gt_finding) - np.sum(gt_finding))
            }
            
            if len(np.unique(gt_finding)) > 1 and len(gt_finding) > 0:
                # ROC curve
                fpr, tpr, thresholds = roc_curve(gt_finding, prob_finding)
                analysis['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist()
                }
                
                # Optimal threshold (Youden's J statistic)
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                analysis['optimal_threshold'] = float(thresholds[optimal_idx])
                analysis['optimal_sensitivity'] = float(tpr[optimal_idx])
                analysis['optimal_specificity'] = float(1 - fpr[optimal_idx])
            
            finding_analysis[finding] = analysis
        
        return finding_analysis
    
    def _analyze_errors(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        test_labels: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze prediction errors"""
        
        error_analysis = {}
        
        # Overall error statistics
        gt_flat = ground_truth.flatten()
        pred_flat = predictions.flatten()
        
        # Remove NaN values
        valid_mask = ~np.isnan(gt_flat)
        gt_flat = gt_flat[valid_mask]
        pred_flat = pred_flat[valid_mask]
        
        # Error counts
        errors = gt_flat != pred_flat
        error_analysis['total_errors'] = int(np.sum(errors))
        error_analysis['error_rate'] = float(np.mean(errors))
        
        return error_analysis
    
    def _run_statistical_tests(
        self,
        ground_truth: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict[str, Any]:
        """Run statistical significance tests"""
        
        statistical_tests = {}
        
        # Compare AUCs between findings
        if len(self.medical_findings) > 1:
            finding_aucs = []
            finding_names = []
            
            for i, finding in enumerate(self.medical_findings):
                gt_finding = ground_truth[:, i]
                prob_finding = probabilities[:, i]
                
                # Remove NaN values
                valid_mask = ~(np.isnan(gt_finding) | np.isnan(prob_finding))
                gt_finding = gt_finding[valid_mask]
                prob_finding = prob_finding[valid_mask]
                
                if len(np.unique(gt_finding)) > 1 and len(gt_finding) > 10:
                    auc = roc_auc_score(gt_finding, prob_finding)
                    finding_aucs.append(auc)
                    finding_names.append(finding)
            
            if len(finding_aucs) > 0:
                # Best and worst performing findings
                best_idx = np.argmax(finding_aucs)
                worst_idx = np.argmin(finding_aucs)
                
                statistical_tests['best_finding'] = {
                    'name': finding_names[best_idx],
                    'auc': float(finding_aucs[best_idx])
                }
                statistical_tests['worst_finding'] = {
                    'name': finding_names[worst_idx],
                    'auc': float(finding_aucs[worst_idx])
                }
        
        return statistical_tests
    
    def _analyze_calibration(
        self,
        ground_truth: np.ndarray,
        probabilities: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """Analyze model calibration"""
        
        calibration_analysis = {}
        
        # Overall calibration
        gt_flat = ground_truth.flatten()
        prob_flat = probabilities.flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(gt_flat) | np.isnan(prob_flat))
        gt_flat = gt_flat[valid_mask]
        prob_flat = prob_flat[valid_mask]
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        total_samples = len(gt_flat)
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (prob_flat > bin_lower) & (prob_flat <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = gt_flat[in_bin].mean()
                avg_confidence_in_bin = prob_flat[in_bin].mean()
                count_in_bin = in_bin.sum()
                
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(count_in_bin)
                
                ece += (count_in_bin / total_samples) * abs(accuracy_in_bin - avg_confidence_in_bin)
            else:
                bin_accuracies.append(0)
                bin_confidences.append(bin_lower + (bin_upper - bin_lower) / 2)
                bin_counts.append(0)
        
        calibration_analysis['ece'] = float(ece)
        calibration_analysis['reliability_diagram'] = {
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts,
            'bin_boundaries': bin_boundaries.tolist()
        }
        
        return calibration_analysis
    
    def _generate_visualizations(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        results: Dict[str, Any]
    ) -> None:
        """Generate comprehensive visualizations"""
        
        # Set up plot style
        plt.style.use('default')
        
        # 1. Performance Summary
        self._plot_performance_summary(results['basic_metrics'])
        
        print(f"Visualizations saved to {self.output_dir}")
    
    def _plot_performance_summary(self, metrics: Dict[str, float]):
        """Plot performance summary"""
        
        # Extract per-finding AUCs
        finding_aucs = []
        finding_names = []
        
        for finding in self.medical_findings:
            auc_key = f'{finding}_auc'
            if auc_key in metrics:
                finding_aucs.append(metrics[auc_key])
                finding_names.append(finding)
        
        if len(finding_aucs) > 0:
            # Sort by AUC
            sorted_indices = np.argsort(finding_aucs)[::-1]
            sorted_aucs = [finding_aucs[i] for i in sorted_indices]
            sorted_names = [finding_names[i] for i in sorted_indices]
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(range(len(sorted_aucs)), sorted_aucs)
            
            # Color bars based on performance
            for i, bar in enumerate(bars):
                if sorted_aucs[i] >= 0.8:
                    bar.set_color('green')
                elif sorted_aucs[i] >= 0.7:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            ax.set_yticks(range(len(sorted_names)))
            ax.set_yticklabels(sorted_names)
            ax.set_xlabel('AUC')
            ax.set_title('Per-Finding AUC Performance')
            ax.grid(True, alpha=0.3)
            
            # Add mean AUC line
            mean_auc = metrics.get('mean_auc', np.mean(sorted_aucs))
            ax.axvline(mean_auc, color='red', linestyle='--', label=f'Mean AUC: {mean_auc:.3f}')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _save_results(self, results: Dict[str, Any]):
        """Save comprehensive results to JSON"""
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy(results)
        
        # Save to JSON
        with open(self.output_dir / 'comprehensive_results.json', 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {self.output_dir}")
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*80)
        
        # Basic metrics
        basic = results['basic_metrics']
        print(f"\nOverall Performance:")
        print(f"  Mean AUC: {basic['mean_auc']:.4f}")
        print(f"  Mean Accuracy: {basic['mean_accuracy']:.4f}")
        print(f"  Mean F1-Score: {basic['mean_f1']:.4f}")
        print(f"  Overall AUC: {basic['overall_auc']:.4f}")
        
        # Best and worst findings
        if 'statistical_tests' in results:
            stats_tests = results['statistical_tests']
            if 'best_finding' in stats_tests:
                best = stats_tests['best_finding']
                print(f"\nBest Performing Finding: {best['name']} (AUC: {best['auc']:.4f})")
            if 'worst_finding' in stats_tests:
                worst = stats_tests['worst_finding']
                print(f"Worst Performing Finding: {worst['name']} (AUC: {worst['auc']:.4f})")
        
        # Calibration
        calib = results['calibration_analysis']
        print(f"\nModel Calibration:")
        print(f"  Expected Calibration Error: {calib['ece']:.4f}")
        
        print(f"\nDetailed results and visualizations saved to: {self.output_dir}")
        print("="*80)


def run_comprehensive_evaluation(
    model_path: str,
    test_ct_path: str,
    test_labels_path: str,
    output_dir: str = 'evaluation_results',
    batch_size: int = 4,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on a trained model
    """
    
    # Load model
    print(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    # Load test data
    print(f"Loading test data from {test_ct_path}")
    from .advanced_data_loader import AdvancedCTDataset
    
    test_dataset = AdvancedCTDataset(
        img_path=test_ct_path,
        txt_path=test_labels_path,
        augment=False,
        cache_data=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load labels
    test_labels = pd.read_csv(test_labels_path)
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(
        model=model,
        device=device,
        output_dir=output_dir,
        use_amp=True
    )
    
    # Run evaluation
    results = evaluator.evaluate_comprehensive(
        test_loader=test_loader,
        test_labels=test_labels,
        bootstrap_samples=1000,
        confidence_level=0.95,
        save_results=True
    )
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive 3D CLIP Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_ct_path', type=str, required=True, help='Path to test CT volumes')
    parser.add_argument('--test_labels_path', type=str, required=True, help='Path to test labels')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    results = run_comprehensive_evaluation(
        model_path=args.model_path,
        test_ct_path=args.test_ct_path,
        test_labels_path=args.test_labels_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device
    )
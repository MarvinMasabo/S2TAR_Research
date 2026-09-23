# Copyright (c) OpenMMLab. All rights reserved.
"""
Custom dataset class for regression tasks.

Save this as: pyskl/datasets/regression_dataset.py
"""

import numpy as np
from collections import OrderedDict

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class RegressionPoseDataset(BaseDataset):
    """Pose dataset for regression tasks (e.g., energy expenditure prediction).
    
    This dataset extends BaseDataset to support regression metrics like MAE, MSE, RMSE, R2.
    
    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        split (str | None): The dataset split. Default: None.
        **kwargs: Keyword arguments for BaseDataset.
    """
    
    def evaluate(self, results, metrics='MAE', metric_options=None, logger=None):
        """Evaluate the dataset with regression metrics.
        
        Args:
            results (list): Testing results of the dataset. Each item should be
                either a prediction value (float) or a dict with 'pred' key.
            metrics (str | list[str]): Metrics to be evaluated.
                Default: 'MAE'. 
                Supported regression metrics: 'MAE', 'MSE', 'RMSE', 'R2', 'MAPE'
                Also supports classification metrics for debugging: 
                'top_k_accuracy', 'mean_class_accuracy'
            metric_options (dict): Options for evaluation metrics. Default: None.
            logger (logging.Logger | None): Logger for recording. Default: None.
            
        Returns:
            dict: Evaluation results with metric names as keys and scores as values.
            
        Examples:
            >>> results = [1.5, 2.3, 1.8, ...]  # Predictions
            >>> metrics = ['MAE', 'RMSE', 'R2']
            >>> eval_results = dataset.evaluate(results, metrics)
            >>> print(eval_results)
            {'MAE': 0.234, 'RMSE': 0.312, 'R2': 0.876}
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        
        if not len(results) == len(self):
            raise ValueError(
                f'Length of results {len(results)} != dataset length {len(self)}')
        
        if not isinstance(metrics, (list, tuple)):
            metrics = [metrics]
        
        # Get ground truth labels
        gt_labels = np.array([x['label'] for x in self.video_infos], dtype=np.float32)
        
        # Extract predictions from results
        # Handle both raw predictions and dict format
        if isinstance(results[0], dict):
            if 'pred' in results[0]:
                predictions = np.array([r['pred'] for r in results], dtype=np.float32)
            else:
                raise ValueError("Results dict must contain 'pred' key")
        else:
            predictions = np.array(results, dtype=np.float32)
        
        # Ensure predictions are 1D
        if predictions.ndim > 1:
            predictions = predictions.squeeze()
        
        if predictions.ndim != 1:
            raise ValueError(f"Predictions must be 1D after squeezing, got shape {predictions.shape}")
        
        eval_results = OrderedDict()
        
        for metric in metrics:
            metric_lower = metric.lower()
            
            # Regression metrics
            if metric_lower == 'mae':
                mae = np.mean(np.abs(predictions - gt_labels))
                eval_results['MAE'] = float(mae)
                if logger:
                    logger.info(f'MAE: {mae:.4f}')
            
            elif metric_lower == 'mse':
                mse = np.mean((predictions - gt_labels) ** 2)
                eval_results['MSE'] = float(mse)
                if logger:
                    logger.info(f'MSE: {mse:.4f}')
            
            elif metric_lower == 'rmse':
                rmse = np.sqrt(np.mean((predictions - gt_labels) ** 2))
                eval_results['RMSE'] = float(rmse)
                if logger:
                    logger.info(f'RMSE: {rmse:.4f}')
            
            elif metric_lower == 'r2':
                # R-squared (coefficient of determination)
                ss_res = np.sum((gt_labels - predictions) ** 2)
                ss_tot = np.sum((gt_labels - np.mean(gt_labels)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
                eval_results['R2'] = float(r2)
                if logger:
                    logger.info(f'R2 Score: {r2:.4f}')
            
            elif metric_lower == 'mape':
                # Mean Absolute Percentage Error
                # Avoid division by zero
                mask = gt_labels != 0
                if np.any(mask):
                    mape = np.mean(np.abs((gt_labels[mask] - predictions[mask]) / gt_labels[mask])) * 100
                    eval_results['MAPE'] = float(mape)
                    if logger:
                        logger.info(f'MAPE: {mape:.2f}%')
                else:
                    if logger:
                        logger.warning('Cannot compute MAPE: all ground truth labels are zero')
            
            # Classification metrics (for backward compatibility/debugging)
            elif metric_lower in ['top_k_accuracy', 'mean_class_accuracy', 
                                  'mean_average_precision', 'mmit_mean_average_precision']:
                # Call parent class method for classification metrics
                classification_results = super().evaluate(
                    results, metrics=metric, 
                    metric_options=metric_options, logger=logger)
                eval_results.update(classification_results)
            
            else:
                raise KeyError(
                    f'metric {metric} is not supported. '
                    f'Supported regression metrics: MAE, MSE, RMSE, R2, MAPE. '
                    f'Supported classification metrics: top_k_accuracy, mean_class_accuracy')
        
        return eval_results
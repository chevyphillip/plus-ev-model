"""Analyze and integrate Orange3 model results."""

import logging
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class OrangeModelAnalyzer:
    """Analyze Orange3 model results and integrate findings."""
    
    def __init__(
        self,
        orange_dir: str = 'data/orange',
        results_dir: str = 'data/orange/results'
    ) -> None:
        """Initialize analyzer.
        
        Args:
            orange_dir: Directory with Orange data files
            results_dir: Directory for Orange results
        """
        self.orange_dir = Path(orange_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_orange_evaluation(
        self,
        prop_type: str,
        model_name: str
    ) -> pd.DataFrame:
        """Load Orange3 model evaluation results.
        
        Args:
            prop_type: Type of prop (points, rebounds, etc.)
            model_name: Name of model (GradientBoosting, RandomForest, etc.)
            
        Returns:
            DataFrame with evaluation metrics
        """
        results_file = self.results_dir / f"{prop_type}_{model_name}_eval.tab"
        if not results_file.exists():
            raise FileNotFoundError(
                f"No evaluation results found for {prop_type} {model_name}"
            )
        
        return pd.read_csv(results_file, sep='\t')
    
    def load_feature_importance(
        self,
        prop_type: str,
        model_name: str
    ) -> pd.DataFrame:
        """Load feature importance scores from Orange3.
        
        Args:
            prop_type: Type of prop
            model_name: Name of model
            
        Returns:
            DataFrame with feature importance scores
        """
        importance_file = self.results_dir / f"{prop_type}_{model_name}_importance.tab"
        if not importance_file.exists():
            raise FileNotFoundError(
                f"No feature importance found for {prop_type} {model_name}"
            )
        
        return pd.read_csv(importance_file, sep='\t')
    
    def analyze_model_performance(
        self,
        prop_type: str,
        models: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Analyze performance of different models.
        
        Args:
            prop_type: Type of prop
            models: Optional list of model names to analyze
            
        Returns:
            Dictionary mapping models to their metrics
        """
        if models is None:
            models = ['GradientBoosting', 'RandomForest', 'NeuralNetwork']
        
        results = {}
        
        for model in models:
            try:
                eval_df = self.load_orange_evaluation(prop_type, model)
                
                metrics = {
                    'rmse': float(eval_df['RMSE'].mean()),
                    'mae': float(eval_df['MAE'].mean()),
                    'r2': float(eval_df['R2'].mean())
                }
                
                results[model] = metrics
                
            except FileNotFoundError:
                logger.warning(f"No results found for {model}")
                continue
        
        return results
    
    def get_best_features(
        self,
        prop_type: str,
        model: str,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """Get top features by importance.
        
        Args:
            prop_type: Type of prop
            model: Model name
            top_n: Number of top features to return
            
        Returns:
            List of (feature, importance) tuples
        """
        try:
            importance_df = self.load_feature_importance(prop_type, model)
            
            # Sort by importance
            importance_df = importance_df.sort_values(
                'Importance',
                ascending=False
            )
            
            # Get top features
            top_features = [
                (row['Feature'], float(row['Importance']))
                for _, row in importance_df.head(top_n).iterrows()
            ]
            
            return top_features
            
        except FileNotFoundError:
            logger.warning(f"No feature importance found for {model}")
            return []
    
    def generate_summary_report(
        self,
        prop_types: Optional[List[str]] = None,
        models: Optional[List[str]] = None
    ) -> str:
        """Generate summary report of Orange3 analysis.
        
        Args:
            prop_types: Optional list of prop types to analyze
            models: Optional list of models to analyze
            
        Returns:
            Formatted report string
        """
        if prop_types is None:
            prop_types = ['points', 'rebounds', 'assists', 'threes']
            
        if models is None:
            models = ['GradientBoosting', 'RandomForest', 'NeuralNetwork']
        
        report = []
        report.append("Orange3 Model Analysis Report")
        report.append("=" * 30)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for prop_type in prop_types:
            report.append(f"\n{prop_type.title()} Models")
            report.append("-" * 20)
            
            # Get model performance
            performance = self.analyze_model_performance(prop_type, models)
            
            for model, metrics in performance.items():
                report.append(f"\n{model}:")
                for metric, value in metrics.items():
                    report.append(f"  {metric}: {value:.3f}")
                
                # Get top features
                top_features = self.get_best_features(prop_type, model, top_n=5)
                if top_features:
                    report.append("\n  Top Features:")
                    for feature, importance in top_features:
                        report.append(f"    {feature}: {importance:.3f}")
            
            report.append("")
        
        return "\n".join(report)
    
    def export_findings(
        self,
        output_file: str = 'data/orange/findings.json'
    ) -> None:
        """Export analysis findings to JSON.
        
        Args:
            output_file: Path to save findings
        """
        findings = {
            'timestamp': datetime.now().isoformat(),
            'prop_types': {}
        }
        
        prop_types = ['points', 'rebounds', 'assists', 'threes']
        models = ['GradientBoosting', 'RandomForest', 'NeuralNetwork']
        
        for prop_type in prop_types:
            prop_findings = {
                'model_performance': self.analyze_model_performance(
                    prop_type,
                    models
                ),
                'best_features': {}
            }
            
            for model in models:
                prop_findings['best_features'][model] = self.get_best_features(
                    prop_type,
                    model
                )
            
            findings['prop_types'][prop_type] = prop_findings
        
        # Save findings
        with open(output_file, 'w') as f:
            json.dump(findings, f, indent=2)
        
        logger.info(f"Findings exported to {output_file}")

def main() -> int:
    """Analyze Orange3 results and generate report.
    
    Returns:
        0 for success, 1 for failure
    """
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        analyzer = OrangeModelAnalyzer()
        
        # Generate report
        report = analyzer.generate_summary_report()
        
        # Save report
        report_file = 'data/orange/analysis_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Analysis report saved to {report_file}")
        
        # Export findings
        analyzer.export_findings()
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

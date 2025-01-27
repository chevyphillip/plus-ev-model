"""Test script for H2O props model."""

import logging
import os
from src.models.h2o_props_model import H2OPropsModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main() -> None:
    """Train and test H2O model."""
    try:
        # Initialize model
        model = H2OPropsModel(
            prop_type='points',
            max_models=20,
            max_runtime_secs=300
        )
        
        # Train model
        metrics = model.train()
        
        # Print metrics
        print("\nModel Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")
            
        # Test prediction
        sga_id = 1628983  # Shai Gilgeous-Alexander
        prediction = model.predict_player(sga_id)
        
        print(f"\nPrediction for SGA:")
        print(f"Value: {prediction['predicted_value']:.1f}")
        print(f"Range: [{prediction['lower_bound']:.1f}, {prediction['upper_bound']:.1f}]")
        print(f"Confidence: {prediction['confidence']:.3f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Cleanup H2O
        import h2o
        h2o.cluster().shutdown()

if __name__ == "__main__":
    main()

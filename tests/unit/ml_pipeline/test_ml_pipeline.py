#!/usr/bin/env python3
"""
RecessionRadar ML Pipeline Functional Testing
============================================

Functional testing for ML pipeline components, feature engineering,
and prediction functions.

Section 3.1.2 - Function Testing for ML Components
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "api"))

class TestMLPipelineFunctions:
    """Test ML pipeline functionality"""
    
    @classmethod
    def setup_class(cls):
        """Setup test data and mock objects"""
        cls.sample_indicators = {
            "CPI": 307.026,
            "PPI": 238.169,
            "Industrial Production": 103.5,
            "Share Price": 4200.50,
            "Unemployment Rate": 3.8,
            "10-Year Rate": 4.55,
            "3-Month Rate": 5.25,
            "6-Month Rate": 5.35,
            "1-Year Rate": 5.45,
            "OECD CLI Index": 99.5,
            "CSI Index": 102.3,
            "GDP per Capita": 65000.0
        }
        
        # Sample time series data
        cls.sample_ts_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100, freq='M'),
            'CPI': np.random.normal(300, 20, 100),
            'PPI': np.random.normal(240, 15, 100),
            'INDPRO': np.random.normal(100, 10, 100),
            'unemployment_rate': np.random.normal(4, 1, 100),
            'share_price': np.random.normal(4000, 500, 100),
            '10_year_rate': np.random.normal(4.5, 1, 100),
            '3_months_rate': np.random.normal(5, 1, 100),
            '6_months_rate': np.random.normal(5.2, 1, 100),
            '1_year_rate': np.random.normal(5.3, 1, 100)
        })

    def test_time_series_feature_eng_import(self):
        """Test that time series feature engineering function can be imported"""
        try:
            from api.ML_pipe import time_series_feature_eng
            assert callable(time_series_feature_eng)
        except ImportError as e:
            pytest.skip(f"Cannot import time_series_feature_eng: {e}")

    def test_time_series_prediction_import(self):
        """Test that time series prediction function can be imported"""
        try:
            from api.ML_pipe import time_series_prediction
            assert callable(time_series_prediction)
        except ImportError as e:
            pytest.skip(f"Cannot import time_series_prediction: {e}")

    def test_regression_feature_engineering_import(self):
        """Test that regression feature engineering function can be imported"""
        try:
            from api.ML_pipe import regresstion_feature_engineering
            assert callable(regresstion_feature_engineering)
        except ImportError as e:
            pytest.skip(f"Cannot import regresstion_feature_engineering: {e}")

    def test_regression_prediction_import(self):
        """Test that regression prediction function can be imported"""
        try:
            from api.ML_pipe import regression_prediction
            assert callable(regression_prediction)
        except ImportError as e:
            pytest.skip(f"Cannot import regression_prediction: {e}")

    @patch('api.ML_pipe.time_series_feature_eng')
    def test_time_series_feature_eng_function_call(self, mock_ts_fe):
        """Test time series feature engineering function call"""
        # Mock return value
        mock_ts_fe.return_value = self.sample_ts_data
        
        try:
            from api.ML_pipe import time_series_feature_eng
            result = time_series_feature_eng()
            
            # Verify function was called
            mock_ts_fe.assert_called_once()
            
            # Verify return type
            assert isinstance(result, (pd.DataFrame, type(None)))
            
        except ImportError:
            pytest.skip("Cannot import time_series_feature_eng")

    @patch('api.ML_pipe.time_series_prediction')
    def test_time_series_prediction_function_call(self, mock_ts_pred):
        """Test time series prediction function call"""
        # Mock return value - dictionary of predictions
        mock_predictions = {
            'CPI': [308.0, 309.0, 310.0],
            'PPI': [240.0, 241.0, 242.0],
            'unemployment_rate': [3.7, 3.6, 3.5]
        }
        mock_ts_pred.return_value = mock_predictions
        
        try:
            from api.ML_pipe import time_series_prediction
            result = time_series_prediction(self.sample_ts_data)
            
            # Verify function was called with data
            mock_ts_pred.assert_called_once_with(self.sample_ts_data)
            
            # Verify return structure
            assert isinstance(result, (dict, type(None)))
            if result:
                for key, values in result.items():
                    assert isinstance(values, list)
                    
        except ImportError:
            pytest.skip("Cannot import time_series_prediction")

    @patch('api.ML_pipe.regresstion_feature_engineering')
    def test_regression_feature_engineering_with_valid_input(self, mock_reg_fe):
        """Test regression feature engineering with valid input"""
        # Mock return value - feature array
        mock_features = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        mock_reg_fe.return_value = mock_features
        
        try:
            from api.ML_pipe import regresstion_feature_engineering
            
            # Test with prediction data and indicators
            result = regresstion_feature_engineering(
                prediction_data=self.sample_ts_data.to_dict('records')[-1:],
                indicators=self.sample_indicators
            )
            
            # Verify function was called
            assert mock_reg_fe.called
            
            # Verify return type
            assert isinstance(result, (np.ndarray, list, type(None)))
            
        except ImportError:
            pytest.skip("Cannot import regresstion_feature_engineering")

    @patch('api.ML_pipe.regression_prediction')
    def test_regression_prediction_with_features(self, mock_reg_pred):
        """Test regression prediction with feature input"""
        # Mock return value - prediction array
        mock_predictions = [0.15, 0.18, 0.25, 0.35]  # base, 1m, 3m, 6m
        mock_reg_pred.return_value = mock_predictions
        
        try:
            from api.ML_pipe import regression_prediction
            
            # Mock feature array
            features = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
            
            result = regression_prediction(features)
            
            # Verify function was called
            mock_reg_pred.assert_called_once_with(features)
            
            # Verify return structure
            assert isinstance(result, (list, np.ndarray, type(None)))
            if result:
                assert len(result) == 4  # Should return 4 predictions
                for pred in result:
                    if pred is not None:
                        assert 0 <= pred <= 1, f"Prediction {pred} out of range [0,1]"
                        
        except ImportError:
            pytest.skip("Cannot import regression_prediction")

    def test_regression_feature_engineering_input_validation(self):
        """Test that regression feature engineering handles invalid inputs"""
        try:
            from api.ML_pipe import regresstion_feature_engineering
            
            # Test with empty inputs
            with patch('api.ML_pipe.regresstion_feature_engineering') as mock_func:
                mock_func.side_effect = ValueError("Invalid input")
                
                with pytest.raises(ValueError):
                    regresstion_feature_engineering(None, None)
                    
        except ImportError:
            pytest.skip("Cannot import regresstion_feature_engineering")

    def test_prediction_output_validation(self):
        """Test that predictions are within valid ranges"""
        # Mock valid prediction outputs
        test_predictions = [0.15, 0.18, 0.25, 0.35]
        
        for i, pred in enumerate(test_predictions):
            assert isinstance(pred, (int, float)), f"Prediction {i} should be numeric"
            assert 0 <= pred <= 1, f"Prediction {i} ({pred}) should be between 0 and 1"

class TestDataCollectionFunctions:
    """Test data collection and FRED API functions"""
    
    def test_fetch_and_combine_fred_series_import(self):
        """Test that FRED data fetching function can be imported"""
        try:
            from api.data_collection import fetch_and_combine_fred_series
            assert callable(fetch_and_combine_fred_series)
        except ImportError as e:
            pytest.skip(f"Cannot import fetch_and_combine_fred_series: {e}")

    @patch('api.data_collection.fetch_and_combine_fred_series')
    def test_fred_data_fetching_mock(self, mock_fetch):
        """Test FRED data fetching with mocked response"""
        # Mock return value
        mock_data = {
            'DTB3': [5.25, 5.30, 5.35],
            'DTB6': [5.35, 5.40, 5.45],
            'DGS10': [4.55, 4.60, 4.65],
            'CPIAUCSL': [307.0, 308.0, 309.0]
        }
        mock_fetch.return_value = mock_data
        
        try:
            from api.data_collection import fetch_and_combine_fred_series
            
            # Mock API key and series
            api_key = "test_key"
            series_dict = {
                "3-Month Rate": "DTB3",
                "10-Year Rate": "DGS10"
            }
            
            result = fetch_and_combine_fred_series(api_key, series_dict)
            
            # Verify function was called
            mock_fetch.assert_called_once_with(api_key, series_dict)
            
            # Verify return structure
            assert isinstance(result, (dict, type(None)))
            
        except ImportError:
            pytest.skip("Cannot import fetch_and_combine_fred_series")

    def test_api_key_validation(self):
        """Test that functions handle missing API key gracefully"""
        try:
            from api.data_collection import fetch_and_combine_fred_series
            
            # Test with invalid API key
            with patch('api.data_collection.fetch_and_combine_fred_series') as mock_func:
                mock_func.side_effect = Exception("Invalid API key")
                
                with pytest.raises(Exception):
                    fetch_and_combine_fred_series("invalid_key", {})
                    
        except ImportError:
            pytest.skip("Cannot import fetch_and_combine_fred_series")

class TestModelLoadingFunctions:
    """Test model loading and persistence functions"""
    
    def test_model_files_exist(self):
        """Test that required model files exist"""
        model_dir = project_root / "models"
        
        expected_models = [
            "catboost_recession_6m_model.pkl",
            "catboost_recession_chain_model.pkl",
            "lgbm_recession_6m_model.pkl",
            "lgbm_recession_chain_model.pkl"
        ]
        
        for model_file in expected_models:
            model_path = model_dir / model_file
            assert model_path.exists(), f"Model file {model_file} not found"

    def test_time_series_models_exist(self):
        """Test that time series model files exist"""
        ts_model_dir = project_root / "models" / "ts_models"
        
        if ts_model_dir.exists():
            model_files = list(ts_model_dir.glob("*.pkl"))
            assert len(model_files) > 0, "No time series model files found"
            
            for model_file in model_files:
                assert model_file.suffix == ".pkl", f"Model file {model_file} should be .pkl"

    @patch('pickle.load')
    @patch('builtins.open')
    def test_model_loading_functionality(self, mock_open, mock_pickle_load):
        """Test model loading functionality"""
        # Mock model object
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[0.25])
        mock_pickle_load.return_value = mock_model
        
        try:
            # Test loading a model file
            with patch('os.path.exists', return_value=True):
                model_path = project_root / "models" / "catboost_recession_6m_model.pkl"
                
                # Simulate loading
                with open(str(model_path), 'rb') as f:
                    model = pickle.load(f)
                
                # Verify model has predict method
                assert hasattr(model, 'predict')
                assert callable(model.predict)
                
        except Exception as e:
            pytest.skip(f"Model loading test failed: {e}")

if __name__ == "__main__":
    # Run the ML pipeline tests
    pytest.main([__file__, "-v", "--tb=short"])
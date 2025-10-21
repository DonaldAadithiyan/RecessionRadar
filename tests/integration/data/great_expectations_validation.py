# Great Expectations Configuration for RecessionRadar Data Validation
"""
This module provides Great Expectations integration for advanced data validation.
Install Great Expectations: pip install great-expectations
"""

import great_expectations as gx
import pandas as pd
import os
from datetime import datetime

class GreatExpectationsValidator:
    """Advanced data validation using Great Expectations framework"""
    
    def __init__(self, data_context_root="great_expectations"):
        """Initialize Great Expectations context"""
        self.context = None
        self.data_context_root = data_context_root
        
    def setup_context(self):
        """Set up Great Expectations context"""
        try:
            if not os.path.exists(self.data_context_root):
                self.context = gx.get_context(context_root_dir=self.data_context_root)
            else:
                self.context = gx.get_context(context_root_dir=self.data_context_root)
            print("✓ Great Expectations context initialized")
            return True
        except Exception as e:
            print(f"⚠️ Great Expectations setup failed: {e}")
            return False
    
    def create_expectation_suite(self, suite_name="recession_data_validation"):
        """Create expectation suite for recession data"""
        if not self.context:
            return None
            
        try:
            # Create or get existing suite
            suite = self.context.get_expectation_suite(
                expectation_suite_name=suite_name
            )
        except:
            suite = self.context.create_expectation_suite(
                expectation_suite_name=suite_name
            )
        
        return suite
    
    def define_recession_data_expectations(self, suite):
        """Define specific expectations for recession probability data"""
        expectations = [
            # Column existence expectations
            {
                "expectation_type": "expect_table_columns_to_match_ordered_list",
                "kwargs": {
                    "column_list": [
                        "date", "recession_probability", 
                        "1_month_recession_probability",
                        "3_month_recession_probability", 
                        "6_month_recession_probability"
                    ]
                }
            },
            
            # Data type expectations
            {
                "expectation_type": "expect_column_values_to_be_of_type",
                "kwargs": {
                    "column": "recession_probability",
                    "type_": "float64"
                }
            },
            
            # Range expectations for probabilities
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "recession_probability",
                    "min_value": 0,
                    "max_value": 100
                }
            },
            
            # Null value expectations
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {
                    "column": "date"
                }
            },
            
            # Uniqueness expectations
            {
                "expectation_type": "expect_column_values_to_be_unique",
                "kwargs": {
                    "column": "date"
                }
            }
        ]
        
        for expectation in expectations:
            suite.add_expectation(**expectation)
        
        return suite
    
    def validate_csv_file(self, csv_path, suite_name="recession_data_validation"):
        """Validate CSV file against expectations"""
        if not os.path.exists(csv_path):
            print(f"⚠️ File not found: {csv_path}")
            return False
            
        try:
            # Read the data
            df = pd.read_csv(csv_path)
            
            # Create batch
            batch = self.context.get_batch(
                {"path": csv_path, "reader_method": "read_csv"},
                suite_name
            )
            
            # Run validation
            results = self.context.run_validation_operator(
                "action_list_operator",
                assets_to_validate=[batch]
            )
            
            print(f"✓ Great Expectations validation completed")
            return results["success"]
            
        except Exception as e:
            print(f"⚠️ Great Expectations validation failed: {e}")
            return False

def run_great_expectations_validation():
    """Run Great Expectations validation if available"""
    print("\n=== Great Expectations Advanced Validation ===")
    
    try:
        import great_expectations as gx
        
        validator = GreatExpectationsValidator()
        
        if validator.setup_context():
            suite = validator.create_expectation_suite()
            if suite:
                validator.define_recession_data_expectations(suite)
                
                # Validate main CSV file
                csv_path = "data/recession_probability.csv"
                if os.path.exists(csv_path):
                    success = validator.validate_csv_file(csv_path)
                    if success:
                        print("✓ All Great Expectations validations passed")
                    else:
                        print("⚠️ Some Great Expectations validations failed")
                else:
                    print("⚠️ CSV file not found for validation")
        
    except ImportError:
        print("⚠️ Great Expectations not installed. Install with: pip install great-expectations")
    except Exception as e:
        print(f"⚠️ Great Expectations error: {e}")

if __name__ == "__main__":
    run_great_expectations_validation()
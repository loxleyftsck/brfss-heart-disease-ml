"""
Performance Test Suite for ADAPTA Project
Tests all modules and measures execution time.

Run this script to verify installation and functionality.
"""

import sys
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")

def timer(func):
    """Decorator to measure execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper


# ============================================================================
# TEST 1: Module Imports
# ============================================================================

def test_imports():
    """Test if all modules can be imported"""
    print_header("TEST 1: Module Imports")
    
    modules_to_test = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
    ]
    
    failed = []
    
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            print_success(f"{display_name:.<30} OK")
        except ImportError as e:
            print_error(f"{display_name:.<30} FAILED")
            failed.append(display_name)
    
    # Test custom modules
    print("\nCustom Modules:")
    custom_modules = [
        'src.config',
        'src.data_loader',
        'src.preprocessing',
        'src.models',
        'src.evaluation',
    ]
    
    for module_name in custom_modules:
        try:
            __import__(module_name)
            short_name = module_name.split('.')[-1]
            print_success(f"{short_name:.<30} OK")
        except Exception as e:
            short_name = module_name.split('.')[-1]
            print_error(f"{short_name:.<30} FAILED: {str(e)}")
            failed.append(short_name)
    
    if failed:
        print_error(f"\n{len(failed)} modules failed to import")
        return False
    else:
        print_success(f"\nAll modules imported successfully!")
        return True


# ============================================================================
# TEST 2: Configuration
# ============================================================================

def test_configuration():
    """Test configuration module"""
    print_header("TEST 2: Configuration Module")
    
    try:
        from src import config
        
        # Check critical variables
        checks = [
            ('RANDOM_SEED', config.RANDOM_SEED),
            ('TARGET_COLUMN', config.TARGET_COLUMN),
            ('SELECTED_FEATURES', config.SELECTED_FEATURES),
            ('RF_PARAMS_DEFAULT', config.RF_PARAMS_DEFAULT),
            ('LR_PARAMS_DEFAULT', config.LR_PARAMS_DEFAULT),
        ]
        
        for name, value in checks:
            if value is not None:
                print_success(f"{name:.<40} {value}")
            else:
                print_error(f"{name:.<40} NOT SET")
        
        print_success("\nConfiguration module OK!")
        return True
        
    except Exception as e:
        print_error(f"Configuration test failed: {str(e)}")
        traceback.print_exc()
        return False


# ============================================================================
# TEST 3: Data Loader (with dummy data)
# ============================================================================

@timer
def test_data_loader():
    """Test data loader with dummy data"""
    print_header("TEST 3: Data Loader Module")
    
    try:
        from src.data_loader import DataLoader
        from src import config
        
        # Create dummy data for testing
        print_info("Creating dummy dataset for testing...")
        
        n_samples = 10000
        dummy_data = pd.DataFrame({
            config.TARGET_COLUMN: np.random.randint(1, 3, n_samples),
            '_SEX': np.random.randint(1, 3, n_samples),
            '_AGEG5YR': np.random.randint(1, 14, n_samples),
            '_RFSMOK3': np.random.randint(1, 3, n_samples),
            '_RFBMI5': np.random.randint(1, 3, n_samples),
            '_TOTINDA': np.random.randint(1, 3, n_samples),
        })
        
        # Save dummy data temporarily
        temp_path = Path("data/raw/temp_test.csv")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_data.to_csv(temp_path, index=False)
        
        # Test DataLoader
        loader = DataLoader(temp_path)
        df = loader.load_raw_data()
        
        print_success(f"Loaded {len(df):,} records")
        print_success(f"Columns: {len(df.columns)}")
        
        # Test validation
        is_valid, issues = loader.validate_data_integrity(df)
        
        if is_valid:
            print_success("Data validation passed")
        else:
            print_warning(f"Validation issues: {issues}")
        
        # Cleanup
        temp_path.unlink()
        
        print_success("\nData Loader test OK!")
        return True
        
    except Exception as e:
        print_error(f"Data Loader test failed: {str(e)}")
        traceback.print_exc()
        return False


# ============================================================================
# TEST 4: Preprocessing Pipeline
# ============================================================================

@timer
def test_preprocessing():
    """Test preprocessing module"""
    print_header("TEST 4: Preprocessing Module")
    
    try:
        from src.preprocessing import DataPreprocessor
        from src import config
        
        # Create dummy data
        print_info("Creating test dataset...")
        n_samples = 5000
        
        df = pd.DataFrame({
            config.TARGET_COLUMN: np.random.randint(1, 3, n_samples),
            '_SEX': np.random.randint(1, 3, n_samples),
            '_AGEG5YR': np.random.randint(1, 14, n_samples),
            '_RFSMOK3': np.random.randint(1, 3, n_samples),
            '_RFBMI5': np.random.randint(1, 3, n_samples),
            '_TOTINDA': np.random.randint(1, 3, n_samples),
        })
        
        # Test preprocessing pipeline
        preprocessor = DataPreprocessor(random_seed=42)
        
        print_info("Running preprocessing pipeline...")
        X_train, X_test, y_train, y_test = preprocessor.full_pipeline(
            df, 
            save_processed=False
        )
        
        # Verify outputs
        print_success(f"X_train shape: {X_train.shape}")
        print_success(f"X_test shape: {X_test.shape}")
        print_success(f"y_train shape: {y_train.shape}")
        print_success(f"y_test shape: {y_test.shape}")
        
        # Verify scaling
        print_info(f"X_train mean: {X_train.mean():.6f} (should be ~0)")
        print_info(f"X_train std: {X_train.std():.6f} (should be ~1)")
        
        print_success("\nPreprocessing test OK!")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print_error(f"Preprocessing test failed: {str(e)}")
        traceback.print_exc()
        return None


# ============================================================================
# TEST 5: Model Training
# ============================================================================

@timer
def test_models(X_train, X_test, y_train, y_test):
    """Test model training"""
    print_header("TEST 5: Model Training")
    
    try:
        from src.models import RandomForestModel, LogisticRegressionModel
        
        results = {}
        
        # Test Random Forest
        print_info("Training Random Forest...")
        rf_model = RandomForestModel()
        rf_model.train(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_proba = rf_model.predict_proba(X_test)
        
        print_success("Random Forest trained successfully")
        print_info(f"  Predictions shape: {rf_pred.shape}")
        print_info(f"  Probabilities shape: {rf_proba.shape}")
        
        results['rf'] = {
            'model': rf_model,
            'predictions': rf_pred,
            'probabilities': rf_proba
        }
        
        # Test Logistic Regression
        print_info("\nTraining Logistic Regression...")
        lr_model = LogisticRegressionModel()
        lr_model.train(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_proba = lr_model.predict_proba(X_test)
        
        print_success("Logistic Regression trained successfully")
        print_info(f"  Predictions shape: {lr_pred.shape}")
        print_info(f"  Probabilities shape: {lr_proba.shape}")
        
        results['lr'] = {
            'model': lr_model,
            'predictions': lr_pred,
            'probabilities': lr_proba
        }
        
        print_success("\nModel training test OK!")
        return results
        
    except Exception as e:
        print_error(f"Model training test failed: {str(e)}")
        traceback.print_exc()
        return None


# ============================================================================
# TEST 6: Model Evaluation
# ============================================================================

@timer
def test_evaluation(y_test, results):
    """Test evaluation module"""
    print_header("TEST 6: Model Evaluation")
    
    try:
        from src.evaluation import ModelEvaluator
        
        evaluation_results = {}
        
        # Evaluate Random Forest
        print_info("Evaluating Random Forest...")
        rf_evaluator = ModelEvaluator("RandomForest_Test")
        rf_metrics = rf_evaluator.calculate_metrics(
            y_test,
            results['rf']['predictions'],
            results['rf']['probabilities']
        )
        rf_evaluator.print_metrics(rf_metrics)
        evaluation_results['rf'] = rf_metrics
        
        # Evaluate Logistic Regression
        print_info("\nEvaluating Logistic Regression...")
        lr_evaluator = ModelEvaluator("LogisticRegression_Test")
        lr_metrics = lr_evaluator.calculate_metrics(
            y_test,
            results['lr']['predictions'],
            results['lr']['probabilities']
        )
        lr_evaluator.print_metrics(lr_metrics)
        evaluation_results['lr'] = lr_metrics
        
        print_success("\nEvaluation test OK!")
        return evaluation_results
        
    except Exception as e:
        print_error(f"Evaluation test failed: {str(e)}")
        traceback.print_exc()
        return None


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run complete test suite"""
    print_header("ADAPTA PROJECT - PERFORMANCE TEST SUITE")
    print(f"{Colors.BOLD}Testing all modules and measuring performance...{Colors.RESET}\n")
    
    start_time = time.time()
    test_results = {}
    
    # Test 1: Imports
    test_results['imports'] = test_imports()
    
    if not test_results['imports']:
        print_error("\n❌ Import test failed! Please install dependencies:")
        print_info("   pip install -r requirements.txt")
        return
    
    # Test 2: Configuration
    test_results['config'] = test_configuration()
    
    # Test 3: Data Loader
    result, elapsed = test_data_loader()
    test_results['data_loader'] = result
    print_info(f"⏱  Execution time: {elapsed:.3f}s")
    
    # Test 4: Preprocessing
    result, elapsed = test_preprocessing()
    test_results['preprocessing'] = result is not None
    print_info(f"⏱  Execution time: {elapsed:.3f}s")
    
    if result is None:
        print_error("\n❌ Preprocessing failed, skipping remaining tests")
        return
    
    X_train, X_test, y_train, y_test = result
    
    # Test 5: Model Training
    result, elapsed = test_models(X_train, X_test, y_train, y_test)
    test_results['models'] = result is not None
    print_info(f"⏱  Execution time: {elapsed:.3f}s")
    
    if result is None:
        print_error("\n❌ Model training failed, skipping evaluation")
        return
    
    # Test 6: Evaluation
    eval_result, elapsed = test_evaluation(y_test, result)
    test_results['evaluation'] = eval_result is not None
    print_info(f"⏱  Execution time: {elapsed:.3f}s")
    
    # Final Summary
    total_time = time.time() - start_time
    
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in test_results.values() if v)
    total = len(test_results)
    
    for test_name, passed_status in test_results.items():
        if passed_status:
            print_success(f"{test_name:.<30} PASSED")
        else:
            print_error(f"{test_name:.<30} FAILED")
    
    print(f"\n{Colors.BOLD}Results:{Colors.RESET}")
    print(f"  {Colors.GREEN}Passed: {passed}/{total}{Colors.RESET}")
    print(f"  {Colors.BLUE}Total time: {total_time:.2f}s{Colors.RESET}")
    
    if passed == total:
        print(f"\n{Colors.BOLD}{Colors.GREEN}✓ ALL TESTS PASSED! 🎉{Colors.RESET}")
        print(f"{Colors.GREEN}ADAPTA project is fully functional and ready to use!{Colors.RESET}\n")
    else:
        print(f"\n{Colors.BOLD}{Colors.RED}✗ Some tests failed{Colors.RESET}")
        print(f"{Colors.YELLOW}Please check the error messages above{Colors.RESET}\n")


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Tests interrupted by user{Colors.RESET}\n")
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {str(e)}{Colors.RESET}")
        traceback.print_exc()

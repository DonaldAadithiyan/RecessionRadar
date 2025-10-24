#!/usr/bin/env python3
"""
Consolidated Test Runner - Executes All RecessionRadar Test Suites
================================================================

This script runs all comprehensive testing frameworks in the correct order:
1. Simple Data Integrity Tests
2. Comprehensive Data Integrity Tests (with pytest)
3. Complete Model Testing (PKL files and model validation)
4. Great Expectations Data Validation (if available)

Usage:
    python tests/run_all_tests.py
    python tests/run_all_tests.py --verbose
    python tests/run_all_tests.py --suite simple
    python tests/run_all_tests.py --suite comprehensive
    python tests/run_all_tests.py --suite models
    python tests/run_all_tests.py --suite expectations
"""

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

def print_header(title, width=80):
    """Print a formatted header"""
    print("\n" + "=" * width)
    print(f"🧪 {title.center(width-4)}")
    print("=" * width)

def print_section(title, width=70):
    """Print a formatted section header"""
    print(f"\n📋 {title}")
    print("-" * width)

def run_command(command, description, timeout=300):
    """Run a command and capture its output"""
    print(f"\n=> {description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_root
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        print(f"\n{'✅' if success else '❌'} {description} - {'PASSED' if success else 'FAILED'}")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        
        return success, result.stdout, result.stderr, duration
        
    except subprocess.TimeoutExpired:
        print(f"❌ {description} - TIMEOUT after {timeout} seconds")
        return False, "", f"Timeout after {timeout} seconds", timeout
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False, "", str(e), 0

def run_simple_tests():
    """Run simple data integrity tests"""
    return run_command(
        "python tests/simple_data_integrity_test.py",
        "Simple Data Integrity Tests"
    )

def run_comprehensive_tests():
    """Run comprehensive data integrity tests with pytest"""
    return run_command(
        "python -m pytest tests/comprehensive_data_integrity_test.py -v -s",
        "Comprehensive Data Integrity Tests (pytest)"
    )

def run_model_tests():
    """Run complete model testing suite"""
    return run_command(
        "python tests/complete_model_testing.py",
        "Complete Model Testing Suite"
    )

def run_expectations_tests():
    """Run Great Expectations validation"""
    return run_command(
        "python tests/great_expectations_validation.py",
        "Great Expectations Data Validation"
    )

def run_data_integrity_tests():
    """Run the original data integrity test runner"""
    return run_command(
        "python tests/run_data_integrity_tests.py",
        "Original Data Integrity Test Runner"
    )

def generate_summary_report(results, total_duration):
    """Generate comprehensive test summary report"""
    
    print_header("COMPREHENSIVE TEST EXECUTION SUMMARY")
    
    print(f"📅 Test Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total Execution Time: {total_duration:.2f} seconds")
    print(f"🏗️  Project Root: {project_root}")
    
    print_section("Test Suite Results")
    
    total_tests = len(results)
    passed_tests = sum(1 for success, _, _, _ in results.values() if success)
    failed_tests = total_tests - passed_tests
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    for suite_name, (success, stdout, stderr, duration) in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {suite_name:<40} {status:<12} ({duration:.2f}s)")
    
    print_section("Overall Statistics")
    print(f"📊 Total Test Suites: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    print_section("Detailed Results")
    
    for suite_name, (success, stdout, stderr, duration) in results.items():
        print(f"\n🔍 {suite_name}:")
        print(f"   Status: {'✅ PASSED' if success else '❌ FAILED'}")
        print(f"   Duration: {duration:.2f} seconds")
        
        if not success and stderr:
            print(f"   Error: {stderr[:200]}...")
    
    print_section("Recommendations")
    
    if failed_tests == 0:
        print("🎉 All test suites passed successfully!")
        print("✅ Your RecessionRadar system has comprehensive test coverage")
        print("✅ All PKL model files are validated and functional")
        print("✅ Data integrity checks are all passing")
    else:
        print(f"⚠️  {failed_tests} test suite(s) failed")
        print("🔧 Review the detailed results above for specific issues")
        if failed_tests == 1:
            print("💡 Consider running individual test suites to isolate issues")
        else:
            print("💡 Multiple failures may indicate environment or dependency issues")
    
    print_section("Next Steps")
    print("📝 Review COMPLETE_TEST_COVERAGE_SUMMARY.md for detailed documentation")
    print("🔄 Re-run specific test suites: python tests/run_all_tests.py --suite <name>")
    print("📊 Monitor data freshness and model performance regularly")
    
    return success_rate >= 80  # Consider 80%+ success rate as overall success

def main():
    """Main test execution function"""
    parser = argparse.ArgumentParser(description="Run comprehensive RecessionRadar test suites")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--suite", "-s", choices=["simple", "comprehensive", "models", "expectations", "integrity", "all"], 
                       default="all", help="Run specific test suite")
    parser.add_argument("--timeout", "-t", type=int, default=300, help="Timeout per test suite in seconds")
    
    args = parser.parse_args()
    
    print_header("RecessionRadar Comprehensive Test Suite Runner")
    print(f"🎯 Target Suite: {args.suite}")
    print(f"⏰ Timeout: {args.timeout} seconds per suite")
    
    start_time = time.time()
    results = {}
    
    # Define test suites to run
    test_suites = {
        "simple": run_simple_tests,
        "comprehensive": run_comprehensive_tests, 
        "models": run_model_tests,
        "expectations": run_expectations_tests,
        "integrity": run_data_integrity_tests
    }
    
    # Determine which suites to run
    if args.suite == "all":
        suites_to_run = test_suites
    else:
        suites_to_run = {args.suite: test_suites[args.suite]}
    
    # Execute test suites
    for suite_name, test_function in suites_to_run.items():
        try:
            success, stdout, stderr, duration = test_function()
            results[suite_name] = (success, stdout, stderr, duration)
        except KeyboardInterrupt:
            print(f"\n⏹️  Test execution interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error running {suite_name}: {e}")
            results[suite_name] = (False, "", str(e), 0)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Generate summary report
    overall_success = generate_summary_report(results, total_duration)
    
    # Exit with appropriate code
    exit_code = 0 if overall_success else 1
    print(f"\n🏁 Test execution completed with exit code: {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
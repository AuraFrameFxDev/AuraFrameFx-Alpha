#!/usr/bin/env python3
"""
Comprehensive test runner for test_conftest.py test suite.

This script runs all tests for the test_conftest.py module,
including unit tests, performance tests, and validation tests.

Usage:
    python run_test_conftest_suite.py [options]

Testing Framework: pytest with pytest-asyncio
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path


def run_tests(verbose=False, coverage=False, test_pattern=None, parallel=False):
    """Run the complete test suite."""
    
    # Build pytest command
    cmd = [sys.executable, '-m', 'pytest']
    
    # Add verbosity
    if verbose:
        cmd.extend(['-v', '--tb=short'])
    else:
        cmd.append('-q')
    
    # Add coverage
    if coverage:
        cmd.extend([
            '--cov=test_conftest',
            '--cov-report=term-missing',
            '--cov-report=html:htmlcov',
            '--cov-fail-under=80'
        ])
    
    # Add parallel execution
    if parallel:
        try:
            import pytest_xdist
            cmd.extend(['-n', 'auto'])
        except ImportError:
            print("Warning: pytest-xdist not available, running sequentially")
    
    # Add test pattern filtering
    if test_pattern:
        cmd.extend(['-k', test_pattern])
    
    # Add test files
    test_files = [
        'test_test_conftest.py',
        'test_conftest_performance.py'
    ]
    
    existing_files = []
    for test_file in test_files:
        if Path(test_file).exists():
            existing_files.append(test_file)
    
    if not existing_files:
        print("No test files found!")
        return 1
    
    cmd.extend(existing_files)
    
    # Add additional options
    cmd.extend([
        '--strict-markers',
        '--durations=10',
        '--maxfail=5',
        '--tb=short'
    ])
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    # Execute tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive test suite for test_conftest.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_test_conftest_suite.py                    # Run all tests
  python run_test_conftest_suite.py -v                 # Verbose output
  python run_test_conftest_suite.py -c                 # With coverage
  python run_test_conftest_suite.py -k "test_fixture"  # Run fixture tests only
  python run_test_conftest_suite.py --unit-only        # Unit tests only
  python run_test_conftest_suite.py --performance      # Performance tests only
        """
    )
    
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '-c', '--coverage', 
        action='store_true',
        help='Generate coverage report'
    )
    
    parser.add_argument(
        '-k', '--keyword',
        help='Run tests matching keyword expression'
    )
    
    parser.add_argument(
        '--unit-only', 
        action='store_true',
        help='Run only unit tests (fast)'
    )
    
    parser.add_argument(
        '--performance', 
        action='store_true',
        help='Run performance tests only'
    )
    
    parser.add_argument(
        '--parallel', 
        action='store_true',
        help='Run tests in parallel (requires pytest-xdist)'
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Quick test run (exclude slow tests)'
    )
    
    args = parser.parse_args()
    
    # Determine test pattern
    test_pattern = args.keyword
    
    if args.unit_only:
        test_pattern = 'not slow and not performance'
    elif args.performance:
        test_pattern = 'performance or slow'
    elif args.quick:
        test_pattern = 'not slow'
    
    # Run tests
    print("🧪 Running test_conftest.py Test Suite")
    print("=" * 60)
    
    exit_code = run_tests(
        verbose=args.verbose,
        coverage=args.coverage,
        test_pattern=test_pattern,
        parallel=args.parallel
    )
    
    print("-" * 60)
    if exit_code == 0:
        print("✅ All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/")
    else:
        print(f"❌ Tests failed with exit code {exit_code}")
        print("💡 Try running with -v for more details")
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
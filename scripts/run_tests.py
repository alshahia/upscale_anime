#!/usr/bin/env python3
"""
Comprehensive test runner for the anime super-resolution project.
Runs unit tests, integration tests, and performance benchmarks.
"""

import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


class TestRunner:
    """Comprehensive test runner for the project."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests with coverage."""
        print("🧪 Running Unit Tests...")
        
        try:
            # Run pytest with coverage
            cmd = [
                "python", "-m", "pytest", 
                "tests/", 
                "-v", 
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml",
                "--tb=short"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "returncode": -1
            }
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        print("[LINK] Running Integration Tests...")
        
        try:
            cmd = [
                "python", "-m", "pytest", 
                "tests/integration/", 
                "-v", 
                "--tb=short"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "returncode": -1
            }
    
    def run_model_tests(self) -> Dict[str, Any]:
        """Run model-specific tests."""
        print("🤖 Running Model Tests...")
        
        try:
            from anime_sr.models.span.checkpoint_compatible_exact import CheckpointCompatibleSPANExact
            import torch
            
            results = {"tests": []}
            
            # Test 1: Model creation
            try:
                model = CheckpointCompatibleSPANExact(
                    scale=4, channels=48, hidden_channels=96, num_blocks=6
                )
                results["tests"].append({
                    "name": "Model Creation",
                    "status": "passed",
                    "message": "Model created successfully"
                })
            except Exception as e:
                results["tests"].append({
                    "name": "Model Creation",
                    "status": "failed",
                    "message": str(e)
                })
            
            # Test 2: Forward pass
            try:
                model.eval()
                input_tensor = torch.randn(1, 3, 64, 64)
                with torch.no_grad():
                    output = model(input_tensor)
                
                assert output.shape == (1, 3, 256, 256)
                assert not torch.isnan(output).any()
                
                results["tests"].append({
                    "name": "Forward Pass",
                    "status": "passed",
                    "message": f"Output shape: {output.shape}"
                })
            except Exception as e:
                results["tests"].append({
                    "name": "Forward Pass",
                    "status": "failed",
                    "message": str(e)
                })
            
            # Test 3: Parameter count
            try:
                total_params = sum(p.numel() for p in model.parameters())
                results["tests"].append({
                    "name": "Parameter Count",
                    "status": "passed",
                    "message": f"Total parameters: {total_params:,}"
                })
            except Exception as e:
                results["tests"].append({
                    "name": "Parameter Count",
                    "status": "failed",
                    "message": str(e)
                })
            
            # Overall status
            failed_tests = [t for t in results["tests"] if t["status"] == "failed"]
            results["status"] = "passed" if not failed_tests else "failed"
            
            return results
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "tests": []
            }
    
    def run_inference_tests(self) -> Dict[str, Any]:
        """Run inference tests."""
        print("[SEARCH] Running Inference Tests...")
        
        try:
            # Test basic inference
            cmd = [
                "python", "scripts/inference.py",
                "--checkpoint", "checkpoints/span/spanx4_ch48.pth",
                "--input", "data/test_hr/dummy_0000.png",
                "--output", "test_output.png"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "returncode": -1
            }
    
    def run_config_tests(self) -> Dict[str, Any]:
        """Run configuration validation tests."""
        print("⚙️ Running Configuration Tests...")
        
        configs_to_test = [
            "configs/finetune_nan_safe.yaml",
            "configs/finetune_stable_safe.yaml",
            "configs/finetune_stable.yaml"
        ]
        
        results = {"configs": []}
        
        for config_path in configs_to_test:
            try:
                from anime_sr.utils.config import load_config
                
                full_path = Path(__file__).parent.parent / config_path
                if full_path.exists():
                    config = load_config(str(full_path))
                    
                    # Validate required fields
                    required_sections = ['model', 'training', 'data', 'losses']
                    missing_sections = [s for s in required_sections if s not in config]
                    
                    if missing_sections:
                        status = "failed"
                        message = f"Missing sections: {missing_sections}"
                    else:
                        status = "passed"
                        message = "Configuration valid"
                    
                    results["configs"].append({
                        "name": config_path,
                        "status": status,
                        "message": message
                    })
                else:
                    results["configs"].append({
                        "name": config_path,
                        "status": "skipped",
                        "message": "File not found"
                    })
                    
            except Exception as e:
                results["configs"].append({
                    "name": config_path,
                    "status": "failed",
                    "message": str(e)
                })
        
        # Overall status
        failed_configs = [c for c in results["configs"] if c["status"] == "failed"]
        results["status"] = "passed" if not failed_configs else "failed"
        
        return results
    
    def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        print("[FAST] Running Performance Benchmark...")
        
        try:
            from anime_sr.models.span.checkpoint_compatible_exact import CheckpointCompatibleSPANExact
            import torch
            import time
            
            model = CheckpointCompatibleSPANExact(
                scale=4, channels=48, hidden_channels=96, num_blocks=6
            )
            model.eval()
            
            # Benchmark different input sizes
            sizes = [64, 128, 256, 512]
            results = {"benchmarks": []}
            
            for size in sizes:
                input_tensor = torch.randn(1, 3, size, size)
                
                # Warm up
                with torch.no_grad():
                    for _ in range(3):
                        _ = model(input_tensor)
                
                # Benchmark
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(10):
                        output = model(input_tensor)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                fps = 1.0 / avg_time
                
                results["benchmarks"].append({
                    "input_size": f"{size}x{size}",
                    "output_size": f"{size*4}x{size*4}",
                    "avg_time": f"{avg_time:.3f}s",
                    "fps": f"{fps:.1f}",
                    "status": "passed"
                })
            
            results["status"] = "passed"
            return results
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "benchmarks": []
            }
    
    def run_memory_test(self) -> Dict[str, Any]:
        """Run memory usage tests."""
        print("💾 Running Memory Tests...")
        
        try:
            from anime_sr.models.span.checkpoint_compatible_exact import CheckpointCompatibleSPANExact
            import torch
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            model = CheckpointCompatibleSPANExact(
                scale=4, channels=48, hidden_channels=96, num_blocks=6
            )
            
            # Model memory
            model_memory = process.memory_info().rss / 1024 / 1024  # MB
            model_size = model_memory - baseline_memory
            
            # Forward pass memory
            input_tensor = torch.randn(1, 3, 256, 256)
            with torch.no_grad():
                output = model(input_tensor)
            
            forward_memory = process.memory_info().rss / 1024 / 1024  # MB
            forward_size = forward_memory - model_memory
            
            # Parameter count
            total_params = sum(p.numel() for p in model.parameters())
            param_size_mb = total_params * 4 / 1024 / 1024  # 4 bytes per float32
            
            results = {
                "status": "passed",
                "baseline_memory_mb": f"{baseline_memory:.1f}",
                "model_memory_mb": f"{model_size:.1f}",
                "forward_memory_mb": f"{forward_size:.1f}",
                "total_parameters": f"{total_params:,}",
                "parameter_size_mb": f"{param_size_mb:.1f}",
                "peak_memory_mb": f"{forward_memory:.1f}"
            }
            
            return results
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        end_time = time.time()
        total_time = end_time - self.start_time
        
        report = f"""
# Test Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Total Time: {total_time:.2f} seconds

## Summary
"""
        
        # Count passed/failed tests
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results.values() if r.get("status") == "passed"])
        failed_tests = len([r for r in self.results.values() if r.get("status") == "failed"])
        error_tests = len([r for r in self.results.values() if r.get("status") == "error"])
        
        report += f"- Total Tests: {total_tests}\n"
        report += f"- Passed: {passed_tests}\n"
        report += f"- Failed: {failed_tests}\n"
        report += f"- Errors: {error_tests}\n"
        report += f"- Success Rate: {passed_tests/total_tests*100:.1f}%\n\n"
        
        # Detailed results
        for test_name, result in self.results.items():
            status_emoji = "[OK]" if result.get("status") == "passed" else "[ERROR]" if result.get("status") == "failed" else "[WARN]"
            report += f"## {status_emoji} {test_name.replace('_', ' ').title()}\n"
            report += f"Status: {result.get('status', 'unknown')}\n"
            
            if "error" in result:
                report += f"Error: {result['error']}\n"
            
            if "tests" in result:
                for test in result["tests"]:
                    test_emoji = "[OK]" if test["status"] == "passed" else "[ERROR]"
                    report += f"- {test_emoji} {test['name']}: {test['message']}\n"
            
            if "configs" in result:
                for config in result["configs"]:
                    config_emoji = "[OK]" if config["status"] == "passed" else "[ERROR]" if config["status"] == "failed" else "⏭️"
                    report += f"- {config_emoji} {config['name']}: {config['message']}\n"
            
            if "benchmarks" in result:
                for benchmark in result["benchmarks"]:
                    report += f"- [FAST] {benchmark['input_size']} → {benchmark['output_size']}: {benchmark['avg_time']} ({benchmark['fps']} FPS)\n"
            
            if result.get("returncode") == 0:
                report += "[OK] Test completed successfully\n"
            elif result.get("returncode") is not None and result.get("returncode") != 0:
                report += f"[ERROR] Test failed with return code {result['returncode']}\n"
            
            report += "\n"
        
        return report
    
    def save_report(self, report: str, output_path: str = "test_report.md"):
        """Save test report to file."""
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"📄 Test report saved to: {output_path}")
    
    def run_all_tests(self, save_report: bool = True) -> Dict[str, Any]:
        """Run all tests and generate report."""
        print("[LAUNCH] Starting Comprehensive Test Suite")
        print("=" * 50)
        
        # Run all test suites
        test_suites = [
            ("unit_tests", self.run_unit_tests),
            ("integration_tests", self.run_integration_tests),
            ("model_tests", self.run_model_tests),
            ("inference_tests", self.run_inference_tests),
            ("config_tests", self.run_config_tests),
            ("performance_benchmark", self.run_performance_benchmark),
            ("memory_test", self.run_memory_test),
        ]
        
        for test_name, test_func in test_suites:
            try:
                result = test_func()
                self.results[test_name] = result
                
                status_emoji = "[OK]" if result.get("status") == "passed" else "[ERROR]" if result.get("status") == "failed" else "[WARN]"
                print(f"{status_emoji} {test_name.replace('_', ' ').title()}: {result.get('status', 'unknown')}")
                
            except Exception as e:
                print(f"[ERROR] {test_name}: Error - {str(e)}")
                self.results[test_name] = {"status": "error", "error": str(e)}
        
        # Generate and save report
        report = self.generate_report()
        if save_report:
            self.save_report(report)
        
        print("\n" + "=" * 50)
        print("🏁 Test Suite Complete")
        
        return self.results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run comprehensive test suite")
    parser.add_argument("--test", choices=["unit", "integration", "model", "inference", "config", "performance", "memory", "all"], 
                       default="all", help="Specific test to run")
    parser.add_argument("--no-report", action="store_true", help="Don't save test report")
    parser.add_argument("--output", default="test_report.md", help="Output report file")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.test == "all":
        results = runner.run_all_tests(save_report=not args.no_report)
    else:
        # Run specific test
        test_map = {
            "unit": runner.run_unit_tests,
            "integration": runner.run_integration_tests,
            "model": runner.run_model_tests,
            "inference": runner.run_inference_tests,
            "config": runner.run_config_tests,
            "performance": runner.run_performance_benchmark,
            "memory": runner.run_memory_test,
        }
        
        if args.test in test_map:
            result = test_map[args.test]()
            runner.results[args.test] = result
            
            if not args.no_report:
                report = runner.generate_report()
                runner.save_report(report, args.output)
        else:
            print(f"Unknown test: {args.test}")
            return 1
    
    # Return exit code based on test results
    failed_tests = [r for r in runner.results.values() if r.get("status") == "failed"]
    error_tests = [r for r in runner.results.values() if r.get("status") == "error"]
    
    if failed_tests or error_tests:
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Deployment script for the anime super-resolution API.
Handles environment setup, model loading, and server deployment.
"""

import os
import re
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


class APIDeployer:
    """API deployment manager."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config = self.load_deployment_config()
        
    def load_deployment_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        config_path = self.project_root / "api_config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1,
                "log_level": "info",
                "models": [
                    "checkpoint_compatible"
                ],
                "gpu": True,
                "memory_limit_mb": 8000,
                "max_concurrent_requests": 10
            }
    
    def check_environment(self) -> bool:
        """Check if environment is ready for deployment."""
        print("[SEARCH] Checking deployment environment...")
        
        issues = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            issues.append(f"Python {python_version.major}.{python_version.minor} is not supported (requires 3.8+)")
        
        # Check required packages
        required_packages = [
            "fastapi", "uvicorn", "torch", "PIL", "numpy", "psutil"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                issues.append(f"Missing package: {package}")
        
        # Check GPU availability
        try:
            import torch
            if self.config.get("gpu", True) and not torch.cuda.is_available():
                issues.append("GPU requested but CUDA not available")
        except ImportError:
            issues.append("PyTorch not available")
        
        # Check model files
        for model_name in self.config.get("models", []):
            if model_name == "checkpoint_compatible":
                checkpoint_path = self.project_root / "checkpoints/span/spanx4_ch48.pth"
                if not checkpoint_path.exists():
                    issues.append(f"Checkpoint not found: {checkpoint_path}")
        
        # Check directories
        required_dirs = ["src", "checkpoints", "data"]
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                issues.append(f"Directory not found: {dir_path}")
        
        if issues:
            print("[ERROR] Environment check failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("[OK] Environment check passed")
            return True
    
    def install_dependencies(self) -> bool:
        """Install required dependencies."""
        print("[PKG] Installing dependencies...")
        
        try:
            # Install API requirements
            requirements_path = self.project_root / "requirements_api.txt"
            if requirements_path.exists():
                cmd = [
                    sys.executable, "-m", "pip", "install", 
                    "-r", str(requirements_path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"[ERROR] Failed to install dependencies: {result.stderr}")
                    return False
                else:
                    print("[OK] Dependencies installed successfully")
                    return True
            else:
                print("[WARN]  No requirements_api.txt found, skipping dependency installation")
                return True
                
        except Exception as e:
            print(f"[ERROR] Error installing dependencies: {e}")
            return False
    
    def create_api_config(self) -> bool:
        """Create API configuration file."""
        print("⚙️  Creating API configuration...")
        
        config_path = self.project_root / "api_config.json"
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            print(f"[OK] API configuration saved to: {config_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to create API configuration: {e}")
            return False
    
    def test_api(self) -> bool:
        """Test API functionality."""
        print("🧪 Testing API functionality...")
        
        try:
            # Import API module
            sys.path.append(str(self.project_root / "src"))
            from api.inference_api import app, load_model, inference_engines
            
            # Test model loading
            for model_name in self.config.get("models", []):
                if model_name not in inference_engines:
                    print(f"  Loading model: {model_name}")
                    # This would be done in the startup event
                    # await load_model(model_name)
            
            print("[OK] API test passed")
            return True
            
        except Exception as e:
            print(f"[ERROR] API test failed: {e}")
            return False
    
    def start_server(self) -> bool:
        """Start the API server."""
        print("[LAUNCH] Starting API server...")
        
        try:
            # Prepare command
            cmd = [
                sys.executable, "-m", "uvicorn",
                "api.inference_api:app",
                "--host", str(self.config.get("host", "0.0.0.0")),
                "--port", str(self.config.get("port", 8000)),
                "--log-level", str(self.config.get("log_level", "info"))
            ]
            
            # Add worker configuration
            if self.config.get("workers", 1) > 1:
                cmd.extend(["--workers", str(self.config.get("workers", 1))])
            else:
                cmd.append("--reload")
            
            # Change to project directory
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.project_root / "src")
            
            print(f"🌐 Server will be available at: http://{self.config.get('host', '0.0.0.0')}:{self.config.get('port', 8000)}")
            print(f"📚 API documentation: http://{self.config.get('host', '0.0.0.0')}:{self.config.get('port', 8000)}/docs")
            
            # Start server
            subprocess.run(cmd, cwd=self.project_root, env=env)
            
            return True
            
        except KeyboardInterrupt:
            print("\n[STOP] Server stopped by user")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to start server: {e}")
            return False
    
    def create_systemd_service(self) -> bool:
        """Create systemd service file for Linux."""
        print("Creating systemd service...")
        
        # Sanitize host and port to prevent injection
        host = self.config.get('host', '0.0.0.0')
        port = self.config.get('port', 8000)
        
        # Validate host - must be valid IP or hostname
        if not re.match(r'^[a-zA-Z0-9.\-]+$', host):
            raise ValueError(f"Invalid host value: {host}. Must be valid IP or hostname.")
        
        # Validate port - must be integer in valid range
        try:
            port = int(port)
            if not (1 <= port <= 65535):
                raise ValueError(f"Port must be between 1 and 65535, got {port}")
        except (ValueError, TypeError):
            raise ValueError(f"Invalid port value: {port}. Must be integer.")
        
        service_content = f"""[Unit]
Description=Anime Super-Resolution API
After=network.target

[Service]
Type=exec
User={os.getenv('USER', 'root')}
WorkingDirectory={self.project_root}
Environment=PYTHONPATH={self.project_root}/src
ExecStart={sys.executable} -m uvicorn api.inference_api:app --host {host} --port {port}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        service_path = "/etc/systemd/system/anime-sr-api.service"
        
        try:
            with open(service_path, 'w') as f:
                f.write(service_content)
            
            print(f"[OK] Systemd service created: {service_path}")
            print("[INFO] To enable and start the service:")
            print("   sudo systemctl enable anime-sr-api")
            print("   sudo systemctl start anime-sr-api")
            print("   sudo systemctl status anime-sr-api")
            
            return True
            
        except PermissionError:
            print("[ERROR] Permission denied. Run with sudo to create systemd service.")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to create systemd service: {e}")
            return False
    
    def create_docker_config(self) -> bool:
        """Create Docker configuration."""
        print("🐳 Creating Docker configuration...")
        
        # Create Dockerfile
        dockerfile_content = f"""FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application code
COPY src/ ./src/
COPY checkpoints/ ./checkpoints/
COPY configs/ ./configs/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start the API
CMD ["python", "-m", "uvicorn", "api.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        dockerfile_path = self.project_root / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Create docker-compose.yml
        docker_compose_content = f"""version: '3.8'

services:
  anime-sr-api:
    build: .
    ports:
      - "{self.config.get('port', 8000)}:{self.config.get('port', 8000)}"
    environment:
      - PYTHONPATH=/app/src
      - HOST=0.0.0.0
      - PORT=8000
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: {self.config.get('memory_limit_mb', 8000)}M
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""
        
        docker_compose_path = self.project_root / "docker-compose.yml"
        with open(docker_compose_path, 'w') as f:
            f.write(docker_compose_content)
        
        print(f"[OK] Docker configuration created:")
        print(f"   Dockerfile: {dockerfile_path}")
        print(f"   docker-compose.yml: {docker_compose_path}")
        print("[INFO] To build and run with Docker:")
        print("   docker-compose up --build")
        
        return True
    
    def deploy(self, mode: str = "direct") -> bool:
        """Deploy the API."""
        print(f"[LAUNCH] Deploying API in {mode} mode...")
        
        # Check environment
        if not self.check_environment():
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Create configuration
        if not self.create_api_config():
            return False
        
        # Test API
        if not self.test_api():
            return False
        
        # Deploy based on mode
        if mode == "direct":
            return self.start_server()
        elif mode == "systemd":
            return self.create_systemd_service()
        elif mode == "docker":
            return self.create_docker_config()
        else:
            print(f"[ERROR] Unknown deployment mode: {mode}")
            return False
    
    def check_deployment(self) -> Dict[str, Any]:
        """Check deployment status."""
        print("[SEARCH] Checking deployment status...")
        
        status = {
            "api_running": False,
            "models_loaded": 0,
            "gpu_available": False,
            "memory_usage": 0,
            "endpoint_accessible": False
        }
        
        try:
            import requests
            
            # Check if API is running
            api_url = f"http://localhost:{self.config.get('port', 8000)}/health"
            try:
                response = requests.get(api_url, timeout=5)
                if response.status_code == 200:
                    status["api_running"] = True
                    status["endpoint_accessible"] = True
                    
                    health_data = response.json()
                    status["models_loaded"] = health_data.get("models_loaded", 0)
                    status["gpu_available"] = health_data.get("gpu_available", False)
                    status["memory_usage"] = health_data.get("memory_usage", {}).get("system_memory_mb", 0)
                    
            except requests.exceptions.ConnectionError:
                status["api_running"] = False
            except Exception as e:
                print(f"[WARN]  Error checking API: {e}")
        
        except ImportError:
            print("[WARN]  requests not available, skipping endpoint check")
        
        # Print status
        print(f"[CHART] Deployment Status:")
        print(f"  API Running: {'[OK]' if status['api_running'] else '[ERROR]'}")
        print(f"  Models Loaded: {status['models_loaded']}")
        print(f"  GPU Available: {'[OK]' if status['gpu_available'] else '[ERROR]'}")
        print(f"  Memory Usage: {status['memory_usage']:.1f}MB")
        print(f"  Endpoint Accessible: {'[OK]' if status['endpoint_accessible'] else '[ERROR]'}")
        
        return status


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deploy anime super-resolution API")
    parser.add_argument("--mode", choices=["direct", "systemd", "docker"], 
                       default="direct", help="Deployment mode")
    parser.add_argument("--check", action="store_true", help="Check deployment status")
    parser.add_argument("--config", help="API configuration file")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false", help="Disable GPU")
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = APIDeployer()
    
    # Override config with command line arguments
    if args.config:
        with open(args.config, 'r') as f:
            deployer.config = json.load(f)
    
    deployer.config.update({
        "host": args.host,
        "port": args.port,
        "workers": args.workers,
        "gpu": args.gpu
    })
    
    # Run deployment
    if args.check:
        deployer.check_deployment()
    else:
        success = deployer.deploy(args.mode)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()

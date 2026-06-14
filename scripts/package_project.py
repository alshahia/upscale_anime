#!/usr/bin/env python3
"""
Project packaging script for the anime super-resolution project.
Creates distributable packages with all necessary components.
"""

import os
import sys
import shutil
import subprocess
import json
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import platform


class ProjectPackager:
    """Comprehensive project packaging utility."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        
        # Ensure build directories exist
        self.build_dir.mkdir(exist_ok=True)
        self.dist_dir.mkdir(exist_ok=True)
        
        # Package configuration
        self.package_config = self.load_package_config()
        
    def load_package_config(self) -> Dict[str, Any]:
        """Load package configuration."""
        config_path = self.project_root / "package_config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "name": "anime-super-resolution",
                "version": "2.0.0",
                "description": "Anime Super-Resolution with Progressive Ensemble Distillation",
                "author": "Anime SR Team",
                "license": "MIT",
                "python_requires": ">=3.8",
                "include_models": True,
                "include_data": False,
                "include_docs": True,
                "include_tests": False,
                "package_formats": ["wheel", "sdist", "conda"],
                "docker": True,
                "docker_tag": "anime-sr:latest",
                "docker_registry": "localhost:5000"
            }
    
    def create_source_package(self) -> bool:
        """Create source distribution package."""
        print("[PKG] Creating source distribution...")
        
        try:
            # Create temporary directory
            temp_dir = self.build_dir / "source_package"
            temp_dir.mkdir(exist_ok=True)
            
            # Copy source files
            src_dir = temp_dir / "src"
            shutil.copytree(self.project_root / "src", src_dir)
            
            # Copy scripts
            scripts_dir = temp_dir / "scripts"
            shutil.copytree(self.project_root / "scripts", scripts_dir)
            
            # Copy configs
            configs_dir = temp_dir / "configs"
            shutil.copytree(self.project_root / "configs", configs_dir)
            
            # Copy requirements
            requirements_files = ["requirements.txt", "requirements_api.txt"]
            for req_file in requirements_files:
                req_path = self.project_root / req_file
                if req_path.exists():
                    shutil.copy2(req_path, temp_dir / req_file)
            
            # Copy README and license
            for file_name in ["README.md", "LICENSE"]:
                file_path = self.project_root / file_name
                if file_path.exists():
                    shutil.copy2(file_path, temp_dir / file_name)
            
            # Copy documentation if requested
            if self.package_config.get("include_docs", True):
                docs_dir = temp_dir / "docs"
                docs_src = self.project_root / "docs"
                if docs_src.exists():
                    shutil.copytree(docs_src, docs_dir)
            
            # Create setup.py
            self.create_setup_py(temp_dir)
            
            # Create MANIFEST.in
            self.create_manifest_in(temp_dir)
            
            # Create source distribution
            result = subprocess.run([
                sys.executable, "setup.py", "sdist"
            ], capture_output=True, text=True, cwd=temp_dir)
            
            if result.returncode == 0:
                # Move created package to dist directory
                dist_files = list(temp_dir.glob("dist/*.tar.gz"))
                for dist_file in dist_files:
                    shutil.move(dist_file, self.dist_dir / dist_file.name)
                
                print(f"[OK] Source distribution created: {len(dist_files)} packages")
                return True
            else:
                print(f"[ERROR] Source distribution failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error creating source distribution: {e}")
            return False
        finally:
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def create_wheel_package(self) -> bool:
        """Create wheel package."""
        print("🎡 Creating wheel package...")
        
        try:
            # Create temporary directory
            temp_dir = self.build_dir / "wheel_package"
            temp_dir.mkdir(exist_ok=True)
            
            # Copy source files
            src_dir = temp_dir / "src"
            shutil.copytree(self.project_root / "src", src_dir)
            
            # Copy requirements
            requirements_files = ["requirements.txt", "requirements_api.txt"]
            for req_file in requirements_files:
                req_path = self.project_root / req_file
                if req_path.exists():
                    shutil.copy2(req_path, temp_dir / req_file)
            
            # Create setup.py
            self.create_setup_py(temp_dir)
            
            # Create MANIFEST.in
            self.create_manifest_in(temp_dir)
            
            # Create pyproject.toml
            self.create_pyproject_toml(temp_dir)
            
            # Create wheel
            result = subprocess.run([
                sys.executable, "setup.py", "bdist_wheel"
            ], capture_output=True, text=True, cwd=temp_dir)
            
            if result.returncode == 0:
                # Move created package to dist directory
                dist_files = list(temp_dir.glob("dist/*.whl"))
                for dist_file in dist_files:
                    shutil.move(dist_file, self.dist_dir / dist_file.name)
                
                print(f"[OK] Wheel package created: {len(dist_files)} packages")
                return True
            else:
                print(f"[ERROR] Wheel package failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error creating wheel package: {e}")
            return False
        finally:
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def create_conda_package(self) -> bool:
        """Create conda package."""
        print("🐍 Creating conda package...")
        
        try:
            # Create conda recipe directory
            conda_dir = self.build_dir / "conda"
            conda_dir.mkdir(exist_ok=True)
            
            # Create meta.yaml
            self.create_conda_meta_yaml(conda_dir)
            
            # Build conda package
            result = subprocess.run([
                "conda-build", ".",
                "--output-folder", str(self.dist_dir / "conda")
            ], capture_output=True, text=True, cwd=conda_dir)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("[OK] Conda package created successfully")
                return True
            else:
                print(f"[ERROR] Conda package failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error creating conda package: {e}")
            return False
    
    def create_docker_package(self) -> bool:
        """Create Docker package."""
        print("🐳 Creating Docker package...")
        
        try:
            # Create Dockerfile
            self.create_dockerfile()
            
            # Create docker-compose.yml
            self.create_docker_compose()
            
            # Build Docker image
            image_name = self.package_config.get("docker_tag", "anime-sr:latest")
            
            result = subprocess.run([
                "docker", "build", "-t", image_name, "."
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                print(f"[OK] Docker image created: {image_name}")
                
                # Tag for registry if specified
                registry = self.package_config.get("docker_registry")
                if registry:
                    registry_tag = f"{registry}/{image_name}"
                    subprocess.run([
                        "docker", "tag", image_name, registry_tag
                    ])
                    print(f"[OK] Docker image tagged for registry: {registry_tag}")
                
                return True
            else:
                print(f"[ERROR] Docker build failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error creating Docker package: {e}")
            return False
    
    def create_setup_py(self, target_dir: Path):
        """Create setup.py file."""
        setup_content = f'''from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="{self.package_config['name']}",
    version="{self.package_config['version']}",
    author="{self.package_config['author']}",
    author_email="team@example.com",
    description="{self.package_config['description']}",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/anime-super-resolution",
    packages=find_packages(where="src"),
    package_dir={{"": "src"}},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires="{self.package_config['python_requires']}",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.5.0",
        "psutil>=5.8.0",
        "PyYAML>=6.0",
        "tqdm>=4.62.0",
    ],
    extras_require={{
        "api": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.23.0",
            "python-multipart>=0.0.6",
            "pydantic>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=8.0.0",
        ],
    }},
    entry_points={{
        "console_scripts": [
            "anime-sr-train=scripts.train:main",
            "anime-sr-infer=scripts.inference:main",
            "anime-sr-api=src.api.inference_api:main",
        ],
    }},
    include_package_data=True,
    package_data={{
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    }},
    zip_safe=False,
)
'''
        
        setup_file = target_dir / "setup.py"
        with open(setup_file, 'w') as f:
            f.write(setup_content)
    
    def create_manifest_in(self, target_dir: Path):
        """Create MANIFEST.in file."""
        manifest_content = '''
include README.md
include LICENSE
include requirements.txt
include requirements_api.txt
recursive-include src *.py
recursive-include scripts *.py
recursive-include configs *.yaml *.yml
recursive-include docs *.md *.rst
recursive-exclude * __pycache__
recursive-exclude * .pytest_cache
recursive-exclude logs/*
recursive-exclude build/*
recursive-exclude dist/*
'''
        
        manifest_file = target_dir / "MANIFEST.in"
        with open(manifest_file, 'w') as f:
            f.write(manifest_content)
    
    def create_pyproject_toml(self, target_dir: Path):
        """Create pyproject.toml file."""
        pyproject_content = f'''[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "{self.package_config['name']}"
version = "{self.package_config['version']}"
description = "{self.package_config['description']}"
readme = "README.md"
requires-python = "{self.package_config['python_requires']}"
license = {{text = "{self.package_config['license']}"}}
authors = [
    {{name = "{self.package_config['author']}", email = "team@example.com"}},
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.21.0",
    "Pillow>=9.0.0",
    "opencv-python>=4.5.0",
    "psutil>=5.8.0",
    "PyYAML>=6.0",
    "tqdm>=4.62.0",
]

[project.optional-dependencies]
api = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "python-multipart>=0.0.6",
    "pydantic>=2.0.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
    "mypy>=1.0.0",
]
docs = [
    "mkdocs>=1.4.0",
    "mkdocs-material>=8.0.0",
]

[project.scripts]
anime-sr-train = "scripts.train:main"
anime-sr-infer = "scripts.inference:main"
anime-sr-api = "src.api.inference_api:main"

[project.urls]
Homepage = "https://github.com/example/anime-super-resolution"
Documentation = "https://github.com/example/anime-super-resolution/docs"
Repository = "https://github.com/example/anime-super-resolution.git"
"Bug Tracker" = "https://github.com/example/anime-super-resolution/issues"
'''
        
        pyproject_file = target_dir / "pyproject.toml"
        with open(pyproject_file, 'w') as f:
            f.write(pyproject_content)
    
    def create_conda_meta_yaml(self, target_dir: Path):
        """Create conda meta.yaml file."""
        meta_content = f'''package:
  name: {self.package_config['name']}
  version: {self.package_config['version']}
  description: {self.package_config['description']}
  source:
    url: https://github.com/example/anime-super-resolution/archive/v{self.package_config['version']}.tar.gz
  sha256: placeholder  # This would be calculated automatically
  
build:
  number: 0
  noarch: python
  script: python -m pip install .
  entry_points:
    - anime-sr-train = scripts.train:main
    - anime-sr-infer = scripts.inference:main
    - anime-sr-api = src.api.inference_api:main

requirements:
  - python >={self.package_config['python_requires'].replace('>=', '')}
  - torch >=2.0.0
  - torchvision >=0.15.0
  - numpy >=1.21.0
  - Pillow >=9.0.0
  - opencv-python >=4.5.0
  - psutil >=5.8.0
  - PyYAML >=6.0
  - tqdm >=4.62.0

test:
  imports:
    - torch
    - torchvision
  commands:
    - python -c "import torch; print(torch.__version__)"

about:
  home: https://github.com/example/anime-super-resolution
  license: MIT
  license_family: MIT
  summary: {self.package_config['description']}
  description: |
    {self.package_config['description']}
  dev_url: https://github.com/example/anime-super-resolution
'''
        
        meta_file = target_dir / "meta.yaml"
        with open(meta_file, 'w') as f:
            f.write(meta_content)
    
    def create_dockerfile(self):
        """Create Dockerfile."""
        dockerfile_content = f'''FROM python:3.10-slim

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
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Copy checkpoints (optional)
{"" if self.package_config.get("include_models", True) else "#"}COPY checkpoints/ ./checkpoints/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "scripts/train.py", "--config", "configs/finetune_nan_safe.yaml"]
'''
        
        dockerfile_path = self.project_root / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
    
    def create_docker_compose(self):
        """Create docker-compose.yml file."""
        compose_content = f'''version: '3.8'

services:
  anime-sr:
    build: .
    container_name: anime-sr
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app/src
      - PYTHONUNBUFFERED=1
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  anime-sr-api:
    build: .
    container_name: anime-sr-api
    ports:
      - "8001:8000"
    environment:
      - PYTHONPATH=/app/src
      - PYTHONUNBUFFERED=1
    command: ["python", "-m", "uvicorn", "src.api.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
    restart: unless-stopped
    depends_on:
      - anime-sr

volumes:
  checkpoints:
  data:
  logs:
'''
        
        compose_file = self.project_root / "docker-compose.yml"
        with open(compose_file, 'w') as f:
            f.write(compose_content)
    
    def create_package_info(self) -> Dict[str, Any]:
        """Create package information summary."""
        info = {
            "name": self.package_config["name"],
            "version": self.package_config["version"],
            "created": datetime.now().isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "build_directory": str(self.build_dir),
            "dist_directory": str(self.dist_dir),
            "package_formats": self.package_config.get("package_formats", []),
            "components": {
                "source_code": True,
                "models": self.package_config.get("include_models", True),
                "data": self.package_config.get("include_data", False),
                "docs": self.package_config.get("include_docs", True),
                "tests": self.package_config.get("include_tests", False),
                "docker": self.package_config.get("docker", True),
                "api": True
            }
        }
        
        return info
    
    def package_all(self) -> bool:
        """Package the project in all configured formats."""
        print("[LAUNCH] Starting complete packaging process...")
        print("=" * 50)
        
        success = True
        
        # Create package info
        package_info = self.create_package_info()
        print(f"[PKG] Package: {package_info['name']} v{package_info['version']}")
        print(f"[MONITOR]  Platform: {package_info['platform']}")
        print(f"🐍 Python: {package_info['python_version']}")
        
        # Package formats to create
        formats = self.package_config.get("package_formats", [])
        
        if "sdist" in formats:
            success &= self.create_source_package()
        
        if "wheel" in formats:
            success &= self.create_wheel_package()
        
        if "conda" in formats:
            success &= self.create_conda_package()
        
        if self.package_config.get("docker", True):
            success &= self.create_docker_package()
        
        # List created packages
        print("\n[LIST] Created Packages:")
        
        if self.dist_dir.exists():
            for package_file in self.dist_dir.iterdir():
                size_mb = package_file.stat().st_size / 1024 / 1024
                print(f"  [PKG] {package_file.name} ({size_mb:.1f}MB)")
        
        # Save package info
        info_file = self.dist_dir / "package_info.json"
        with open(info_file, 'w') as f:
            json.dump(package_info, f, indent=2, default=str)
        
        print(f"\n📄 Package info saved to: {info_file}")
        
        return success
    
    def clean_build_artifacts(self):
        """Clean build artifacts."""
        print("🧹 Cleaning build artifacts...")
        
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
            print(f"[OK] Removed build directory: {self.build_dir}")
        
        if self.dist_dir.exists():
            shutil.rmtree(self.dist_dir)
            print(f"[OK] Removed dist directory: {self.dist_dir}")
        
        # Remove Docker artifacts
        docker_files = ["Dockerfile", "docker-compose.yml"]
        for docker_file in docker_files:
            file_path = self.project_root / docker_file
            if file_path.exists():
                file_path.unlink()
                print(f"[OK] Removed {docker_file}")
        
        print("[OK] Build artifacts cleaned")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Package anime super-resolution project")
    parser.add_argument("--format", choices=["all", "sdist", "wheel", "conda", "docker"], 
                       default="all", help="Package format to create")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--info", action="store_true", help="Show package information")
    
    args = parser.parse_args()
    
    packager = ProjectPackager()
    
    if args.clean:
        packager.clean_build_artifacts()
        return 0
    
    if args.info:
        info = packager.create_package_info()
        print("[PKG] Package Information:")
        print(json.dumps(info, indent=2, default=str))
        return 0
    
    # Package based on format
    success = True
    
    if args.format == "all":
        success = packager.package_all()
    elif args.format == "sdist":
        success = packager.create_source_package()
    elif args.format == "wheel":
        success = packager.create_wheel_package()
    elif args.format == "conda":
        success = packager.create_conda_package()
    elif args.format == "docker":
        success = packager.create_docker_package()
    
    if success:
        print("\n[CELEBRATE] Packaging completed successfully!")
        return 0
    else:
        print("\n[ERROR] Packaging failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

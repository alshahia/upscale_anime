"""
Checkpoint Management System
Handles run naming, duplicate detection, and tracker CSV for all training runs.
"""
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List


class CheckpointManager:
    """Manages checkpoint directories, naming, and tracking."""

    def __init__(self, config: Dict, cli_run_name: Optional[str] = None, resume_from: Optional[str] = None):
        self.config = config
        self.base_dir = Path(config.get('paths', {}).get('checkpoint_dir',
                           config.get('training', {}).get('checkpoint_dir', 'checkpoints')))
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.tracker_path = self.base_dir / 'run_tracker.csv'

        self.output_cfg = config.get('output', {})
        self.cli_run_name = cli_run_name
        self.config_run_name = self.output_cfg.get('run_name')
        self.duplicate_handling = self.output_cfg.get('duplicate_handling', 'auto_increment')

        self.run_id = None
        self.run_name = None
        self.checkpoint_dir = None
        self.run_config = None

        if resume_from:
            self._resolve_from_resume(resume_from)
        else:
            self._resolve_run_path()

    def _generate_hybrid_name(self) -> str:
        """Generate auto name: run_{timestamp}_{config}_{suffix}"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_type = self.config.get('model', {}).get('type', 'span')
        model_name = self.config.get('model', {}).get('name', 'unknown')

        progressive = self.config.get('training', {}).get('finetune', {}).get('progressive_crop', {}).get('enabled', False)
        suffix = "progCrop" if progressive else "fixedCrop"

        if model_name and model_name != 'unknown':
            return f"run_{timestamp}_{model_name}_{suffix}"
        return f"run_{timestamp}_{model_type}_{suffix}"

    def _sanitize_name(self, name: str) -> str:
        """Sanitize run name for filesystem compatibility."""
        name = re.sub(r'[<>:"/\\|?*]', '_', name)
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        return name[:100]

    def _find_existing_runs(self, base_name: str) -> List[str]:
        """Find all existing runs matching base_name pattern."""
        existing = []
        if not self.base_dir.exists():
            return existing

        for d in self.base_dir.iterdir():
            if d.is_dir() and d.name.startswith(base_name):
                existing.append(d.name)
        return sorted(existing)

    def _extract_number_suffix(self, name: str, base_name: str) -> Optional[int]:
        """Extract number suffix from name if it matches base_name pattern."""
        if not name.startswith(base_name):
            return None

        remainder = name[len(base_name):]

        if remainder.startswith('_'):
            num_str = remainder[1:]
            if num_str.isdigit():
                return int(num_str)
            if num_str.startswith('v') and num_str[1:].isdigit():
                return int(num_str[1:])

        return None

    def _get_next_increment(self, base_name: str) -> str:
        """Get next increment suffix for duplicate name."""
        existing = self._find_existing_runs(base_name)

        if not existing:
            return f"{base_name}_001"

        max_num = 0
        for name in existing:
            num = self._extract_number_suffix(name, base_name)
            if num is not None:
                max_num = max(max_num, num)

        return f"{base_name}_{max_num + 1:03d}"

    def _resolve_run_path(self):
        """Resolve the final run name and checkpoint path."""
        raw_name = self.cli_run_name or self.config_run_name

        if raw_name:
            candidate_name = self._sanitize_name(raw_name)
        else:
            candidate_name = self._generate_hybrid_name()

        candidate_path = self.base_dir / candidate_name

        if candidate_path.exists():
            if self.duplicate_handling == 'auto_increment':
                final_name = self._get_next_increment(candidate_name)
                final_path = self.base_dir / final_name
            else:
                final_name = candidate_name
                final_path = candidate_path
        else:
            final_name = candidate_name
            final_path = candidate_path

        self.run_name = final_name
        self.checkpoint_dir = final_path
        self.run_id = self._generate_run_id()

        finetune_cfg = self.config.get('training', {}).get('finetune', {})
        self.run_config = {
            'run_id': self.run_id,
            'run_name': self.run_name,
            'config_file': self._get_config_file_name(),
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
            'status': 'started',
            'best_loss': '',
            'final_epoch': '',
            'lr': finetune_cfg.get('lr', ''),
            'batch_size': finetune_cfg.get('batch_size', ''),
            'epochs': finetune_cfg.get('epochs', ''),
            'checkpoint_dir': str(self.checkpoint_dir),
            'psnr': '',
            'ssim': '',
            'lpips': '',
        }

    def _resolve_from_resume(self, resume_path: str):
        """Resolve run info from existing checkpoint when resuming."""
        resume_path = Path(resume_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        self.checkpoint_dir = resume_path.parent
        self.run_name = self.checkpoint_dir.name

        existing_run = self._find_run_by_dir(str(self.checkpoint_dir))
        if existing_run:
            self.run_id = existing_run.get('run_id', self._generate_run_id())
            rows = self._load_tracker()
            for row in rows:
                if row.get('run_id') == self.run_id:
                    row['status'] = 'resumed'
                    row['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                    break
            self._save_tracker(rows)
            self._is_resumed = True
        else:
            self.run_id = self._generate_run_id()
            self._is_resumed = False

        finetune_cfg = self.config.get('training', {}).get('finetune', {})
        self.run_config = {
            'run_id': self.run_id,
            'run_name': self.run_name,
            'config_file': self._get_config_file_name(),
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
            'status': 'resumed',
            'best_loss': '',
            'final_epoch': '',
            'lr': finetune_cfg.get('lr', ''),
            'batch_size': finetune_cfg.get('batch_size', ''),
            'epochs': finetune_cfg.get('epochs', ''),
            'checkpoint_dir': str(self.checkpoint_dir),
            'psnr': '',
            'ssim': '',
            'lpips': '',
        }

    def is_resumed(self) -> bool:
        """Check if this run was resumed from existing checkpoint."""
        return getattr(self, '_is_resumed', False)

    def _find_run_by_dir(self, checkpoint_dir: str) -> Optional[Dict]:
        """Find existing run in tracker by checkpoint directory."""
        rows = self._load_tracker()
        for row in rows:
            if row.get('checkpoint_dir') == checkpoint_dir:
                return row
        return None

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        existing = self._load_tracker()
        used_ids = set()
        for row in existing:
            if row.get('run_id'):
                used_ids.add(row['run_id'])

        run_num = 1
        while f"run_{run_num:03d}" in used_ids:
            run_num += 1
        return f"run_{run_num:03d}"

    def _get_config_file_name(self) -> str:
        """Get config file name from config object."""
        if hasattr(self.config, '_source_path'):
            return Path(self.config._source_path).name
        return self.config.get('_config_file', 'unknown.yaml')

    def get_checkpoint_dir(self) -> Path:
        """Get the resolved checkpoint directory."""
        return self.checkpoint_dir

    def is_duplicate(self) -> bool:
        """Check if run name was auto-incremented."""
        if not self.cli_run_name and not self.config_run_name:
            return False
        original = self._sanitize_name(self.cli_run_name or self.config_run_name)
        return self.run_name != original

    def get_duplicate_info(self) -> Optional[Dict]:
        """Get info about duplicate handling if applicable."""
        if not self.is_duplicate():
            return None

        original = self._sanitize_name(self.cli_run_name or self.config_run_name)
        existing = self._find_existing_runs(original)

        return {
            'original_name': original,
            'assigned_name': self.run_name,
            'existing_runs': existing,
            'handling': self.duplicate_handling,
        }

    def _init_tracker(self):
        """Initialize tracker CSV with headers."""
        headers = ['run_id', 'run_name', 'config_file', 'timestamp', 'status',
                   'best_loss', 'final_epoch', 'lr', 'batch_size', 'epochs', 'checkpoint_dir',
                   'psnr', 'ssim', 'lpips']
        with open(self.tracker_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    def _load_tracker(self) -> List[Dict]:
        """Load tracker CSV."""
        if not self.tracker_path.exists():
            return []
        try:
            with open(self.tracker_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception:
            return []

    def _save_tracker(self, rows: List[Dict]):
        """Save tracker CSV."""
        if not rows:
            return
        # Use union of all keys from all rows to handle schema evolution
        headers = []
        for row in rows:
            for key in row.keys():
                if key not in headers:
                    headers.append(key)
        with open(self.tracker_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

    def register_run(self):
        """Register new run in tracker."""
        if not self.tracker_path.exists():
            self._init_tracker()

        rows = self._load_tracker()
        rows.append(self.run_config)
        self._save_tracker(rows)

    def update_run(self, updates: Dict):
        """Update run info in tracker."""
        rows = self._load_tracker()
        for row in rows:
            if row.get('run_id') == self.run_id:
                row.update(updates)
                row['timestamp'] = row.get('timestamp', datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
                break
        self._save_tracker(rows)

    def get_all_runs(self) -> List[Dict]:
        """Get all runs from tracker."""
        return self._load_tracker()

    def get_run_info(self, run_id: Optional[str] = None, run_name: Optional[str] = None) -> Optional[Dict]:
        """Get info for specific run."""
        rows = self._load_tracker()
        for row in rows:
            if run_id and row.get('run_id') == run_id:
                return row
            if run_name and row.get('run_name') == run_name:
                return row
        return None

    def print_run_info(self):
        """Print current run info."""
        print(f"\n[Checkpoint Manager]")
        print(f"  Run ID: {self.run_id}")
        print(f"  Run Name: {self.run_name}")
        print(f"  Checkpoint Dir: {self.checkpoint_dir}")
        if self.is_duplicate():
            dup_info = self.get_duplicate_info()
            print(f"  [AUTO-INCREMENT] Original name was taken, assigned: {dup_info['assigned_name']}")
        print(f"  Tracker: {self.tracker_path}")
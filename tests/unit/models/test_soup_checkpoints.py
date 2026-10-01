"""Tests for the model-souping script."""
import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent.parent / 'scripts'

spec = importlib.util.spec_from_file_location(
    'soup_checkpoints',
    SCRIPT_DIR / 'soup_checkpoints.py',
)
soup_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soup_module)
soup_checkpoints = soup_module.soup_checkpoints
load_state_dict = soup_module.load_state_dict


def _make_dummy_checkpoint(path: str, ema_sd: dict, model_sd: dict = None,
                           epoch: int = 0, has_scaler: bool = True):
    """Write a fake checkpoint to disk."""
    ckpt = {
        'ema_state_dict': ema_sd,
        'epoch': epoch,
    }
    if model_sd is not None:
        ckpt['model_state_dict'] = model_sd
    if has_scaler:
        ckpt['scaler_state_dict'] = {'scale': torch.tensor(1.0)}
        ckpt['optimizer_state_dict'] = {'step': 0}
    torch.save(ckpt, path)


class TestSoupCheckpoints:
    """soup_checkpoints should average state-dicts and produce a valid output."""

    def test_soup_two_checkpoints_ema_only(self):
        """Average 2 EMA-only checkpoints, output contains EMA only (no scaler)."""
        with tempfile.TemporaryDirectory() as tmp:
            a_path = str(Path(tmp) / 'a.pth')
            b_path = str(Path(tmp) / 'b.pth')
            out_path = str(Path(tmp) / 'out.pth')

            ema_a = {'w1': torch.tensor([1.0, 2.0]), 'w2': torch.tensor([3.0])}
            ema_b = {'w1': torch.tensor([3.0, 4.0]), 'w2': torch.tensor([5.0])}
            _make_dummy_checkpoint(a_path, ema_a, epoch=10)
            _make_dummy_checkpoint(b_path, ema_b, epoch=20)

            info = soup_checkpoints([a_path, b_path], out_path)

            assert info['keys'] == 2
            assert info['epoch'] == 20
            assert info['sources'] == ['ema_state_dict', 'ema_state_dict']

            out_ckpt = torch.load(out_path, map_location='cpu', weights_only=False)
            assert 'ema_state_dict' in out_ckpt
            assert torch.allclose(out_ckpt['ema_state_dict']['w1'],
                                  torch.tensor([2.0, 3.0]))
            assert torch.allclose(out_ckpt['ema_state_dict']['w2'],
                                  torch.tensor([4.0]))

            assert 'scaler_state_dict' not in out_ckpt
            assert 'optimizer_state_dict' not in out_ckpt

            assert out_ckpt['soup_inputs'] == [a_path, b_path]
            assert out_ckpt['soup_n'] == 2

    def test_soup_falls_back_to_model_state_dict_when_no_ema(self):
        """If EMA is absent, fall back to model_state_dict."""
        with tempfile.TemporaryDirectory() as tmp:
            a_path = str(Path(tmp) / 'a.pth')
            b_path = str(Path(tmp) / 'b.pth')
            out_path = str(Path(tmp) / 'out.pth')

            a_sd = {'w1': torch.tensor([2.0, 4.0])}
            b_sd = {'w1': torch.tensor([6.0, 8.0])}
            torch.save({'model_state_dict': a_sd}, a_path)
            torch.save({'model_state_dict': b_sd}, b_path)

            info = soup_checkpoints([a_path, b_path], out_path)
            assert info['sources'] == ['model_state_dict', 'model_state_dict']
            out_ckpt = torch.load(out_path, map_location='cpu', weights_only=False)
            assert torch.allclose(out_ckpt['ema_state_dict']['w1'],
                                  torch.tensor([4.0, 6.0]))

    def test_soup_key_mismatch_raises(self):
        """Checkpoints with different architectures raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            a_path = str(Path(tmp) / 'a.pth')
            b_path = str(Path(tmp) / 'b.pth')
            out_path = str(Path(tmp) / 'out.pth')

            a_sd = {'w1': torch.tensor([1.0]), 'w2': torch.tensor([2.0])}
            b_sd = {'w1': torch.tensor([3.0]), 'w3': torch.tensor([4.0])}
            _make_dummy_checkpoint(a_path, a_sd)
            _make_dummy_checkpoint(b_path, b_sd)

            with pytest.raises(ValueError, match='key mismatch'):
                soup_checkpoints([a_path, b_path], out_path)

    def test_soup_requires_two_or_more(self):
        """Single checkpoint raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            a_path = str(Path(tmp) / 'a.pth')
            out_path = str(Path(tmp) / 'out.pth')
            _make_dummy_checkpoint(a_path, {'w1': torch.tensor([1.0])})

            with pytest.raises(ValueError, match='>= 2'):
                soup_checkpoints([a_path], out_path)

    def test_soup_three_checkpoints(self):
        """Three-checkpoint soup averages correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, val in enumerate([[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]]):
                p = str(Path(tmp) / f'c{i}.pth')
                paths.append(p)
                _make_dummy_checkpoint(p, {'w1': torch.tensor(val)})

            out_path = str(Path(tmp) / 'out.pth')
            info = soup_checkpoints(paths, out_path)
            assert info['keys'] == 1
            out_ckpt = torch.load(out_path, map_location='cpu', weights_only=False)
            assert torch.allclose(out_ckpt['ema_state_dict']['w1'],
                                  torch.tensor([4.0, 6.0]))

    def test_soup_preserves_dtype(self):
        """Output tensor dtype matches input dtype (no implicit upcast)."""
        with tempfile.TemporaryDirectory() as tmp:
            a_path = str(Path(tmp) / 'a.pth')
            b_path = str(Path(tmp) / 'b.pth')
            out_path = str(Path(tmp) / 'out.pth')

            ema_a = {'w1': torch.tensor([1.0, 2.0], dtype=torch.float16)}
            ema_b = {'w1': torch.tensor([3.0, 4.0], dtype=torch.float16)}
            _make_dummy_checkpoint(a_path, ema_a)
            _make_dummy_checkpoint(b_path, ema_b)

            soup_checkpoints([a_path, b_path], out_path)
            out_ckpt = torch.load(out_path, map_location='cpu', weights_only=False)
            assert out_ckpt['ema_state_dict']['w1'].dtype == torch.float16


class TestLoadStateDict:
    """load_state_dict helper should prefer ema_state_dict over other keys."""

    def test_prefers_ema(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = str(Path(tmp) / 'c.pth')
            ema = {'w': torch.tensor([1.0])}
            model = {'w': torch.tensor([99.0])}
            torch.save({'ema_state_dict': ema, 'model_state_dict': model}, p)
            sd, src, full = load_state_dict(p)
            assert src == 'ema_state_dict'
            assert torch.allclose(sd['w'], torch.tensor([1.0]))

    def test_handles_raw_state_dict(self):
        """If the checkpoint IS a state_dict (no nested keys), use it as-is."""
        with tempfile.TemporaryDirectory() as tmp:
            p = str(Path(tmp) / 'c.pth')
            raw = {'w': torch.tensor([42.0])}
            torch.save(raw, p)
            sd, src, full = load_state_dict(p)
            assert src == '<raw>'
            assert torch.allclose(sd['w'], torch.tensor([42.0]))

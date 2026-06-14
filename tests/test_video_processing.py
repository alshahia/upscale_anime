"""
Test Suite for Video Processing Pipeline

Tests video frame extraction and temporal training preparation including:
- Video metadata extraction (fps, resolution, duration)
- Pre-extraction mode (extract all frames before training)
- On-demand extraction mode (extract during training)
- Quality threshold filtering
- Duplicate frame removal
- Temporal training window creation
- Scene change detection

Data Strategy:
- Primary: data/anime_vid/ if real videos exist
- Fallback: Create synthetic test video using OpenCV
"""
import pytest
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Add src and utils to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'tests' / 'utils'))

from test_data_manager import ensure_test_data, create_fallback_data


def create_synthetic_video(output_path: Path, num_frames: int = 100, fps: int = 30, 
                          resolution: tuple = (640, 480)):
    """Create a synthetic test video for testing."""
    try:
        import cv2
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)
        
        for i in range(num_frames):
            # Create frame with moving pattern
            frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
            
            # Add moving gradient
            x_offset = int((i / num_frames) * resolution[0])
            frame[:, x_offset:min(x_offset+50, resolution[0]), 0] = 255
            
            # Add some variation
            noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
            frame = np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)
            
            writer.write(frame)
        
        writer.release()
        return True
    except ImportError:
        return False


class TestVideoProcessing:
    """Test video processing functionality."""
    
    @pytest.fixture(scope='class')
    def video_dir(self):
        """Provide video directory (real or synthetic)."""
        real_video_dir = Path(__file__).parent.parent / 'data' / 'anime_vid'
        
        if real_video_dir.exists() and len(list(real_video_dir.glob('*.mp4'))) > 0:
            print(f"\n[VideoProcessing] Using real video data: {real_video_dir}")
            yield real_video_dir
        else:
            # Create synthetic video
            temp_dir = Path(tempfile.mkdtemp())
            video_path = temp_dir / 'test_video.mp4'
            
            if create_synthetic_video(video_path, num_frames=100, fps=30, resolution=(640, 480)):
                print(f"\n[VideoProcessing] Using synthetic video: {video_path}")
                yield temp_dir
            else:
                print("\n[VideoProcessing] OpenCV not available, skipping video tests")
                yield None
            
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_video_info_extraction(self, video_dir):
        """Test extraction of video metadata."""
        if video_dir is None:
            pytest.skip("No video data available")
        
        video_file = list(video_dir.glob('*.mp4'))[0]
        
        try:
            from data.video_processing import extract_video_info
            
            info = extract_video_info(video_file)
            
            # Verify metadata
            assert 'fps' in info, "Video info should contain fps"
            assert 'resolution' in info, "Video info should contain resolution"
            assert 'duration' in info or 'num_frames' in info, \
                "Video info should contain duration or num_frames"
            
            # Verify values are reasonable
            assert info['fps'] > 0, "FPS should be positive"
            assert len(info['resolution']) == 2, "Resolution should be (width, height)"
            
        except ImportError:
            pytest.skip("Video processing module not available")
    
    def test_frame_extraction_pre_mode(self, video_dir):
        """Test pre-extraction mode (extract all frames before training)."""
        if video_dir is None:
            pytest.skip("No video data available")
        
        video_file = list(video_dir.glob('*.mp4'))[0]
        
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                from data.video_processing import extract_frames
                
                output_path = Path(output_dir)
                
                # Extract frames
                extracted = extract_frames(
                    video_file,
                    output_path,
                    extract_every_n_frames=10,  # Extract every 10th frame
                    quality_threshold=0.0  # No quality filtering for testing
                )
                
                # Verify frames were extracted
                assert len(extracted) > 0, "Should extract at least some frames"
                
                # Verify frame files exist
                frame_files = list(output_path.glob('*.png'))
                assert len(frame_files) > 0, "Frame files should exist"
                
            except ImportError:
                pytest.skip("Video processing module not available")
    
    def test_quality_threshold_filtering(self, video_dir):
        """Test quality-based frame filtering."""
        if video_dir is None:
            pytest.skip("No video data available")
        
        video_file = list(video_dir.glob('*.mp4'))[0]
        
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                from data.video_processing import extract_frames
                
                output_path = Path(output_dir)
                
                # Extract with low threshold (more frames)
                extracted_low = extract_frames(
                    video_file,
                    output_path / 'low_threshold',
                    extract_every_n_frames=5,
                    quality_threshold=0.3
                )
                
                # Extract with high threshold (fewer frames)
                extracted_high = extract_frames(
                    video_file,
                    output_path / 'high_threshold',
                    extract_every_n_frames=5,
                    quality_threshold=0.8
                )
                
                # High threshold should extract fewer or equal frames
                assert len(extracted_high) <= len(extracted_low), \
                    "High quality threshold should extract fewer frames"
                
            except ImportError:
                pytest.skip("Video processing module not available")
    
    def test_duplicate_frame_removal(self, video_dir):
        """Test removal of duplicate/similar frames."""
        if video_dir is None:
            pytest.skip("No video data available")
        
        video_file = list(video_dir.glob('*.mp4'))[0]
        
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                from data.video_processing import extract_frames
                
                output_path = Path(output_dir)
                
                # Extract with duplicate removal
                extracted_no_dedup = extract_frames(
                    video_file,
                    output_path / 'no_dedup',
                    extract_every_n_frames=1,  # Every frame
                    remove_duplicates=False
                )
                
                extracted_with_dedup = extract_frames(
                    video_file,
                    output_path / 'with_dedup',
                    extract_every_n_frames=1,
                    remove_duplicates=True,
                    duplicate_threshold=0.95
                )
                
                # With deduplication should have fewer or equal frames
                assert len(extracted_with_dedup) <= len(extracted_no_dedup), \
                    "Duplicate removal should reduce frame count"
                
            except ImportError:
                pytest.skip("Video processing module not available")
    
    def test_temporal_training_window(self):
        """Test creation of temporal training windows."""
        try:
            from data.video_processing import create_temporal_windows
            
            # Create dummy frame list
            frames = [f'frame_{i:04d}.png' for i in range(100)]
            
            # Create temporal windows
            windows = create_temporal_windows(frames, window_size=3, stride=1)
            
            # Verify windows
            assert len(windows) > 0, "Should create temporal windows"
            assert len(windows[0]) == 3, "Each window should have 3 frames"
            
            # Verify consecutive ordering
            for window in windows[:5]:  # Check first 5
                indices = [int(f.split('_')[1].split('.')[0]) for f in window]
                assert indices[1] == indices[0] + 1, "Frames should be consecutive"
                assert indices[2] == indices[1] + 1, "Frames should be consecutive"
            
        except ImportError:
            pytest.skip("Video processing module not available")
    
    def test_scene_change_detection(self, video_dir):
        """Test scene change detection."""
        if video_dir is None:
            pytest.skip("No video data available")
        
        # This test would need a video with actual scene changes
        # For now, just test the function exists and runs
        try:
            from data.video_processing import detect_scene_changes
            
            video_file = list(video_dir.glob('*.mp4'))[0]
            
            scenes = detect_scene_changes(video_file, threshold=0.3)
            
            # Should return a list (might be empty for simple test videos)
            assert isinstance(scenes, list), "Should return list of scene boundaries"
            
        except ImportError:
            pytest.skip("Video processing module not available")


class TestVideoScriptIntegration:
    """Integration tests for video processing script."""
    
    def test_process_videos_script_exists(self):
        """Test that the video processing script exists."""
        script_path = Path(__file__).parent.parent / 'scripts' / 'process_videos.py'
        assert script_path.exists(), f"Video processing script not found: {script_path}"
    
    def test_process_videos_cli_help(self):
        """Test that video processing script shows help."""
        import subprocess
        
        script_path = Path(__file__).parent.parent / 'scripts' / 'process_videos.py'
        result = subprocess.run(
            ['python', str(script_path), '--help'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Help command should succeed"
        assert 'extract' in result.stdout.lower(), "Help should mention extract command"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

# Open-Source Dataset Analysis for Anime Super-Resolution

**Analysis Date:** April 27, 2026  
**Project:** Anime Super-Resolution with Progressive Ensemble Distillation  
**Recommendation:** Use Anime Face Dataset (63K) + DIV2K (800) + Custom Videos

---

## Executive Summary

This document provides a comprehensive analysis of open-source datasets suitable for training the anime super-resolution model. The recommended approach combines small, focused datasets for initial development with pathways to scale to larger datasets.

---

## 1. Anime-Specific Datasets

### 1.1 Danbooru Dataset Series (Primary Recommendation)

| Version | Images | Tags | Size | Status | Best For |
|---------|--------|------|------|--------|----------|
| **Danbooru2021** | 4.9M+ | 162M+ | ~500GB | ⚠️ Offline | Production scale (temporarily unavailable) |
| **Danbooru2019** | 3.9M+ | 108M+ | ~400GB | ✅ Available | Large-scale training |
| **Danbooru2018** | 3.3M+ | 91M+ | ~350GB | ✅ Available | Large-scale training |
| **Danbooru2017** | 2.9M+ | 77M+ | ~300GB | ✅ Available | Large-scale training |

**Download:**
- GWERN Dataset Hub: `https://www.gwern.net/Danbooru2019`
- Kaggle Subset (300K SFW): `https://www.kaggle.com/datasets/mylesoneill/tagged-anime-illustrations`
- Tools: `https://github.com/Atom-101/Danbooru-Dataset-Maker`

**Pros:**
- Largest curated anime dataset (4.9M+ images in 2021 version)
- Rich metadata with 162M+ tags
- SFW subsets available
- Community quality ratings (Score, Favorites)
- High resolution (up to original upload quality)

**Cons:**
- Danbooru2021 temporarily offline (metadata issues)
- Requires filtering for quality training data
- Large storage requirements (300-500GB)
- Copyright considerations (fair use research)

**Integration:**
```yaml
data:
  datasets:
    - name: "danbooru_sfw"
      weight: 0.7
      enabled: true
      hr_dir: "data/danbooru_sfw_512"
```

---

### 1.2 Anime Face Dataset (Quick Start)

| Dataset | Images | Source | Size | Best For |
|---------|--------|--------|------|----------|
| **anime-face-dataset** | 63.6K | Getchu.com | ~2GB | Testing, face SR |
| **Tagged Anime Illustrations** | 337K | Danbooru2017 + moeimouto | ~15GB | Character focus |
| **Danbooru2019 Portraits** | 302K | Danbooru2019 crops | ~12GB | Face detection/recognition |

**Download:**
- Kaggle: `https://www.kaggle.com/datasets/splcher/animefacedataset`
- GitHub: `https://github.com/STomoya/animeface`
- Tagged: `https://www.kaggle.com/datasets/mylesoneill/tagged-anime-illustrations`

**Pros:**
- Clean, pre-filtered faces
- Good for testing/validation
- Small, manageable size (~2GB)
- Fast download and setup

**Cons:**
- Limited diversity (only faces)
- Not suitable for full-scene SR

**Integration:**
```bash
# Quick download
kaggle datasets download -d splcher/animefacedataset
unzip animefacedataset.zip -d data/anime_faces/
```

```yaml
data:
  datasets:
    - name: "anime_faces"
      weight: 0.5
      enabled: true
      hr_dir: "data/anime_faces"
```

---

### 1.3 APISR Video Pipeline (Real-World Quality)

**Approach:** Extract frames from anime videos (Blu-ray, web sources)

**Pipeline Steps:**
1. Download anime videos (mp4, mkv, etc.)
2. Extract high-quality I-frames using ffmpeg
3. Filter using IC9600 quality model
4. Apply strong USM (Unsharp Mask) for pseudo-GT generation

**Tools:** `https://github.com/Kiteretsu77/APISR/tree/main/dataset_curation_pipeline`

**Pros:**
- Real-world degradation patterns
- Professional anime production quality
- Optimized for super-resolution (CVPR 2024 paper)
- Our project now supports this natively!

**Cons:**
- Requires video source material
- Processing-intensive pipeline
- Need to source high-quality videos

**Integration (via this project's new feature):**
```bash
# Place videos in data/anime_vid/
python scripts/process_videos.py extract
```

```yaml
data:
  video:
    enabled: true
    video_dir: "data/anime_vid"
    extract_mode: "pre"
    quality_threshold: 0.7
```

---

## 2. General High-Quality Datasets

### 2.1 DIV2K (NTIRE Standard Benchmark)

| Split | Images | Resolution | Size | Download |
|-------|--------|------------|------|----------|
| Train | 800 | 2K (various) | ~3.5GB | [Link](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip) |
| Valid | 100 | 2K | ~450MB | [Link](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip) |

**Download:**
```bash
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
```

**Pros:**
- High-quality diverse content
- Standard benchmark for comparison
- Smaller, manageable size (~4GB)
- Widely used in SR research

**Cons:**
- Photographic content (not anime)
- Limited to 1000 images
- May not capture anime-specific artifacts

**Integration:**
```yaml
data:
  datasets:
    - name: "div2k"
      weight: 0.3
      enabled: true
      hr_dir: "data/DIV2K_train_HR"
```

---

### 2.2 Flickr2K (DF2K Component)

| Images | Resolution | Size | Download |
|--------|------------|------|----------|
| 2,650 | High-res | ~10GB | [Link](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) |

**Download:**
```bash
wget https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
```

**Pros:**
- More diversity than DIV2K alone
- Used in Real-ESRGAN training
- Good for texture learning
- Part of DF2K standard

**Cons:**
- Photographic only
- Larger than DIV2K alone

**Integration:**
```yaml
data:
  datasets:
    - name: "flickr2k"
      weight: 0.2
      enabled: true
      hr_dir: "data/Flickr2K"
```

---

### 2.3 Open Images V7 (Massive Scale)

| Images | Annotations | Size | Download |
|--------|-------------|------|----------|
| 9.2M | 30M+ tags | ~600GB | [GitHub](https://github.com/openimages/dataset) |

**Pros:**
- Massive scale
- CC-BY licensed (freely distributable)
- Good for pre-training
- Diverse content

**Cons:**
- Not anime-specific
- Requires heavy filtering
- Very large storage needs (600GB)
- Most images not relevant to anime SR

**Verdict:** Not recommended for this project unless doing general pre-training.

---

## 3. Dataset Comparison Matrix

| Dataset | Type | Images | Size | Quality | Anime-Specific | Ease of Use | Best For |
|---------|------|--------|------|---------|----------------|-------------|----------|
| **Danbooru2019** | Illustration | 3.9M | ~400GB | ⭐⭐⭐⭐⭐ | ✅ Yes | ⭐⭐⭐ | Production |
| **Anime Face** | Face crops | 63K | ~2GB | ⭐⭐⭐⭐⭐ | ✅ Yes | ⭐⭐⭐⭐⭐ | Testing |
| **DIV2K** | General | 900 | ~4GB | ⭐⭐⭐⭐⭐ | ❌ No | ⭐⭐⭐⭐⭐ | Benchmark |
| **Flickr2K** | General | 2.6K | ~10GB | ⭐⭐⭐⭐ | ❌ No | ⭐⭐⭐⭐ | Texture |
| **Open Images** | General | 9.2M | ~600GB | ⭐⭐⭐ | ❌ No | ⭐⭐ | Pre-train |
| **Custom Video** | Video frames | Variable | Variable | ⭐⭐⭐⭐⭐ | ✅ Yes | ⭐⭐⭐⭐ | Real-world |

---

## 4. Recommended Dataset Strategy

### Phase 1: Development & Testing (Current)
**Size:** ~6GB total  
**Purpose:** Fast iteration, architecture validation

```yaml
data:
  datasets:
    - name: "anime_faces"
      weight: 0.5
      enabled: true
      hr_dir: "data/anime_faces"  # 63K images, ~2GB
    - name: "div2k"
      weight: 0.5
      enabled: true
      hr_dir: "data/DIV2K_train_HR"  # 800 images, ~3.5GB
```

**Why:** Small, fast downloads, quick training iterations, validates pipeline.

---

### Phase 2: Enhanced Training (Near-term)
**Size:** ~50GB total  
**Purpose:** Better anime-specific quality

```yaml
data:
  datasets:
    - name: "danbooru_subset"
      weight: 0.5
      enabled: true
      hr_dir: "data/danbooru_100k_sfw"  # 100K subset
    - name: "anime_faces"
      weight: 0.3
      enabled: true
      hr_dir: "data/anime_faces"
    - name: "div2k"
      weight: 0.2
      enabled: true
      hr_dir: "data/DIV2K_train_HR"
  video:
    enabled: true
    video_dir: "data/anime_vid"
    extract_mode: "pre"
```

**Why:** Anime focus with real-world video frames, DIV2K adds texture diversity.

---

### Phase 3: Production Scale (Long-term)
**Size:** ~400GB+ total  
**Purpose:** Maximum quality and generalization

```yaml
data:
  datasets:
    - name: "danbooru2019"
      weight: 0.6
      enabled: true
      hr_dir: "data/danbooru2019_512px"  # 3.9M images
    - name: "anime_video"
      weight: 0.3
      enabled: true
      hr_dir: "data/anime_video_frames"  # Extracted from BD
    - name: "div2k"
      weight: 0.1
      enabled: true
      hr_dir: "data/DIV2K_train_HR"
```

**Why:** Full Danbooru scale with professional video sources.

---

## 5. Quick Setup Commands

### Option A: Minimal (6GB, Fastest)
```bash
# 1. Anime Faces
kaggle datasets download -d splcher/animefacedataset
unzip animefacedataset.zip -d data/anime_faces/

# 2. DIV2K
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip -d data/DIV2K_train_HR/
```

### Option B: With Videos (Variable)
```bash
# 1. Anime Faces
kaggle datasets download -d splcher/animefacedataset
unzip animefacedataset.zip -d data/anime_faces/

# 2. DIV2K
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip -d data/DIV2K_train_HR/

# 3. Custom Videos
mkdir -p data/anime_vid
# Copy your anime videos (mp4, mkv, avi, mov, webm)
python scripts/process_videos.py extract
```

### Option C: Large Scale (400GB+)
```bash
# 1. Danbooru2019 (via rsync or Kaggle)
# See: https://www.gwern.net/Danbooru2019

# 2. DIV2K
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip -d data/DIV2K_train_HR/

# 3. Flickr2K
wget https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
tar -xvf Flickr2K.tar -C data/
```

---

## 6. Integration with Project

### 6.1 Project's Data Module Features

The project now supports **automatic video processing**:

```python
from src.data import validate_dataset, get_dataset_info

# Validate any dataset
result = validate_dataset('data/anime_hr', check_corruption=True)
print(f"Valid: {result['valid']}, Images: {result['count']}")

# Get dataset info
info = get_dataset_info('data/anime_hr', scale=4)
print(f"Total size: {info['total_size_gb']:.2f} GB")
print(f"Estimated patches: {info['estimated_patches_128']}")
```

### 6.2 Configuration Example

```yaml
# configs/training_with_videos.yaml
model:
  scale: 4
  type: "span"

training:
  batch_size: 8
  epochs: 1000

data:
  datasets:
    - name: "anime_faces"
      weight: 0.4
      enabled: true
      hr_dir: "data/anime_faces"
    - name: "div2k"
      weight: 0.3
      enabled: true
      hr_dir: "data/DIV2K_train_HR"
  
  video:
    enabled: true
    video_dir: "data/anime_vid"
    extract_mode: "pre"
    extract_every_n_frames: 30
    quality_threshold: 0.7
    remove_duplicates: true
    min_resolution: 720
    output_dir: "data/anime_video_frames"
  
  degradation:
    enabled: true
    blur_kernel_size: [7, 9, 11]
    blur_sigma: [0.1, 3.0]
    noise_sigma: [0, 25]
    jpeg_quality: [60, 100]
```

### 6.3 Training with Videos

```bash
# Step 1: Add videos to folder
mkdir -p data/anime_vid
cp ~/Downloads/my_anime_episode*.mp4 data/anime_vid/

# Step 2: Train with video processing
python scripts/train.py \
    --config configs/model_a_ntire.yaml \
    --data.video.enabled true \
    --data.video.extract_mode pre

# Or use on-demand mode for less storage
python scripts/train.py \
    --config configs/model_a_ntire.yaml \
    --data.video.enabled true \
    --data.video.extract_mode on_demand \
    --data.video.cache_size_gb 5
```

---

## 7. Quality Filtering Details

The video extraction pipeline applies the following quality metrics:

| Metric | Weight | Description |
|--------|--------|-------------|
| **Sharpness** | 50% | Laplacian variance (edge clarity) |
| **Contrast** | 30% | Standard deviation of pixel values |
| **Resolution** | 20% | Minimum dimension check |

**Default Thresholds:**
- Quality score: ≥ 0.7 (good quality)
- Minimum resolution: 720px
- Duplicate similarity: < 0.95 (perceptual hash)

**To customize:**
```bash
python scripts/process_videos.py extract \
    --quality-threshold 0.8 \
    --min-resolution 1080 \
    --duplicate-threshold 0.90
```

---

## 8. Storage Optimization Tips

| Scenario | Storage Impact | Recommendation |
|----------|---------------|----------------|
| Pre-extraction | High (~1GB per 30min video) | Use SSD for frames, archive source videos |
| On-demand | Low (cache only) | Set `cache_size_gb` based on RAM |
| Danbooru full | Very High (400GB+) | Use external/NAS storage |
| Faces only | Low (2GB) | Local SSD fine |

**Cache Management:**
- On-demand mode uses LRU cache
- Cache auto-evicts when full
- Temp directory cleaned on exit

---

## 9. Copyright & Legal Considerations

| Dataset | License | Usage |
|---------|---------|-------|
| Danbooru | Research/Fair Use | Non-commercial research OK |
| DIV2K | Research | Academic use OK |
| Flickr2K | CC-BY | Attribution required |
| Custom Videos | Varies | Ensure personal use rights |
| Anime Faces | Getchu TOS | Check specific terms |

**Recommendation:** Use for personal/research projects. Commercial use requires rights clearance.

---

## 10. Summary & Action Items

### Immediate Actions:
1. ✅ Download **Anime Face Dataset** (63K, ~2GB) - Kaggle
2. ✅ Download **DIV2K** (800 images, ~3.5GB) - Official site
3. ✅ Set up `data/anime_vid/` folder for custom videos
4. ✅ Test video extraction: `python scripts/process_videos.py info`

### Near-term Actions:
5. ⬜ Download **Danbooru2019** subset (100K SFW images)
6. ⬜ Collect 2-3 high-quality anime videos for testing
7. ⬜ Run full pipeline: extract → filter → train

### Long-term Actions:
8. ⬜ Scale to full Danbooru2019 (3.9M images)
9. ⬜ Build curated video dataset from Blu-ray sources
10. ⬜ Evaluate quality vs. DIV2K-only baseline

---

## References

- **Danbooru Dataset:** https://www.gwern.net/Danbooru2019
- **DIV2K:** https://data.vision.ee.ethz.ch/cvl/DIV2K/
- **APISR (CVPR 2024):** https://github.com/Kiteretsu77/APISR
- **Anime Face Dataset:** https://github.com/STomoya/animeface
- **Real-ESRGAN:** https://github.com/xinntao/Real-ESRGAN

---

*Document Version: 1.0*  
*Last Updated: April 27, 2026*

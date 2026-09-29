import os

readme_path = r"E:\\python projects\\upscale_anime\\docs\\research\\anime_sr_2026\\README.md"
with open(readme_path, encoding="utf-8") as f:
    readme_content = f.read()

# Add RECOMMENDED_PATH to file table
old_readme_line = "| WSL2_DOCKER_PROBE.md | MambaIRv2 infrastructure probe: WSL2 broken, Docker works. Includes Dockerfile at tmp/mambair-build/Dockerfile. | ~6 KB |"
new_readme_line = "| WSL2_DOCKER_PROBE.md | MambaIRv2 infrastructure probe: WSL2 broken, Docker works. Includes Dockerfile at tmp/mambair-build/Dockerfile. | ~6 KB |\n| RECOMMENDED_PATH.md | **Phase 5 candidates sorted by ROI**. Rank #1 = warm-start v1 -> SRVGG no-adv (1-2 days). | ~9 KB |"
if old_readme_line in readme_content:
    readme_content = readme_content.replace(old_readme_line, new_readme_line, 1)
    print("file table updated")
else:
    print("file table anchor not found")

# Update TL;DR #5
old_tldr1 = "5. **Hybrid approach (recommended)**: lightweight CNN student for real-time path + optional SUPIR/SeeSR/OSEDiff post-pass for hero-shot quality. Use **2x+2x cascade student** to halve inference cost vs single 4x."
new_tldr1 = "5. **Hybrid approach (recommended)**: lightweight CNN student for real-time path + optional SUPIR/SeeSR/OSEDiff post-pass for hero-shot quality. Use **2x+2x cascade student** to halve inference cost vs single 4x. **Phase 5 Rank #1 path**: warm-start v1 -> SRVGG-body + no adversarial + LPIPS-heavy (1-2 days; reuses I3; see RECOMMENDED_PATH.md). Rank #1 ROI is highest among all candidates."
if old_tldr1 in readme_content:
    readme_content = readme_content.replace(old_tldr1, new_tldr1, 1)
    print("TL;DR #5 updated")
else:
    print("TL;DR #5 anchor not found")

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)
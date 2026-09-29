import os

path = r"E:\\python projects\\upscale_anime\\docs\\PROJECT_MEMORY.md"
with open(path, encoding="utf-8") as f:
    content = f.read()

# Find by unicode section symbol marker
marker = "Decision matrix at REPORT.md"
idx = content.index(marker) if marker in content else -1
if idx == -1:
    print("marker not found")
    exit()

# Find the start of this bullet (look back for "- **2026-09-03")
start = content.rfind("\n- **2026-09-03**:", 0, idx)
if start == -1:
    print("bullet start not found")
    exit()
start += 1  # skip the \n

# Find the end (next bullet)
end = content.index("\n- **", idx)
old_bullet = content[start:end]
print("Old bullet (first 200 chars):", old_bullet[:200])

new_bullet = """- **2026-09-03**: Phase 5 candidates sorted by ROI; Rank #1 = warm-start v1 -> SRVGG no-adv. New doc docs/research/anime_sr_2026/RECOMMENDED_PATH.md (~9 KB) with full ranking + Rank #1 recipe. REPORT.md section 9 decision matrix re-sorted; WSL2_DOCKER_PROBE.md Alternative section re-sorted; PROJECT_MEMORY section 4 candidates re-numbered 5#1..5#7. Rank #1: warm-start v1 RFDN -> SRVGG body + no adversarial + LPIPS weight 1.0 (1-2 days, reuses I3, expected PSNR >= 29.0 + lap_var 40-60 + latency 61 ms/frame). Rank #6 MambaIRv2 gated on Docker build completion (~30-60 min remaining).

""" + old_bullet

content = content[:start] + new_bullet + content[end:]

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("PROJECT_MEMORY §9 prepended with Rank #1 sort entry")
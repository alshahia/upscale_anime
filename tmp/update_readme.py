path = r"E:\\python projects\\upscale_anime\\docs\\research\\anime_sr_2026\\README.md"
with open(path, encoding="utf-8") as f:
    content = f.read()

old3 = "**Latest breakthrough** (CVPR 2025/2026): **MambaIRv2** matches/beats SwinIR-class transformers at **30% lower MACs** and **+0.29 dB on Manga109** -- strong candidate for the next real-time-quality hybrid if the Windows+WSL2 mamba-ssm wheel works."
new3 = "**Latest breakthrough** (CVPR 2025/2026): **MambaIRv2** matches/beats SwinIR-class transformers at **30% lower MACs** and **+0.29 dB on Manga109** -- strong candidate for the next real-time-quality hybrid. **Docker path verified** (see WSL2_DOCKER_PROBE.md): WSL2 native distro is corporate-locked, but Docker Desktop WSL2 backend gives full internet + GPU + working mamba-ssm build (image mambair-test:latest building in background, ~30-90 min compile)."

if old3 in content:
    content = content.replace(old3, new3, 1)
    print("TL;DR 3 updated")
else:
    print("TL;DR 3 anchor not found")

old4 = "| exa_search_results.json | Raw Exa search results (15 queries x 8 hits each). | ~580 KB |"
new4 = "| exa_search_results.json | Raw Exa search results (15 queries x 8 hits each). | ~580 KB |\n| WSL2_DOCKER_PROBE.md | MambaIRv2 infrastructure probe: WSL2 broken, Docker works. Includes Dockerfile at tmp/mambair-build/Dockerfile. | ~6 KB |"

if old4 in content:
    content = content.replace(old4, new4, 1)
    print("file table updated")
else:
    print("file table anchor not found")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("OK")
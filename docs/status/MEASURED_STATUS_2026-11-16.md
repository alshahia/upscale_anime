# Measured status - 2026-11-16

| Component | State | Evidence |
|---|---|---|
| GUI test suite | green | 388 passed in 38.84 s |
| GUI wheel build | green | 3.1 MB wheel, 51 images, data/registry.json |
| GUI installability | broken | imports anime_sr, declares no dependency; pyav not on PyPI |
| GUI video output | broken | video only, no audio |
| GUI failed-encode handling | broken | failed encode reported as finished |
| GUI output path guard | broken | output path can equal input path |
| Training package import | broken | SyntaxError at stage_state.py:86 (fixed 2026-11-16) |
| Root wheel build (worktree) | broken | PEP 639 classifier conflict |
| Root wheel build (HEAD) | green | built in rehearsal |
| CI on this branch | never run | triggers main/develop; branch is refactor/restructure |
| Uncommitted work | 1,792 paths | committed 2026-11-16, tag snapshot-2026-11-16 |

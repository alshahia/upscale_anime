import sys

PATH = r"E:\\python projects\\upscale_anime\\docs\\research\\anime_sr_2026\\RECOMMENDED_PATH.md"

with open(PATH, "r", encoding="utf-8", newline="") as f:
    content = f.read()

NL = "\r\n" if "\r\n" in content else "\n"
print('NL:', repr(NL))

idx = content.find('### Expected recipe')
print('Found at idx:', idx)
# Show raw bytes around idx 0-400
chunk = content[idx:idx+450]
print('Chunk len:', len(chunk))
print('Last 200 chars repr:')
print(repr(chunk[-200:]))

# Now show the bytes of the trailing region
import re
m = re.search(r'--epochs 40\r?\n```', content)
print('Match for `--epochs 40\n\`\`\``:', m)
if m:
    s = m.start()
    e = m.end()
    print('Match range:', s, e, 'len:', e-s)
    print('Surrounding 50 chars:', repr(content[e:e+50]))

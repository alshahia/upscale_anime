import json, glob
rows = []
for f in sorted(glob.glob('results/cv2_*/summary.json')):
    s = json.load(open(f))[0]
    rows.append((s['fps_achieved'], s['median_ms'], s['batch_size'], s['decode'], s['backend'], f))

rows.sort(key=lambda r: -r[0])
print(f'{"backend":10} {"batch":>5} {"decode":6}  {"fps":>6}  {"median_ms":>9}')
print('-' * 50)
for fps, ms, b, d, be, _ in rows:
    print(f'{be:10} {b:>5} {d:6}  {fps:>6.2f}  {ms:>9.1f}')
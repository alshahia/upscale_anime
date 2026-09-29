import json, glob, re, os
files = []
for root, dirs, names in os.walk('runs/distill_i3_srvgg_body'):
    for n in names:
        if n.startswith('epoch_') and n.endswith('_metrics.json'):
            files.append(os.path.join(root, n))
def epoch_num(p):
    m = re.search(r'epoch_(\d+)_', p.replace('\\','/'))
    return int(m.group(1)) if m else -1
files = sorted(set(files), key=epoch_num)
print(f"Found {len(files)} metric files\n")
print('epoch | val_psnr | val_psnr_ema | val_lap_var | shortcut_w | adv')
print('-' * 75)
best_psnr = 0; best_lap = 0; best_ema = 0
best_psnr_ep = best_lap_ep = best_ema_ep = 0
for f in files:
    d = json.load(open(f))
    print(f"{d['epoch']:5d} | {d['val_psnr']:7.3f}  | {d['val_psnr_ema']:11.3f}  | "
          f"{d['val_lap_var']:10.2f}  | {d['shortcut_weight']:.2f}       | "
          f"{d['lambda_adv']:.4f}")
    if d['val_psnr'] > best_psnr:
        best_psnr = d['val_psnr']; best_psnr_ep = d['epoch']
    if d['val_lap_var'] > best_lap:
        best_lap = d['val_lap_var']; best_lap_ep = d['epoch']
    if d['val_psnr_ema'] > best_ema:
        best_ema = d['val_psnr_ema']; best_ema_ep = d['epoch']
print(f'\nBest raw PSNR: {best_psnr:.3f} @ ep {best_psnr_ep}')
print(f'Best EMA PSNR: {best_ema:.3f} @ ep {best_ema_ep}')
print(f'Best val lap_var (in-batch): {best_lap:.2f} @ ep {best_lap_ep}')

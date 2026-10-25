"""Compare frozen before/after push evaluations and annotate the four fixed demo scenes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def compare(folder):
    before = json.loads((folder / 'before/push-summary.json').read_text())
    after = json.loads((folder / 'after/push-summary.json').read_text())
    assert before['protocol'] == after['protocol']
    rows = []
    for left, right in zip(before['groups'], after['groups']):
        assert (left['direction'], left['level_bw']) == (right['direction'], right['level_bw'])
        rows.append({
            'direction': left['direction'], 'level_bw': left['level_bw'],
            'before_trials': left['eligible_trials'], 'after_trials': right['eligible_trials'],
            'before_prepush_failures': left['prepush_failures'], 'after_prepush_failures': right['prepush_failures'],
            'before_any_failure_rate': (left['fall_count'] + left['prepush_failures']) / left['total_trials'],
            'after_any_failure_rate': (right['fall_count'] + right['prepush_failures']) / right['total_trials'],
            'before_falls': left['fall_count'], 'after_falls': right['fall_count'],
            'before_fall_rate': left['fall_rate'], 'after_fall_rate': right['fall_rate'],
            'fall_rate_change_percentage_points': 100 * (right['fall_rate'] - left['fall_rate']),
            'before_recovery_rate': left['recovery_rate'], 'after_recovery_rate': right['recovery_rate'],
            'before_recovered_count': left['recovered_count'], 'after_recovered_count': right['recovered_count'],
            'before_median_recovery_s': left['median_recovery_s_among_recovered'],
            'after_median_recovery_s': right['median_recovery_s_among_recovered'],
            'before_mean_speed_error_m_s': left['mean_postpush_speed_error_before_fall_m_s'],
            'after_mean_speed_error_m_s': right['mean_postpush_speed_error_before_fall_m_s'],
        })
    walking = [json.loads((folder / f'walking-after-seed{seed}.json').read_text()) for seed in (42, 43)]
    report = {
        'training': json.loads((folder / 'training-summary.json').read_text()),
        'corrected_curriculum': json.loads((folder / 'corrected/curriculum.json').read_text()),
        'before_checkpoint': before['checkpoint'], 'after_checkpoint': after['checkpoint'],
        'before_trials': before['total_trials'], 'after_trials': after['total_trials'],
        'protocol_unchanged': True, 'comparison': rows,
        'walking_after': [{'seed': run['seed'], 'trials': run['trials'],
                           'success_rate': run['success_rate'], 'no_fall_rate': run['no_fall_rate'],
                           'groups': run['groups']} for run in walking],
        'walking_after_passed': all(run['stable_forward_walking_passed'] for run in walking),
    }
    (folder / 'robust-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    with (folder / 'robust-comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    directions = ['forward', 'backward', 'left', 'right']
    levels = [.25, .5, 1., 2.]
    matrix_before = np.array([[next(row for row in rows if row['direction'] == direction and row['level_bw'] == level)['before_fall_rate']
                               for level in levels] for direction in directions]) * 100
    matrix_after = np.array([[next(row for row in rows if row['direction'] == direction and row['level_bw'] == level)['after_fall_rate']
                              for level in levels] for direction in directions]) * 100
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True)
    for axis, matrix, title in zip(axes, [matrix_before, matrix_after, matrix_after - matrix_before],
                                    ['Before: fall rate (%)', 'After: fall rate (%)', 'Change (percentage points; lower is better)']):
        is_delta = axis is axes[-1]
        image = axis.imshow(matrix, cmap='RdBu_r' if is_delta else 'OrRd', vmin=-100 if is_delta else 0, vmax=100)
        axis.set_xticks(range(4), ['0.25', '0.5', '1.0', '2.0'])
        axis.set_yticks(range(4), directions)
        axis.set_xlabel('Force / body weight (0.2 s pulse)')
        axis.set_title(title, fontsize=10)
        for y in range(4):
            for x in range(4):
                axis.text(x, y, f'{matrix[y,x]:.1f}', ha='center', va='center',
                          color='white' if abs(matrix[y,x]) >= 65 else 'black')
        fig.colorbar(image, ax=axis, shrink=.8)
    fig.suptitle('Go2: direction-weighted force curriculum | identical protocol | 64 trials per condition')
    fig.savefig(folder / 'robust-comparison.png', dpi=180)
    plt.close(fig)
    print('ROBUST_COMPARISON=' + json.dumps({'walking_after_passed': report['walking_after_passed'],
                                           'one_bw': [row for row in rows if row['level_bw'] == 1.0]}), flush=True)


def video(folder):
    output = folder / 'go2-robust-before-after.mp4'
    writer = imageio.get_writer(output, fps=50, codec='libx264', quality=7, macro_block_size=16)
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
    scenes, previews = [], []
    for direction in ('backward', 'right'):
        for stage in ('before', 'after'):
            tag = f'{stage}_{direction}'
            row = json.loads((folder / 'demo' / f'{tag}.json').read_text())['rows'][0]
            trace = np.load(folder / 'demo' / f'{tag}-trace.npz')['trace'][:, 0]
            source = max((folder / 'demo' / f'video-raw-{tag}').glob('*.mp4'), key=lambda p: p.stat().st_mtime)
            reader = imageio.get_reader(source)
            assert reader.count_frames() == 750
            for index, frame in enumerate(reader):
                image = Image.fromarray(frame).convert('RGB')
                draw = ImageDraw.Draw(image)
                time = index / 50
                pushing = 5 <= time < 5.2
                status = 'PUSH ACTIVE' if pushing else ('BEFORE PUSH' if time < 5 else 'POST PUSH')
                if row['fall_time_s'] is not None and time >= row['fall_time_s']:
                    status = 'FALL (auto-reset is not recovery)'
                elif row['recovery_time_s'] is not None and time >= 5.2 + row['recovery_time_s'] + .5:
                    status = f"RECOVERED in {row['recovery_time_s']:.2f}s"
                draw.rectangle((0, 0, 1280, 94), fill='#92232a' if pushing else '#14212e')
                draw.text((18, 8), f"{stage.upper()} TRAINING | {direction.upper()} | 1 BW = {row['nominal_force_N']:.1f} N for 0.2 s", font=font, fill='white')
                error = np.linalg.norm(trace[index, :2] - [.5, 0.])
                draw.text((18, 43), f't={time:.2f}s | command 0.5 m/s | speed error {error:.2f} m/s | {status}', font=font, fill='white')
                if pushing:
                    draw.rectangle((2, 96, 1277, 717), outline='#ed3743', width=6)
                writer.append_data(np.asarray(image))
                if index in (265, 350):
                    previews.append(image.copy())
            reader.close()
            scenes.append({'tag': tag, 'trial': row, 'source': str(source)})
    writer.close()
    reader = imageio.get_reader(output)
    metadata = reader.get_meta_data()
    assert reader.count_frames() == 3000 and metadata['size'] == (1280, 720)
    reader.close()
    preview = Image.new('RGB', (1280, 1440), '#14212e')
    for slot, image in enumerate(previews):
        image.thumbnail((640, 360))
        preview.paste(image, ((slot % 2) * 640, (slot // 2) * 360))
    preview.save(folder / 'robust-video-preview.png')
    report = {'source': str(output), 'frames': 3000, 'duration_s': 60, 'fps': 50,
              'resolution': [1280, 720], 'scenes': scenes,
              'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    (folder / 'robust-video-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print('ROBUST_VIDEO_COMPLETE=' + json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['compare', 'video'])
    parser.add_argument('--folder', required=True)
    args = parser.parse_args()
    if args.action == 'compare':
        compare(Path(args.folder))
    else:
        video(Path(args.folder))

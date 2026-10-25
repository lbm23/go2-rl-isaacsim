"""Aggregate push experiments, plot measured results, and assemble annotated real videos."""
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

from push_protocol import (DIRECTIONS, HORIZON, LEVELS, PUSH_DURATION, PUSH_START,
                           TARGET_SPEED, aggregate)


def summarize(folder, checkpoint):
    runs = [json.loads((folder / f'seed{seed}.json').read_text()) for seed in (42, 43)]
    rows = [row for run in runs for row in run['rows']]
    groups = aggregate(rows)
    control = groups[0]
    pushed = groups[1:]
    measured_response = max(group['mean_peak_speed_error_before_fall_m_s'] for group in pushed) > control['mean_peak_speed_error_before_fall_m_s'] + 0.20
    assert measured_response, 'No measurable response to the strongest forces; investigate force injection'
    recoverable = [group for group in pushed if group['recovery_rate'] >= 0.9 and group['fall_rate'] <= 0.1]
    left = [group for group in recoverable if group['direction'] == 'left']
    recovery = max(left or recoverable or pushed, key=lambda group: (group['recovery_rate'], group['level_bw']))
    challenge = max(pushed, key=lambda group: (group['fall_rate'], group['level_bw'], group['direction'] == 'left'))
    demo = [{'tag': tag, 'direction': item['direction'], 'level_bw': item['level_bw'], 'seed': 42}
            for tag, item in [('recovery', recovery), ('challenge', challenge)]]
    summary = {
        'checkpoint': checkpoint, 'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'seeds': [42, 43], 'total_trials': len(rows),
        'pushed_trials': sum(row['level_bw'] > 0 for row in rows),
        'control_trials': sum(row['level_bw'] == 0 for row in rows),
        'protocol': {
            'target_forward_speed_m_s': TARGET_SPEED, 'push_onset_s': PUSH_START,
            'push_duration_s': PUSH_DURATION, 'horizon_s': HORIZON,
            'directions': [item[0] for item in DIRECTIONS], 'strengths_in_body_weight': LEVELS,
            'force': 'F=level*actual_total_robot_mass*9.81, applied at base CoM, horizontal direction frozen from heading at onset',
            'recovery': 'After release: planar speed error <=0.15m/s, abs yaw rate<=0.3rad/s, tilt<=15deg, base height>=0.28m continuously for0.5s; no later fall',
            'fall': 'Official termination OR tilt>60deg or base height<0.18m sustained for0.2s; prepush falls excluded and counted separately',
            'velocity_error': 'Body-frame planar velocity vector norm error; postpush mean uses samples before first fall only',
            'censoring': 'Recovery is null for falls or no recovery within9.8s; medians condition on recovered trials',
            'observation_noise': False, 'other_pushes': False,
            'randomized': 'Material, mass and center of mass; independent trial samples, not paired across conditions',
            'initial_state': 'Standard joint pose, zero initial velocity and zero reset yaw offset from authored orientation; heading measured again at push onset',
        },
        'force_buffer_max_norm_error_N': max(run['force_buffer_max_norm_error_N'] for run in runs),
        'measured_response_detected': bool(measured_response), 'groups': groups, 'demo_selection': demo,
    }
    (folder / 'push-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (folder / 'push-trials.json').write_text(json.dumps(rows, indent=2) + '\n')
    with (folder / 'push-results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    print('PUSH_AGGREGATED=' + json.dumps(summary), flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    configurations = [('fall_rate', 'Fall rate (%)', 100, 'OrRd'),
                      ('median_recovery_s_among_recovered', 'Median recovery (s; recovered only)', 1, 'YlOrRd'),
                      ('mean_postpush_speed_error_before_fall_m_s', 'Mean post-push speed error (m/s; before fall)', 1, 'YlGnBu')]
    for axis, (key, title, scale, color) in zip(axes, configurations):
        matrix = np.array([[next(group for group in pushed if group['direction'] == direction and group['level_bw'] == level)[key]
                            for level in LEVELS] for direction, _ in DIRECTIONS], dtype=float) * scale
        rendered = axis.imshow(np.ma.masked_invalid(matrix), cmap=color, aspect='auto', vmin=0)
        axis.set_xticks(range(4), ['0.25', '0.5', '1.0', '2.0'])
        axis.set_yticks(range(4), [direction for direction, _ in DIRECTIONS])
        axis.set_xlabel('Force / body weight (0.2 s pulse)')
        axis.set_title(title, fontsize=10)
        for y in range(4):
            for x in range(4):
                label = 'N/A' if not np.isfinite(matrix[y, x]) else (f'{matrix[y,x]:.1f}' if scale == 100 else f'{matrix[y,x]:.2f}')
                axis.text(x, y, label,
                          ha='center', va='center', color='black', fontsize=10)
        fig.colorbar(rendered, ax=axis, shrink=0.8)
    fig.suptitle('Go2 PPO push test | 64 trials / cell | 0.5 m/s | 15 s / trial', fontsize=12)
    fig.savefig(folder / 'push-results.png', dpi=180)
    plt.close(fig)


def assemble_video(folder):
    summary = json.loads((folder / 'push-summary.json').read_text())
    output = folder / 'go2-push-demo.mp4'
    writer = imageio.get_writer(output, fps=50, codec='libx264', quality=7, macro_block_size=16,
                                ffmpeg_params=['-pix_fmt', 'yuv420p'])
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 22) if Path(font_path).exists() else ImageFont.load_default()
    previews = []
    video_runs = []
    for selection in summary['demo_selection']:
        tag = selection['tag']
        result = json.loads((folder / f'{tag}.json').read_text())
        row = result['rows'][0]
        trace = np.load(folder / f'{tag}-trace.npz')['trace'][:, 0]
        source = max((folder / f'video-raw-{tag}').glob('*.mp4'), key=lambda path: path.stat().st_mtime)
        reader = imageio.get_reader(source)
        assert reader.count_frames() == 750
        for index, frame in enumerate(reader):
            image = Image.fromarray(frame).convert('RGB')
            draw = ImageDraw.Draw(image)
            time = index / 50
            active = PUSH_START <= time < PUSH_START + PUSH_DURATION
            status = 'PUSH ACTIVE' if active else ('BEFORE PUSH' if time < PUSH_START else 'POST PUSH')
            if row['fall_time_s'] is not None and time >= row['fall_time_s']:
                status = 'FALL DETECTED (later auto-reset is not recovery)'
            elif row['recovery_time_s'] is not None and time >= PUSH_START + PUSH_DURATION + row['recovery_time_s'] + 0.5:
                status = f"RECOVERED in {row['recovery_time_s']:.2f}s"
            draw.rectangle((0, 0, 1280, 94), fill='#9c2026' if active else '#14212e')
            draw.text((18, 9), f"{tag.upper()} SCENARIO | {row['direction'].upper()} | {row['level_bw']:.2f} BW = {row['nominal_force_N']:.1f} N for 0.20 s", font=font, fill='white')
            error = float(np.linalg.norm(trace[index, :2] - [0.5, 0.0]))
            draw.text((18, 44), f't={time:5.2f}s | command 0.50 m/s | speed error {error:.2f} m/s | {status}', font=font, fill='white')
            if active:
                draw.rectangle((2, 96, 1277, 717), outline='#ed3743', width=7)
            writer.append_data(np.asarray(image))
            if index in (230, 259, 350):
                previews.append((image.copy(), f"{tag}: t={time:.2f}s"))
        reader.close()
        video_runs.append({'tag': tag, 'source': str(source), 'trial': row})
    writer.close()
    reader = imageio.get_reader(output)
    metadata = reader.get_meta_data()
    frame_count = reader.count_frames()
    assert frame_count == 1500 and metadata['size'] == (1280, 720)
    reader.close()
    sheet = Image.new('RGB', (1280, 1140), '#17202b')
    for slot, (image, _) in enumerate(previews):
        image.thumbnail((640, 375))
        sheet.paste(image, ((slot % 2) * 640, (slot // 2) * 380))
    sheet.save(folder / 'push-preview.png')
    video_report = {'source': str(output), 'frames': frame_count, 'fps': 50, 'duration_s': 30,
                    'resolution': metadata['size'], 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                    'scenes': video_runs}
    (folder / 'push-video-summary.json').write_text(json.dumps(video_report, indent=2) + '\n')
    # Plot time series from actual demo rollouts; falls and the pulse remain explicit.
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    for axis, run in zip(axes, video_runs):
        trace = np.load(folder / f"{run['tag']}-trace.npz")['trace'][:, 0]
        time = (np.arange(len(trace)) + 1) * 0.02
        error = np.linalg.norm(trace[:, :2] - [0.5, 0], axis=-1)
        fall_time = run['trial']['fall_time_s']
        before_fall = np.ones(len(time), dtype=bool) if fall_time is None else time < fall_time
        axis.plot(time[before_fall], error[before_fall], label='planar speed error (m/s)')
        if fall_time is not None:
            axis.plot(time[~before_fall], error[~before_fall], color='gray', linestyle=':',
                      label='auto-reset data (excluded; not recovery)')
            axis.axvspan(fall_time, HORIZON, color='gray', alpha=0.08)
        axis.axhline(0.15, color='green', linestyle='--', label='recovery speed threshold')
        axis.axvspan(5, 5.2, color='red', alpha=0.3, label='force pulse')
        if run['trial']['fall_time_s'] is not None:
            axis.axvline(run['trial']['fall_time_s'], color='black', linestyle=':', label='fall; later reset excluded')
        axis.set_title(f"{run['tag']}: {run['trial']['direction']}, {run['trial']['level_bw']} BW")
        axis.set_ylabel('Speed error (m/s)')
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel('Simulation time (s)')
    fig.savefig(folder / 'push-demo-timeseries.png', dpi=170)
    plt.close(fig)
    print('PUSH_VIDEO_ASSEMBLED=' + json.dumps(video_report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['aggregate', 'video'])
    parser.add_argument('--folder', required=True)
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    if args.action == 'aggregate':
        summarize(Path(args.folder), args.checkpoint)
    else:
        assemble_video(Path(args.folder))

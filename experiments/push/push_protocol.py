"""Push protocol and metrics, kept separate from simulator setup for testing."""
import math

import numpy as np

DIRECTIONS = [('forward', (1.0, 0.0)), ('backward', (-1.0, 0.0)),
              ('left', (0.0, 1.0)), ('right', (0.0, -1.0))]
LEVELS = [0.25, 0.5, 1.0, 2.0]
PUSH_START = 5.0
PUSH_DURATION = 0.2
HORIZON = 15.0
TARGET_SPEED = 0.5


def conditions():
    result = [{'direction': 'control', 'level_bw': 0.0, 'unit_xy': (0.0, 0.0)}]
    for direction, vector in DIRECTIONS:
        for level in LEVELS:
            result.append({'direction': direction, 'level_bw': level, 'unit_xy': vector})
    return result


def world_directions(yaw, unit_xy):
    """Freeze a horizontal body-heading direction in the world frame at push onset."""
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.stack([cosine * unit_xy[:, 0] - sine * unit_xy[:, 1],
                     sine * unit_xy[:, 0] + cosine * unit_xy[:, 1], np.zeros_like(yaw)], axis=-1)


def first_stable_window(mask, dt, hold_seconds=0.5):
    length = round(hold_seconds / dt)
    if len(mask) < length:
        return None
    starts = np.flatnonzero(np.convolve(np.asarray(mask, dtype=np.int32), np.ones(length, dtype=np.int32), 'valid') == length)
    return float(starts[0] * dt) if len(starts) else None


def first_fall_index(terminated, gravity_z, height, dt):
    """Official termination OR a severe tilt/low height sustained for 0.2 seconds."""
    severe = (gravity_z > -math.cos(math.radians(60))) | (height < 0.18)
    duration = round(0.2 / dt)
    sustained = np.convolve(severe.astype(np.int32), np.ones(duration, dtype=np.int32), 'full')[:len(severe)] >= duration
    indices = np.flatnonzero(terminated.astype(bool) | sustained)
    return int(indices[0]) if len(indices) else len(terminated)


def trial_metrics(trace, dt, specification, seed, trial_id, mass, force_world):
    # Columns: vx_body, vy_body, yaw_rate, gravity_z_body, base_height_world,
    # official_termination, commanded_external_force_norm_N.
    onset = round(PUSH_START / dt)
    release = round((PUSH_START + PUSH_DURATION) / dt)
    error = np.linalg.norm(trace[:, :2] - np.array([TARGET_SPEED, 0.0]), axis=-1)
    fall = first_fall_index(trace[:, 5], trace[:, 3], trace[:, 4], dt)
    eligible = fall >= onset
    fell_after_push = eligible and fall < len(trace)
    stop = min(fall, len(trace))
    stable = ((error[release:stop] <= 0.15) & (np.abs(trace[release:stop, 2]) <= 0.30)
              & (trace[release:stop, 3] <= -math.cos(math.radians(15)))
              & (trace[release:stop, 4] >= 0.28))
    recovery = first_stable_window(stable, dt) if eligible and not fell_after_push else None
    post_error = error[release:stop]
    event_error = error[onset:stop]
    return {
        'seed': seed, 'trial_id': trial_id, 'direction': specification['direction'],
        'level_bw': specification['level_bw'], 'eligible_at_push': bool(eligible),
        'prepush_fall': bool(not eligible), 'fell_after_push': bool(fell_after_push),
        'fall_time_s': float((fall + 1) * dt) if fall < len(trace) else None,
        'mass_kg': float(mass), 'nominal_force_N': float(mass * 9.81 * specification['level_bw']),
        'force_world_N': [float(value) for value in force_world],
        'commanded_impulse_Ns': float(trace[:, 6].sum() * dt),
        'prepush_mean_speed_error_m_s': float(error[round(4.0/dt):onset].mean()),
        'postpush_mean_speed_error_before_fall_m_s': float(post_error.mean()) if len(post_error) else None,
        'peak_speed_error_before_fall_m_s': float(event_error.max()) if len(event_error) else None,
        'recovery_time_s': recovery, 'recovered_without_later_fall': recovery is not None,
        'recovery_observation_limit_s': HORIZON - PUSH_START - PUSH_DURATION,
    }


def aggregate(rows):
    result = []
    for specification in conditions():
        selected = [row for row in rows if row['direction'] == specification['direction']
                    and row['level_bw'] == specification['level_bw']]
        eligible = [row for row in selected if row['eligible_at_push']]
        recovered = [row['recovery_time_s'] for row in eligible if row['recovery_time_s'] is not None]
        def average(key):
            values = [row[key] for row in eligible if row[key] is not None]
            return float(np.mean(values)) if values else None
        result.append({
            'direction': specification['direction'], 'level_bw': specification['level_bw'],
            'total_trials': len(selected), 'eligible_trials': len(eligible),
            'prepush_failures': len(selected) - len(eligible),
            'fall_count': sum(row['fell_after_push'] for row in eligible),
            'fall_rate': sum(row['fell_after_push'] for row in eligible) / len(eligible) if eligible else None,
            'recovered_count': len(recovered),
            'recovery_rate': len(recovered) / len(eligible) if eligible else None,
            'median_recovery_s_among_recovered': float(np.median(recovered)) if recovered else None,
            'p90_recovery_s_among_recovered': float(np.percentile(recovered, 90)) if recovered else None,
            'mean_postpush_speed_error_before_fall_m_s': average('postpush_mean_speed_error_before_fall_m_s'),
            'mean_peak_speed_error_before_fall_m_s': average('peak_speed_error_before_fall_m_s'),
            'mean_prepush_speed_error_m_s': average('prepush_mean_speed_error_m_s'),
            'mean_force_N': average('nominal_force_N'),
            'minimum_force_N': min((row['nominal_force_N'] for row in eligible), default=None),
            'maximum_force_N': max((row['nominal_force_N'] for row in eligible), default=None),
        })
    return result

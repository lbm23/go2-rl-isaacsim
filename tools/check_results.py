"""Recompute recorded headline counts, exact p-values and telemetry, without simulation."""
import gzip
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/data'


def load(name):
    path = DATA/name
    raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return json.loads(raw)


def sign_flip(values):
    observed = abs(sum(values))
    return sum(abs(sum(s*v for s,v in zip(signs,values))) >= observed-1e-12
               for signs in itertools.product((-1,1), repeat=len(values))) / 2**len(values)


def holm(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.]*len(values)
    previous = 0.
    for rank,index in enumerate(order):
        previous = max(previous, min(1., (len(values)-rank)*values[index]))
        adjusted[index] = previous
    return adjusted


def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10), (a,b)


def main():
    walk = load('walking-summary.json')['evaluation']
    assert walk['successful_trials'] == walk['trials'] == 192
    left = load('left2-heldout-summary.json')
    rows = [r for r in load('left2-heldout-seed-results.json') if r['direction']=='left' and r['level_bw']==2]
    assert len(rows)==16 and len(set(r['seed'] for r in rows))==16
    primary = left['primary']
    for stage, expected in [('before',551),('after',167)]:
        assert sum(r[stage]['falls'] for r in rows)==expected==primary[stage]['falls']
        assert sum(r[stage]['eligible'] for r in rows)==1024
        trials = [r for r in load(f'left2-heldout-{stage}-trials.json.gz') if r['direction']=='left' and r['level_bw']==2]
        assert len(trials)==1024
        assert sum(r['fell_after_push'] for r in trials if r['eligible_at_push'])==expected
    differences = [r['after_fall_rate']-r['before_fall_rate'] for r in rows]
    close(statistics.mean(differences)*100, primary['fall_change_pp'])
    close(sign_flip(differences), primary['fall_p_exact_two_sided'])

    ablation = load('ablation-summary.json.gz')
    assert ablation['confirmatory_complete'] and ablation['actual_matrix_trials']==435200
    p_values = []
    for contrast in ablation['contrasts']:
        after,before = contrast['contrast'].split('-')
        values = [(ablation['policies'][f'{after}{seed}']['metrics']['left:2']['fall_rate']-
                   ablation['policies'][f'{before}{seed}']['metrics']['left:2']['fall_rate'])*100
                  for seed in ablation['training_seeds']]
        assert len(values)==8
        for a,b in zip(values,contrast['paired_changes_pp']): close(a,b)
        close(statistics.mean(values),contrast['change_pp'])
        p_values.append(sign_flip(values))
        close(p_values[-1],contrast['p_exact'])
        assert not contrast['reliable_practical_improvement']
    for actual,contrast in zip(holm(p_values),ablation['contrasts']): close(actual,contrast['p_holm'])

    sim = load('straight-results.json')
    assert all(sim['policy_actor_tensor_matches'].values())
    validations = sim['validation_30s']
    assert len(validations)==3 and all(row['no_fall'] for row in validations)
    half = next(row for row in validations if row['speed_command']==.5)
    close(half['max_abs_lateral_m'], .041712772101163864)
    close(half['max_abs_yaw_deg'], 1.8234533054746174)
    assert not next(row for row in validations if row['speed_command']==.3)['within_3deg_heading']
    print(json.dumps(dict(walking_passes=192,left2_falls=[551,167],left2_difference_pp=primary['fall_change_pp'],
        left2_exact_p=primary['fall_p_exact_two_sided'],ablation_trials=435200,
        ablation_holm_p=holm(p_values),deployment_half_m_s_max_lateral_cm=100*half['max_abs_lateral_m']),indent=2))
    print('PASS: stored raw outcomes, seed-level estimates, exact tests and reported telemetry agree')


if __name__=='__main__':
    main()

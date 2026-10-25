"""Inference clusters on fresh paired seeds, not individual environments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sign_flip_p(differences):
    values = np.asarray(differences, dtype=float)
    assert values.ndim == 1 and 0 < len(values) <= 20
    observed = abs(values.sum())
    count = 0
    for start in range(0, 2 ** len(values), 4096):
        ids = np.arange(start, min(start + 4096, 2 ** len(values)), dtype=np.uint32)
        signs = 2 * ((ids[:, None] >> np.arange(len(values))) & 1).astype(float) - 1
        count += np.count_nonzero(np.abs(signs @ values) >= observed - 1e-12)
    return count / (2 ** len(values))


def paired_ci(differences, indices):
    values = np.asarray(differences, dtype=float)
    assert len(values) == indices.shape[1]
    return np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()


def holm(pvalues):
    order = np.argsort(pvalues)
    result = np.zeros(len(pvalues))
    last = 0.
    for rank, index in enumerate(order):
        last = max(last, min(1., (len(order) - rank) * pvalues[index]))
        result[index] = last
    return result.tolist()


def completed_seed_pairs(seeds, progress, exists):
    successful={(r['seed'],r['stage']) for r in progress if r['exit_code']==0}
    return [seed for seed in seeds if all((seed,stage) in successful and exists(seed,stage)
                                         for stage in ('before','after'))]


def describe(rows):
    eligible = [r for r in rows if r['eligible_at_push']]
    recovered = [r['recovery_time_s'] for r in eligible if r['recovery_time_s'] is not None]
    errors = [r['postpush_mean_speed_error_before_fall_m_s'] for r in eligible
              if r['postpush_mean_speed_error_before_fall_m_s'] is not None]
    assert eligible, 'No eligible trials in a seed-condition group'
    return dict(total=len(rows), eligible=len(eligible), prepush_failures=len(rows)-len(eligible),
                falls=sum(r['fell_after_push'] for r in eligible),
                fall_rate=float(np.mean([r['fell_after_push'] for r in eligible])),
                recovery_rate=len(recovered)/len(eligible), recovered=len(recovered),
                median_recovery_s=float(np.median(recovered)) if recovered else None,
                mean_speed_error_m_s=float(np.mean(errors)) if errors else None,
                capped_recovery_mean_s=float(np.mean([r['recovery_time_s'] if r['recovery_time_s'] is not None
                    else r['recovery_observation_limit_s'] for r in eligible])))


def analyze(folder):
    plan = json.loads((folder / 'heldout-plan.json').read_text())
    root = Path(__import__('os').environ.get('GO2_PROJECT_ROOT', Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(root / 'experiments/push'))
    from push_protocol import conditions
    progress=json.loads((folder/'heldout-progress.json').read_text())
    complete=completed_seed_pairs(plan['seeds'],progress,
        lambda seed,stage:(folder/stage/f'seed{seed}.json').exists())
    incomplete = [s for s in plan['seeds'] if s not in complete]
    assert complete, 'No complete paired seeds'
    rng = np.random.default_rng(plan['resampling_seed'])
    indices = rng.integers(0, len(complete), size=(20000, len(complete)))
    raw = {'before': [], 'after': []}
    runs = {}
    audits = []
    for seed in complete:
        for stage in raw:
            run = json.loads((folder / stage / f'seed{seed}.json').read_text())
            assert run['seed'] == seed
            assert len(run['rows']) == 17 * plan['samples_per_condition_per_seed']
            assert run['force_buffer_max_norm_error_N'] < .01
            assert run['checkpoint'] == str(root / plan['frozen_models'][stage]['relative_path'])
            runs[seed, stage] = run['rows']
            raw[stage].extend(run['rows'])
        a, b = runs[seed, 'before'], runs[seed, 'after']
        assert all((x['trial_id'],x['direction'],x['level_bw']) ==
                   (y['trial_id'],y['direction'],y['level_bw']) for x,y in zip(a,b))
        audits.append(dict(seed=seed, maximum_matched_mass_difference_kg=max(
            abs(x['mass_kg']-y['mass_kg']) for x,y in zip(a,b))))
    output = []
    seed_rows = []
    for spec in conditions():
        def select(rows):
            return [r for r in rows if r['direction']==spec['direction'] and r['level_bw']==spec['level_bw']]
        pooled = {stage:describe(select(rows)) for stage,rows in raw.items()}
        paired = {stage:[describe(select(runs[s,stage])) for s in complete] for stage in raw}
        record = dict(**spec, before=pooled['before'], after=pooled['after'])
        fall_delta = np.array([b['fall_rate']-a['fall_rate'] for a,b in zip(paired['before'],paired['after'])])
        record['fall_change_pp'] = float(100*fall_delta.mean())
        record['fall_change_ci95_pp'] = [100*x for x in paired_ci(fall_delta, indices)]
        record['fall_p_exact_two_sided'] = sign_flip_p(fall_delta)
        record['seeds_worse'] = int(sum(fall_delta > 0))
        record['seeds_better'] = int(sum(fall_delta < 0))
        record['seeds_tied'] = int(sum(fall_delta == 0))
        for field in ['recovery_rate','mean_speed_error_m_s','capped_recovery_mean_s']:
            diffs = [b[field]-a[field] for a,b in zip(paired['before'],paired['after'])]
            record[field+'_change'] = float(np.mean(diffs))
            record[field+'_change_ci95'] = paired_ci(diffs, indices)
        record['primary'] = spec['direction']=='left' and spec['level_bw']==2.
        for i,seed in enumerate(complete):
            seed_rows.append(dict(seed=seed,**spec,before_fall_rate=paired['before'][i]['fall_rate'],
                after_fall_rate=paired['after'][i]['fall_rate'],fall_change_pp=float(fall_delta[i]*100),
                before=paired['before'][i],after=paired['after'][i]))
        output.append(record)
    secondary = [r for r in output if r['level_bw']>0 and not r['primary']]
    assert len(secondary)==15
    for record, p in zip(secondary,holm([r['fall_p_exact_two_sided'] for r in secondary])):
        record['fall_p_holm_secondary']=p
    report = dict(plan_sha256=hashlib.sha256((folder/'heldout-plan.json').read_bytes()).hexdigest(),
        completed_seeds=complete,incomplete_seeds=incomplete,independent_replicates=len(complete),
        inferential_minimum_met=len(complete)>=12,planned_trials_per_model=len(plan['seeds'])*17*plan['samples_per_condition_per_seed'],
        trials_per_model=len(raw['before']),before_checkpoint=plan['frozen_models']['before'],
        after_checkpoint=plan['frozen_models']['after'],seed_pair_audits=audits,
        results=output,primary=next(r for r in output if r['primary']),
        notes=['Primary p-value is exact conditional on paired-seed sign-exchangeability under the null.',
               'Bootstrap resamples whole paired seeds. Zero-width intervals at all-zero observed failures are not proof of zero risk.',
               'Recovery medians condition on success; capped recovery assigns 9.8 s to every failure or non-recovery, a descriptive failure-aware score, not estimated eventual recovery time.',
               'Speed error stops at first fall and therefore has unequal exposure and survivor selection.',
               'Fresh seeds were held out within available project records; there are no repeated training runs, and conclusions concern these two frozen policies only.'])
    (folder/'heldout-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (folder/'heldout-seed-results.json').write_text(json.dumps(seed_rows,indent=2)+'\n')
    for stage,rows in raw.items():
        (folder/f'heldout-{stage}-trials.json').write_text(json.dumps(rows,indent=2)+'\n')
    flat=[]
    for r in output:
        flat.append(dict(direction=r['direction'],level_bw=r['level_bw'],
            before_trials=r['before']['eligible'],after_trials=r['after']['eligible'],
            before_falls=r['before']['falls'],after_falls=r['after']['falls'],
            before_fall_rate=r['before']['fall_rate'],after_fall_rate=r['after']['fall_rate'],
            fall_change_pp=r['fall_change_pp'],ci95_low_pp=r['fall_change_ci95_pp'][0],
            ci95_high_pp=r['fall_change_ci95_pp'][1],p_exact=r['fall_p_exact_two_sided'],
            p_holm=r.get('fall_p_holm_secondary'),
            before_recovery_rate=r['before']['recovery_rate'],after_recovery_rate=r['after']['recovery_rate'],
            before_median_recovery_s=r['before']['median_recovery_s'],after_median_recovery_s=r['after']['median_recovery_s'],
            before_mean_speed_error=r['before']['mean_speed_error_m_s'],after_mean_speed_error=r['after']['mean_speed_error_m_s']))
    (folder/'heldout-table.json').write_text(json.dumps(flat,indent=2)+'\n')
    with (folder/'heldout-table.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in output if r['level_bw'] in (1.,2.)]
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for i,r in enumerate(selected):
        lo,hi=r['fall_change_ci95_pp']; x=r['fall_change_pp']
        axes[0].errorbar(x,i,xerr=[[max(0,x-lo)],[max(0,hi-x)]],fmt='o',color='#c0392b' if r['primary'] else '#2471a3')
    axes[0].set_yticks(range(len(selected)),[f"{r['direction']} {r['level_bw']:g} BW" for r in selected])
    axes[0].axvline(0,color='gray',linestyle='--');axes[0].set_xlabel('After - before fall rate (percentage points), 95% seed bootstrap CI')
    axes[0].set_title('Fresh-seed effects (negative = improvement)')
    left=[r for r in seed_rows if r['direction']=='left' and r['level_bw']==2.]
    axes[1].bar(range(len(left)),[r['fall_change_pp'] for r in left],color=['#c0392b' if r['fall_change_pp']>0 else '#2471a3' for r in left])
    axes[1].axhline(0,color='gray');axes[1].set_xticks(range(len(left)),[str(r['seed'])[-2:] for r in left],rotation=90)
    axes[1].set_xlabel('Seed suffix (920001-920016)');axes[1].set_ylabel('Fall-rate change (percentage points)')
    axes[1].set_title('Primary: 2 BW left, each paired seed')
    fig.suptitle(f"Go2 held-out replication: {len(complete)} paired seeds, 64 trials/condition/seed/model")
    fig.savefig(folder/'heldout-effects.png',dpi=160);plt.close(fig)
    print('HELDOUT_PRIMARY='+json.dumps(report['primary']),flush=True)
    print('HELDOUT_COUNTS='+json.dumps(dict(seeds=len(complete),trials_per_model=len(raw['before']),incomplete=incomplete)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,required=True)
    analyze(parser.parse_args().folder)

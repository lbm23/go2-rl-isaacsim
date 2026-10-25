"""End-to-end analysis test using artificial outcomes, never simulator/evaluation data."""
import json
from pathlib import Path
import subprocess
import sys

folder=Path(sys.argv[1]);mock=folder/'analysis-selftest';mock.mkdir(exist_ok=False)
seeds=list(range(8));evaluation=list(range(16))
jobs=[dict(id=f'{g}{s}') for g in 'ABC' for s in seeds]
(mock/'spec.json').write_text(json.dumps(dict(training_seeds=seeds,evaluation_seeds=evaluation,jobs=jobs)))
conditions=[('control',0.)]+[(d,l) for d in ['forward','backward','left','right'] for l in [.25,.5,1.,2.]]
for label in ['baseline']+[j['id'] for j in jobs]:
    dest=mock/'evaluations'/label;dest.mkdir(parents=True)
    (dest/'policy.json').write_text(json.dumps(dict(checkpoint='ARTIFICIAL_DATA_ONLY')))
    (dest/'evaluation-complete.json').write_text('{}')
    rows=[]
    for direction,level in conditions:
        for index in range(64):
            falls=direction=='left' and level==2 and index<({'A':8,'B':4,'C':0}.get(label[0],8))
            rows.append(dict(trial_id=len(rows),direction=direction,level_bw=level,
                eligible_at_push=True,fell_after_push=falls,mass_kg=13.,
                recovery_time_s=None if falls else 1.,postpush_mean_speed_error_before_fall_m_s=.1))
    for seed in evaluation:
        (dest/f'seed{seed}.json').write_text(json.dumps(dict(seed=seed,rows=rows,
            checkpoint='ARTIFICIAL_DATA_ONLY',force_buffer_max_norm_error_N=0.)))
    for seed in evaluation[:2]:(dest/f'walking-{seed}.json').write_text(json.dumps(dict(stable_forward_walking_passed=True,success_rate=1.)))
subprocess.run([sys.executable,str(folder/'sources/ablation_analyze.py'),str(mock)],check=True)
result=json.loads((mock/'ablation-summary.json').read_text())
assert result['confirmatory_complete'] and result['actual_matrix_trials']==435200
assert [c['change_pp'] for c in result['contrasts']]==[-6.25,-6.25,-12.5]
assert all(c['p_holm']==.0234375 for c in result['contrasts'])
assert all(c['reliable_practical_improvement'] for c in result['contrasts'])
assert all(v['noninferiority_passed'] for v in result['retention'].values())
assert (mock/'ablation-effects.png').is_file()
# Exercise incomplete reporting, without an accidental complete-case significance claim.
(mock/'evaluations/A0/evaluation-complete.json').unlink()
subprocess.run([sys.executable,str(folder/'sources/ablation_analyze.py'),str(mock)],check=True)
result=json.loads((mock/'ablation-summary.json').read_text())
assert not result['confirmatory_complete'] and not result['contrasts']
print('ABLATION_ANALYSIS_SYNTHETIC_INTEGRATION_PASS',flush=True)

import json
from pathlib import Path
import sys


def decide(summary,walks,exact2_samples):
    primary=summary['primary']
    retention=[]
    for direction in ['backward','right']:
        row=next(r for r in summary['results'] if r['direction']==direction and r['level_bw']==1.)
        retention.append(dict(direction=direction,delta_pp=row['fall_change_pp'],
            ci95_pp=row['fall_change_ci95_pp'],after_fall_rate=row['after']['fall_rate'],
            passed=row['fall_change_ci95_pp'][1]<2 and row['after']['fall_rate']<=.02))
    checks=dict(enough_independent_seeds=summary['inferential_minimum_met'],
        left2_improvement=primary['fall_change_ci95_pp'][1]<0 and primary['fall_p_exact_two_sided']<.05,
        backward_right_preserved=all(x['passed'] for x in retention),
        walking_passed=len(walks)==2 and all(w['stable_forward_walking_passed'] for w in walks),
        exact2_training_allocations_nonzero=exact2_samples>0)
    return dict(checks=checks,passed=all(checks.values()),retention=retention,
        note='Predeclared engineering gates; percentile seed-bootstrap bounds are approximate, not a proof of exact noninferiority or zero risk.')


if __name__=='__main__':
    folder=Path(sys.argv[1])
    summary=json.loads((folder/'heldout-summary.json').read_text())
    walks=[json.loads((folder/f'walking-after-seed{s}.json').read_text()) for s in [920001,920002]]
    curriculum=json.loads((folder/'curriculum.json').read_text())
    result=decide(summary,walks,curriculum['sampled_left_exact2'])
    (folder/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print('LEFT2_ACCEPTANCE='+json.dumps(result),flush=True)

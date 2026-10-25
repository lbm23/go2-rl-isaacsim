"""Matched evaluation of one frozen checkpoint against existing training replicates."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

FOCUS=['left:2','backward:2','right:2']


def describe(rows):
    eligible=[r for r in rows if r['eligible_at_push']]
    recovered=[r['recovery_time_s'] for r in eligible if r['recovery_time_s'] is not None]
    errors=[r['postpush_mean_speed_error_before_fall_m_s'] for r in eligible
            if r['postpush_mean_speed_error_before_fall_m_s'] is not None]
    survivors=[r['postpush_mean_speed_error_before_fall_m_s'] for r in eligible
               if not r['fell_after_push'] and r['postpush_mean_speed_error_before_fall_m_s'] is not None]
    return dict(trials=len(rows),eligible=len(eligible),prepush_failures=len(rows)-len(eligible),
        falls=sum(r['fell_after_push'] for r in eligible),
        fall_rate=float(np.mean([r['fell_after_push'] for r in eligible])) if eligible else None,
        recovery_rate=len(recovered)/len(eligible) if eligible else None,
        recovered=len(recovered),median_recovery_s=float(np.median(recovered)) if recovered else None,
        p90_recovery_s=float(np.percentile(recovered,90)) if recovered else None,
        speed_error_m_s=float(np.mean(errors)) if errors else None,
        survivor_speed_error_m_s=float(np.mean(survivors)) if survivors else None,
        capped_recovery_s=float(np.mean([r['recovery_time_s'] if r['recovery_time_s'] is not None
            else 9.8 for r in eligible])) if eligible else None)


def audit_pair(reference,current):
    assert len(reference)==len(current)==1088
    assert len({r['trial_id'] for r in current})==1088
    assert all((a['seed'],a['trial_id'],a['direction'],a['level_bw'])==
        (b['seed'],b['trial_id'],b['direction'],b['level_bw']) for a,b in zip(reference,current))
    mass=max(abs(a['mass_kg']-b['mass_kg']) for a,b in zip(reference,current))
    force=max(abs(a['nominal_force_N']-b['nominal_force_N']) for a,b in zip(reference,current))
    assert mass<1e-4 and force<.01
    return dict(maximum_mass_difference_kg=mass,maximum_nominal_force_difference_N=force)


def crossed_ci(values,reference,training_indices,evaluation_indices):
    """Resample whole training seeds and whole matched evaluation-seed blocks."""
    boot=values[training_indices[:,:,None],evaluation_indices[:,None,:]].mean(axis=(1,2))
    boot-=reference[evaluation_indices].mean(axis=1)
    return dict(change=float(values.mean()-reference.mean()),
        exploratory_ci95=np.quantile(boot,[.025,.975]).tolist())


def analyze(folder):
    spec=json.loads((folder/'plan.json').read_text());original=Path(spec['original_experiment'])
    old=json.loads((original/'ablation-summary.json').read_text());destination=folder/'evaluation'
    conditions=[('control',0.)]+[(d,l) for d in ['forward','backward','left','right'] for l in [.25,.5,1.,2.]]
    current=[];seed_metrics={};audits=[]
    for seed in spec['evaluation_seeds']:
        run=json.loads((destination/f'seed{seed}.json').read_text())
        assert run['checkpoint']==spec['checkpoint'] and run['seed']==seed
        assert run['force_buffer_max_norm_error_N']<.01
        reference=json.loads((original/'evaluations/baseline'/f'seed{seed}.json').read_text())
        audits.append(dict(seed=seed,**audit_pair(reference['rows'],run['rows']),
            force_buffer_max_norm_error_N=run['force_buffer_max_norm_error_N']))
        current.extend(run['rows'])
        seed_metrics[str(seed)]={f'{d}:{l:g}':describe([r for r in run['rows'] if r['direction']==d and r['level_bw']==l])
                                for d,l in conditions}
    assert len(current)==17408
    current_metrics={f'{d}:{l:g}':describe([r for r in current if r['direction']==d and r['level_bw']==l]) for d,l in conditions}
    group_rows={g:{k:[] for k in FOCUS} for g in 'ABC'}
    for group in 'ABC':
        for training_seed in old['training_seeds']:
            for seed in spec['evaluation_seeds']:
                run=json.loads((original/'evaluations'/f'{group}{training_seed}'/f'seed{seed}.json').read_text())
                for row in run['rows']:
                    key=f'{row["direction"]}:{row["level_bw"]:g}'
                    if key in FOCUS:group_rows[group][key].append(row)
    groups={g:{k:describe(rows) for k,rows in data.items()} for g,data in group_rows.items()}
    rng=np.random.default_rng(735093);ti=rng.integers(0,8,(20000,8));ei=rng.integers(0,16,(20000,16))
    comparisons=[]
    for group in 'ABC':
        for condition in FOCUS:
            entry=dict(group=group,condition=condition,changes={})
            for field in ['fall_rate','recovery_rate','speed_error_m_s','capped_recovery_s']:
                values=np.asarray([[old['policies'][f'{group}{t}']['seed_metrics'][str(e)][condition][field]
                    for e in spec['evaluation_seeds']] for t in old['training_seeds']],float)
                baseline=np.asarray([seed_metrics[str(e)][condition][field] for e in spec['evaluation_seeds']],float)
                assert np.isfinite(values).all() and np.isfinite(baseline).all()
                entry['changes'][field]=crossed_ci(values,baseline,ti,ei)
            comparisons.append(entry)
    walking=[json.loads((destination/f'walking-{s}.json').read_text()) for s in spec['evaluation_seeds'][:2]]
    report=dict(checkpoint=spec['checkpoint'],checkpoint_sha256=spec['checkpoint_sha256'],
        evaluation_seeds=spec['evaluation_seeds'],matrix_trials=17408,walking_trials=192,
        current=current_metrics,groups=groups,seed_metrics=seed_metrics,comparisons=comparisons,
        walking_passed=all(w['stable_forward_walking_passed'] for w in walking),
        walking_success_rates=[w['success_rate'] for w in walking],audits=audits,
        limitations=['One fixed default checkpoint versus eight trained policies per group; no eight independent default training replicates exist.',
          'Training histories/budgets differ. This is a matched deployment-checkpoint comparison, not a new causal training ablation.',
          'Intervals are exploratory, resampling training seeds and matched evaluation-seed blocks; no multiplicity-adjusted significance claim.',
          'Recovery quantiles pool successful recoveries; speed error stops at first fall and has unequal exposure.',
          'The reused evaluation panel has already been inspected. No model selection or tuning was performed in this follow-up.'])
    (folder/'default7350-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['当前默认模型 model_7350.pt 的同协议补测','',
        '17个条件 × 16个种子 × 每条件每种子64次 = 17,408次矩阵试验；无扰行走192次。',
        '每个重点条件：默认模型1024次；A/B/C每组8192次。',
        '恢复中位数和P90仅包含成功恢复试验；速度误差在首次跌倒前截止。','']
    direction_names={'left:2':'向左 2 BW','backward:2':'向后 2 BW','right:2':'向右 2 BW'}
    for condition in FOCUS:
        lines+=[direction_names[condition],'模型 | 跌倒率 | 恢复率 | 恢复中位数(s) | 恢复P90(s) | 速度误差(m/s) | 封顶恢复分数(s)']
        for label,metrics in [('7350',current_metrics[condition])]+[(g,groups[g][condition]) for g in 'ABC']:
            median=f'{metrics["median_recovery_s"]:.3f}' if metrics['median_recovery_s'] is not None else '无成功恢复'
            p90=f'{metrics["p90_recovery_s"]:.3f}' if metrics['p90_recovery_s'] is not None else '无成功恢复'
            lines.append(f'{label} | {metrics["fall_rate"]*100:.3f}% | {metrics["recovery_rate"]*100:.3f}% | {median} | {p90} | {metrics["speed_error_m_s"]:.4f} | {metrics["capped_recovery_s"]:.4f}')
        lines.append('')
    lines+=['探索性差值区间：各组减7350；负跌倒率差值表示该组更低。区间同时重采样训练种子和配对评测种子，不作多重比较校正后的显著性断言。']
    for comparison in comparisons:
        effect=comparison['changes']['fall_rate']
        lines.append(f'{comparison["group"]}, {direction_names[comparison["condition"]]}: {100*effect["change"]:+.3f}个百分点，探索性95%区间 {[100*x for x in effect["exploratory_ci95"]]}')
    lines+=['',f'无扰行走检查通过：{report["walking_passed"]}',
        '本补测只比较现有固定默认模型与现有训练结果；默认模型没有8个训练重复，训练历史也不同，不能据此作算法因果归因。']
    (folder/'default7350-results.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['7350','A','B','C'];colors=['#59677b','#3979b9','#ed9836','#339b68']
    fig,axes=plt.subplots(2,2,figsize=(12,8));x=np.arange(3)
    for ax,field,title,scale in zip(axes.flat,['fall_rate','median_recovery_s','speed_error_m_s','capped_recovery_s'],
        ['Fall rate (%)','Recovery median, recovered only (s)','Speed error until first fall (m/s)','Failure-aware capped recovery score (s)'],[100,1,1,1]):
        for index,(label,color) in enumerate(zip(labels,colors)):
            values=[(current_metrics[k] if label=='7350' else groups[label][k])[field] for k in FOCUS]
            ax.bar(x+(index-1.5)*.19,np.asarray(values,dtype=float)*scale,width=.18,label=label,color=color)
        ax.set_xticks(x,['Left 2 BW','Backward 2 BW','Right 2 BW']);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0,0].legend(ncol=4);fig.suptitle('Same evaluation protocol; one fixed default checkpoint versus 8 trained models per group')
    fig.tight_layout();fig.savefig(folder/'default7350-comparison.png',dpi=170);plt.close(fig)
    print('DEFAULT7350_ANALYSIS_COMPLETE',json.dumps(dict(matrix_trials=17408,walking_passed=report['walking_passed'])),flush=True)
    return report


if __name__=='__main__':analyze(Path(sys.argv[1]))

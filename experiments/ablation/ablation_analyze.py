"""Training-seed clustered, preregistered analysis; incomplete runs remain visible."""
import csv
import itertools
import json
from pathlib import Path
import sys
import numpy as np


def sign_flip(values):
    values=np.asarray(values,float)
    signs=np.asarray(list(itertools.product([-1,1],repeat=len(values))))
    return float(np.mean(np.abs(signs@values)>=abs(values.sum())-1e-12))


def holm(p):
    result=[0.]*len(p);last=0.
    for rank,index in enumerate(np.argsort(p)):
        last=max(last,min(1.,(len(p)-rank)*p[index]));result[index]=last
    return result


def describe(rows):
    eligible=[r for r in rows if r['eligible_at_push']]
    recovered=[r['recovery_time_s'] for r in eligible if r['recovery_time_s'] is not None]
    errors=[r['postpush_mean_speed_error_before_fall_m_s'] for r in eligible
            if r['postpush_mean_speed_error_before_fall_m_s'] is not None]
    return dict(trials=len(rows),eligible=len(eligible),prepush_failures=len(rows)-len(eligible),
        fall_rate=float(np.mean([r['fell_after_push'] for r in eligible])) if eligible else None,
        recovery_rate=len(recovered)/len(eligible) if eligible else 0.,
        median_recovery_s=float(np.median(recovered)) if recovered else None,
        p90_recovery_s=float(np.percentile(recovered,90)) if recovered else None,
        speed_error_m_s=float(np.mean(errors)) if errors else None,
        capped_recovery_s=float(np.mean([r['recovery_time_s'] if r['recovery_time_s'] is not None
            else 9.8 for r in rows])) if rows else None)


def analyze(folder):
    spec=json.loads((folder/'spec.json').read_text());records={};outcomes={};audits=[];trial_count=0
    conditions=[('control',0.)]+[(d,l) for d in ['forward','backward','left','right'] for l in [.25,.5,1.,2.]]
    for label in ['baseline']+[j['id'] for j in spec['jobs']]:
        outcome=folder/'cases'/label/'outcome.json'
        if outcome.exists():outcomes[label]=json.loads(outcome.read_text())
        dest=folder/'evaluations'/label
        if not (dest/'evaluation-complete.json').exists():continue
        rows=[];per_seed=[]
        for seed in spec['evaluation_seeds']:
            run=json.loads((dest/f'seed{seed}.json').read_text())
            assert run['seed']==seed and len(run['rows'])==1088
            assert run['force_buffer_max_norm_error_N']<.01
            assert run['checkpoint']==json.loads((dest/'policy.json').read_text())['checkpoint']
            assert len({r['trial_id'] for r in run['rows']})==1088
            rows.extend(run['rows']);per_seed.append(run['rows'])
        trial_count+=len(rows)
        metrics={f'{d}:{l:g}':describe([r for r in rows if r['direction']==d and r['level_bw']==l])
                 for d,l in conditions}
        seed_metrics={str(seed):{f'{d}:{l:g}':describe([r for r in seed_rows
            if r['direction']==d and r['level_bw']==l]) for d,l in conditions}
            for seed,seed_rows in zip(spec['evaluation_seeds'],per_seed)}
        walking=[json.loads((dest/f'walking-{s}.json').read_text()) for s in spec['evaluation_seeds'][:2]]
        records[label]=dict(metrics=metrics,seed_metrics=seed_metrics,
            walking_passed=all(w['stable_forward_walking_passed'] for w in walking),
            walking_success_rates=[w['success_rate'] for w in walking])
        if label!='baseline' and 'baseline' in records:
            # Match randomized masses and trial identities to detect an accidental protocol drift.
            baseline=json.loads((folder/'evaluations/baseline'/f'seed{spec["evaluation_seeds"][0]}.json').read_text())['rows']
            compared=per_seed[0]
            assert all((a['trial_id'],a['direction'],a['level_bw'])==(b['trial_id'],b['direction'],b['level_bw'])
                       for a,b in zip(baseline,compared))
            delta=max(abs(a['mass_kg']-b['mass_kg']) for a,b in zip(baseline,compared))
            assert delta<1e-4;audits.append(dict(policy=label,matched_mass_max_difference_kg=delta))
    missing=[j['id'] for j in spec['jobs'] if j['id'] not in records]
    complete=not missing and 'baseline' in records
    report=dict(confirmatory_complete=complete,missing_or_failed_policies=missing,
        training_outcomes=outcomes,actual_matrix_trials=trial_count,planned_matrix_trials=435200,
        training_seeds=spec['training_seeds'],evaluation_seeds=spec['evaluation_seeds'],
        policies=records,audits=audits,contrasts=[],retention={},
        limitations=['Sign-flip inference assumes paired training-seed sign exchangeability/symmetry.',
          'Inference is conditional on the fixed checkpoint and the held-out evaluation panel.',
          'Recovery quantiles condition on recovery; speed error stops at first fall.',
          'Capped recovery assigns 9.8 s to falls/non-recovery; it is not eventual recovery time.',
          'Eight training seeds give coarse uncertainty; no replay-only or interaction effect is isolated.',
          'Incomplete/infrastructure-failed runs are reported; no confirmatory complete-case claim is made.'])
    lines=['# Go2 消融实验结果','',f'完整完成：{complete}；已评测矩阵试验：{trial_count:,} / 435,200。',
           '', 'A：均匀方向、无课程；B：方向加权、无课程；C：方向加权加课程。',
           '每次固定从 model_4575.pt 恢复全部训练状态，2400 个更新周期；独立重复单位为训练种子。','']
    if complete:
        rng=np.random.default_rng(240024);indices=rng.integers(0,8,size=(20000,8))
        values={g:np.asarray([np.mean([records[f'{g}{s}']['seed_metrics'][str(e)]['left:2']['fall_rate']
                    for e in spec['evaluation_seeds']]) for s in spec['training_seeds']]) for g in 'ABC'}
        for new,old in [('B','A'),('C','B'),('C','A')]:
            delta=values[new]-values[old];ci=np.quantile(delta[indices].mean(1),[.025,.975])
            report['contrasts'].append(dict(contrast=f'{new}-{old}',change_pp=float(100*delta.mean()),
                ci95_pp=(100*ci).tolist(),p_exact=sign_flip(delta),paired_changes_pp=(100*delta).tolist()))
        for contrast,p in zip(report['contrasts'],holm([c['p_exact'] for c in report['contrasts']])):
            contrast['p_holm']=p
            contrast['reliable_practical_improvement']=p<.05 and contrast['ci95_pp'][1]<0 and contrast['change_pp']<=-5
            lines.append(f"- {contrast['contrast']}：左侧 2 BW 跌倒率变化 {contrast['change_pp']:.2f} 个百分点，95% 区间 {contrast['ci95_pp']}，Holm p={p:.4g}。")
        for g in 'ABC':
            endpoints=[]
            for d,l in [('backward',1),('backward',2),('right',1),('right',2)]:
                key=f'{d}:{l}';baseline=np.mean([records['baseline']['seed_metrics'][str(e)][key]['fall_rate'] for e in spec['evaluation_seeds']])
                delta=np.asarray([np.mean([records[f'{g}{s}']['seed_metrics'][str(e)][key]['fall_rate']
                       for e in spec['evaluation_seeds']])-baseline for s in spec['training_seeds']])
                upper=float(np.quantile(delta[indices].mean(1),.9875)*100)
                endpoints.append(dict(condition=key,change_pp=float(delta.mean()*100),simultaneous_upper_pp=upper))
            report['retention'][g]=dict(endpoints=endpoints,noninferiority_passed=all(e['simultaneous_upper_pp']<2 for e in endpoints),
                all_walking_passed=all(records[f'{g}{s}']['walking_passed'] for s in spec['training_seeds']))
            lines.append(f"- {g} 组：保留向后/向右提升检验通过={report['retention'][g]['noninferiority_passed']}；所有无扰行走检查通过={report['retention'][g]['all_walking_passed']}。")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4))
        for i in range(8):ax.plot(range(3),[100*values[g][i] for g in 'ABC'],color='0.75',linewidth=1)
        for x,g in enumerate('ABC'):ax.scatter(np.full(8,x),100*values[g],label=g,s=35)
        ax.set(xticks=range(3),xticklabels=['A uniform','B weighted','C curriculum'],ylabel='Left 2 BW fall rate (%)',title='Eight paired training seeds; fixed held-out panel')
        ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(folder/'ablation-effects.png',dpi=180);plt.close(fig)
    else:
        lines+=['本轮存在缺失或失败重复，暂不作完整消融结论。',f'缺失：{missing}；不将未完成检查点替代固定训练预算。']
    lines+=['','判断标准：主比较 Holm 校正 p<0.05、95% 区间上界<0，并至少下降5个百分点；保留性为四项向后/向右条件的同时上界均<+2个百分点。',
            '统计单位是8个训练种子；评测种子和环境样本不能当作独立训练重复。恢复时间只对成功恢复者统计，需结合恢复率和9.8秒封顶分数解读。',
            '当前默认模型没有更换。']
    (folder/'ablation-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (folder/'ABLATION_RESULTS.md').write_text('\n'.join(lines)+'\n')
    with (folder/'ablation-conditions.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['policy','condition']+list(next(iter(next(iter(records.values()))['metrics'].values())).keys()) if records else ['policy','condition'])
        writer.writeheader()
        for label,record in records.items():
            for key,metric in record['metrics'].items():writer.writerow(dict(policy=label,condition=key,**metric))
    print('ABLATION_ANALYSIS_COMPLETE',complete,trial_count,flush=True)


if __name__=='__main__':analyze(Path(sys.argv[1]))

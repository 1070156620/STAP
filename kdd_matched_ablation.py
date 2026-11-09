"""KDD-only matched component ablation; cached scores, source-selected alpha."""
import argparse,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import metric_refinement as mr
import selector_optimization as opt
from support_selection import quality
from support_selection.data import ROOT,sha256
from support_selection.experiment import descriptor,select_top
io=opt.gate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['plan','select','evaluate','all'],default='plan')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/kdd_matched_ablation_v1')
    a=p.parse_args();source=ROOT/'outputs/metric_refinement_v1'
    if not io.read_json(source/'complete.json')['complete']:raise ValueError('Metric refinement incomplete')
    suite=io.read_json(ROOT/'outputs/selector_replacement_suite_v1/suite_manifest.json')
    case=next(c for c in suite['cases'] if c['name']=='kdd2015_primary_14_behavior')
    protocol,config=mr.case_config(case);protocol['seeds']=[20261032]
    policy=io.read_json(ROOT/'configs/metric_refinement.json')
    grid=opt.parameter_grid(config);theta_map={opt.theta_id(t):t for t in grid}
    allowed=[k for k,t in theta_map.items() if t['pool']=='full' and t['features']=='full' and np.isclose(t['margin'],.01)]
    if a.stage=='plan':
        print(json.dumps(dict(dataset='kdd2015',seed=20261032,courses=39,variants=3,budgets=[.1,.2,.3],
            full_margin_pp=1,base_models_retrained=False,scope='Matched explanatory ablation, not the heterogeneous final Ours policy'),indent=2));return
    manifest=dict(config=config,protocol=protocol,seed=20261032,margin=.01,
        hashes={str(path.relative_to(ROOT)):sha256(path) for path in [Path(__file__),ROOT/'metric_refinement.py',ROOT/'selector_optimization.py',ROOT/'support_selection/quality.py']})
    out=io.lock_output(a.out,manifest)
    old=source/'runs'/case['name']
    if a.stage in ['select','all']:
        frozen=[]
        for uid,tag,train,target,cols,cfg in opt.tasks(protocol,['kdd2015']):
            summary=pd.read_csv(old/'tuning'/(uid+'.source_summary.csv'))
            theta=mr.choose_with_ap(summary,theta_map,policy,allowed)
            f=out/'frozen'/(uid+'.json')
            value=dict(**tag,theta=theta,selection='source Recall then AP, full pool/full features/margin 1 pp fixed',target_labels_used=False,
                source_summary_sha256=sha256(old/'tuning'/(uid+'.source_summary.csv')))
            if f.exists() and io.read_json(f)!=value:raise ValueError('Frozen ablation configuration changed')
            io.write_json(f,value);frozen.append(dict(path=str(f.relative_to(out)),sha256=sha256(f)))
        io.write_json(out/'all_frozen.json',dict(complete=True,tasks=len(frozen),files=frozen))
    if a.stage in ['evaluate','all']:
        for item in io.read_json(out/'all_frozen.json')['files']:
            if sha256(out/item['path'])!=item['sha256']:raise ValueError('Frozen configuration modified')
        with threadpool_limits(limits=2),patch.object(quality,'score_pool',mr.cached_scores):
            for uid,tag,train,target,cols,cfg in opt.tasks(protocol,['kdd2015']):
                marker=out/'evaluation'/(uid+'.complete.json')
                if io.check_done(out,marker):continue
                theta=io.read_json(out/'frozen'/(uid+'.json'))['theta']
                e=quality.build_evidence(train,cols,cfg,ROOT/'.cache/quality_scores')
                score,_=mr.cached_scores(train,target.drop(columns='y'),cols,cfg,ROOT/'.cache/quality_scores')
                rows=[]
                for b in cfg['budgets']:
                    info=opt.utility_estimates(e,descriptor(train,target.drop(columns='y'),b,cols),b,theta,cfg,config)
                    if info['fallback']:raise ValueError('Cannot mark adaptation enabled when fallback is active')
                    mapping={'baseline_fixed':info['fixed'],'adaptive_no_margin':opt.apply_margin(info,0),
                             'adaptive_with_margin':opt.apply_margin(info,.01)}
                    for name,candidate in mapping.items():
                        metrics=mr.course_metrics(target,score[candidate],b)
                        rows.append(dict(**tag,budget=b,method=name,candidate=candidate,
                            adaptive=name!='baseline_fixed',margin=name=='adaptive_with_margin',**metrics))
                file=out/'evaluation'/(uid+'.csv');io.write_csv(file,pd.DataFrame(rows))
                io.write_json(marker,dict(complete=True,files={str(file.relative_to(out)):sha256(file)}))
                print('KDD ABLATION DONE '+uid,flush=True)
        files=list((out/'evaluation').glob('*.csv'))
        if len(files)!=39:raise ValueError('Incomplete KDD ablation')
        frame=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
        io.write_csv(out/'analysis/course_metrics.csv',frame)
        summary=frame.groupby(['budget','method','adaptive','margin'],as_index=False)[['recall','precision','ap','lift']].mean().merge(
            frame.groupby(['budget','method'],as_index=False)[['tp','n','selected','positives']].sum(),on=['budget','method'])
        order={'baseline_fixed':0,'adaptive_no_margin':1,'adaptive_with_margin':2}
        summary['order']=summary.method.map(order);summary=summary.sort_values(['budget','order']).drop(columns='order')
        io.write_csv(out/'analysis/absolute_ablation.csv',summary)
        io.write_json(out/'complete.json',dict(complete=True,courses=39,seed=20261032,
            no_new_base_model_training=True,matched_pool_features_alpha=True,full_margin_pp=1,
            full_is_not_identical_to_heterogeneous_tuned_ours=True))
        print(summary.loc[np.isclose(summary.budget,.2)].to_string(index=False))


if __name__=='__main__':main()

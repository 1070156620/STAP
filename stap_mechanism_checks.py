"""Read-only course audit and matched KDD policy interventions; base scores cache-only."""
import argparse, copy, json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import metric_refinement as ref
import selector_optimization as opt
from support_selection import quality
from support_selection.experiment import descriptor,select_top
from support_selection.data import ROOT,sha256

BASE=ROOT/'outputs/metric_refinement_v1'
OUT=ROOT/'outputs/stap_mechanism_v1'
def cases():
    return [x for x in json.loads((ROOT/'outputs/selector_replacement_suite_v1/suite_manifest.json').read_text())['cases'] if x['family']=='primary' and x['dataset'] in ['oulad','kdd2015']]

def audit():
    OUT.mkdir(exist_ok=True,parents=True)
    rows=[]; components=[]
    for c in cases():
        run=BASE/'runs'/c['name']
        metrics=pd.read_csv(run/'metric_analysis/course_metrics.csv')
        for f in sorted((run/'evaluation').glob('*.decisions.json')):
            d=json.loads(f.read_text());uid=f.name.removesuffix('.decisions.json')
            for choice in d['choices']:
                b=choice['budget'];mapping=choice['mapping'];theta=choice['tuned_theta']
                ours=metrics[(metrics.seed==d['seed'])&(metrics.test_course.astype(str)==str(d['test_course']))&np.isclose(metrics.budget,b)&(metrics.method=='tuned_ours')]
                if len(ours)!=1:raise ValueError('Unpaired course decision '+uid)
                val=ours.iloc[0]
                row={k:d[k] for k in ['dataset','seed','test_course','source_courses','draw']}
                row.update(task=uid,budget=b,candidate=mapping['tuned_ours'],**theta,fallback=choice['fallback'],source_only=choice['source_only'])
                row.update({k:val[k] for k in ['recall','precision','tp','ap','lift','n','selected','positives']})
                rows.append(row)
                for method in ['feature_structural','feature_full','feature_source_validation','tuned_no_margin','tuned_ours']:
                    m=metrics[(metrics.seed==d['seed'])&(metrics.test_course.astype(str)==str(d['test_course']))&np.isclose(metrics.budget,b)&(metrics.method==method)]
                    if len(m)!=1:raise ValueError('Missing archived variant '+method)
                    components.append({**{k:row[k] for k in ['dataset','seed','test_course','budget']},'method':method,**{k:m.iloc[0][k] for k in ['recall','precision','tp','ap','lift']}})
    df=pd.DataFrame(rows);df.to_csv(OUT/'course_choices.csv',index=False)
    df.groupby(['dataset','seed','budget','candidate'],as_index=False).size().to_csv(OUT/'candidate_frequencies.csv',index=False)
    df.groupby(['dataset','test_course','budget']).agg(seeds=('seed','nunique'),distinct_candidates=('candidate','nunique'),recall_mean=('recall','mean'),recall_sd=('recall','std'),precision_mean=('precision','mean'),fallback_count=('fallback','sum')).reset_index().to_csv(OUT/'course_stability.csv',index=False)
    raw=pd.DataFrame(components);raw.to_csv(OUT/'archived_variants_course.csv',index=False)
    seed=raw.groupby(['dataset','seed','budget','method'],as_index=False).agg(recall=('recall','mean'),precision=('precision','mean'),ap=('ap','mean'),lift=('lift','mean'),tp=('tp','sum'))
    seed.to_csv(OUT/'archived_variants_seed.csv',index=False)
    ext=pd.read_csv(ROOT/'outputs/published_baselines_v1/analysis/course_metrics.csv')
    ext=ext[ext.method.isin(['RF','Linear SVM'])]
    # Existing LR/HGB overlaps are already stored with course metrics; export explicitly.
    lists=[]
    for c in cases():
        run=BASE/'runs'/c['name'];metrics=pd.read_csv(run/'metric_analysis/course_metrics.csv')
        for uid,d in df[df.dataset==c['dataset']].groupby('task'):
            pred=pd.read_csv(run/'predictions'/(uid+'.csv.gz'))
            other=pd.read_csv(ROOT/'outputs/published_baselines_v1/predictions'/(uid+'.csv.gz'))
            pred=pred.merge(other[['record_id','score__RF','score__Linear SVM']],on='record_id',validate='one_to_one')
            for b in [.1,.2,.3]:
                for model,old in [('LR','tuned_lr'),('HGB','tuned_hgb'),('RF',None),('Linear SVM',None)]:
                    values=[]
                    for _,q in pred.groupby('cohort'):
                        ours=q[f'selected__{b:g}__tuned_ours'].to_numpy(bool)
                        mask=q[f'selected__{b:g}__{old}'].to_numpy(bool) if old else select_top(q['score__'+model],q.record_id,b)
                        values.append(float((ours&mask).sum()/(ours|mask).sum()))
                    lists.append(dict(task=uid,budget=b,comparator=model,jaccard=np.mean(values)))
    pd.DataFrame(lists).merge(df[['task','dataset','seed','test_course','budget']],on=['task','budget'],validate='many_to_one').to_csv(OUT/'course_list_overlap.csv',index=False)
    report=dict(completed=True,course_tasks=df.task.nunique(),policy_capacity_rows=len(df),fallback_rows=int(df.fallback.sum()),source_only_rows=int(df.source_only.sum()),new_predictor_training=False,
        interpretation='Feature-specific archived variants reselect source configurations; they are restricted-procedure comparisons, not one-component matched ablations. No-margin retains the frozen configuration. A fallback effect is not estimable when it never activates.')
    (OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

def matched(pilot=False):
    c=next(c for c in cases() if c['dataset']=='kdd2015')
    protocol,config=ref.case_config(c);run=BASE/'runs'/c['name']
    folder=OUT/('pilot_matched' if pilot else 'matched');folder.mkdir(parents=True,exist_ok=True)
    # No new parameter search; same source set, pool, alpha and target records in every variant.
    for idx,(uid,tag,train,target,cols,cfg) in enumerate(opt.tasks(protocol,['kdd2015'])):
        if pilot and idx:break
        dest=folder/(uid+'.csv');marker=dest.with_suffix('.json')
        frozen_path=run/'tuning'/(uid+'.frozen.json');freeze_hash=sha256(frozen_path)
        if marker.exists() and dest.exists():
            m=json.loads(marker.read_text())
            if m['freeze_hash']!=freeze_hash or m['output_hash']!=sha256(dest):raise ValueError('Changed matched record')
            print('MATCHED REUSE '+uid,flush=True);continue
        print('MATCHED START '+uid,flush=True)
        frozen=json.loads(frozen_path.read_text());theta=frozen['tuned']
        with patch.object(quality,'score_pool',ref.cached_scores):
            evidence=quality.build_evidence(train,cols,cfg,ROOT/'.cache/quality_scores')
        tx=target.drop(columns='y');scores,_=ref.cached_scores(train,tx,cols,cfg,ROOT/'.cache/quality_scores')
        saved=json.loads((run/'evaluation'/(uid+'.decisions.json')).read_text())
        archived=pd.read_csv(run/'metric_analysis/course_metrics.csv')
        rows=[]
        for b in cfg['budgets']:
            query=descriptor(train,tx,b,cols)
            variants={'source_mean':{**theta,'features':'source_validation'},'no_mean_shifts':{**theta,'features':'structural'} if theta['features']=='full' else copy.deepcopy(theta),'no_margin':{**theta,'margin':0},'STAP':copy.deepcopy(theta)}
            full,info=opt.choice(evidence,query,b,theta,cfg,config)
            expected=next(d for d in saved['choices'] if np.isclose(d['budget'],b))['mapping']['tuned_ours']
            if full!=expected:raise ValueError('Full STAP decision differs from frozen original')
            baseline_masks={str(q.cohort.iloc[0]):select_top(scores[full][target.index.get_indexer(q.index)],q.record_id,b) for _,q in target.groupby('cohort')}
            for name,t in variants.items():
                candidate,inf=opt.choice(evidence,query,b,t,cfg,config)
                metrics=ref.course_metrics(target,scores[candidate],b)
                if name=='STAP':
                    reference=archived[(archived.seed==tag['seed'])&(archived.test_course.astype(str)==tag['test_course'])&np.isclose(archived.budget,b)&(archived.method=='tuned_ours')]
                    if len(reference)!=1:raise ValueError('Missing original full-method metrics')
                    for metric in ['recall','precision','ap','lift','tp','n','selected','positives']:
                        if not np.isclose(metrics[metric],reference.iloc[0][metric],atol=1e-10,rtol=0):raise ValueError('Full-method metric mismatch: '+metric)
                overlaps=[]
                for key,q in target.groupby('cohort'):
                    mask=select_top(scores[candidate][target.index.get_indexer(q.index)],q.record_id,b);orig=baseline_masks[str(key)]
                    overlaps.append((mask&orig).sum()/(mask|orig).sum())
                rows.append(dict(**tag,budget=b,variant=name,candidate=candidate,source_only=inf['source_only'],fallback=inf['fallback'],intervention_inactive=name=='no_mean_shifts' and theta['features']!='full',jaccard_vs_stap=float(np.mean(overlaps)),**metrics))
        pd.DataFrame(rows).to_csv(dest,index=False)
        marker.write_text(json.dumps(dict(complete=True,freeze_hash=freeze_hash,output_hash=sha256(dest),base_training=False)),encoding='utf-8')
        print('MATCHED DONE '+uid,flush=True)
    summarize(pilot)

def summarize(pilot=False):
    folder=OUT/('pilot_matched' if pilot else 'matched');files=sorted(f for f in folder.glob('*.csv') if f.with_suffix('.json').exists())
    if not files:raise ValueError('No matched outputs')
    if not pilot and len(files)!=195:raise ValueError('Incomplete matched study: expected 195 tasks')
    df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    seed=df.groupby(['dataset','seed','budget','variant'],as_index=False).agg(recall=('recall','mean'),precision=('precision','mean'),ap=('ap','mean'),lift=('lift','mean'),tp=('tp','sum'),jaccard_vs_stap=('jaccard_vs_stap','mean'))
    seed.to_csv(folder/'../matched_seed_summary.csv' if not pilot else folder/'pilot_summary.csv',index=False)
    print(seed[np.isclose(seed.budget,.2)].to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['plan','audit','pilot','matched','all'],default='plan');a=p.parse_args()
    with threadpool_limits(limits=2):
        if a.stage=='plan':print(json.dumps(dict(archived_course_tasks=230,matched_kdd_tasks=195,variants=4,capacities=3,matched_policy_evaluations=2340,new_base_model_training=0,retuning=False,source_score_cache_required=True),indent=2))
        if a.stage in ['audit','all']:audit()
        if a.stage=='pilot':matched(True)
        if a.stage in ['matched','all']:matched()

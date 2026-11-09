"""RF and linear SVM benchmarks, source-only tuning, same outer populations as Ours."""
from __future__ import annotations
import argparse,hashlib,json,platform,os,warnings
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import selector_optimization as opt
import metric_refinement as metrics
from support_selection import quality
from support_selection.data import ROOT,sha256
io=opt.gate


def specifications(config):
    return {'RF':[{**config['rf'],'min_samples_leaf':leaf} for leaf in config['rf']['min_samples_leaf']],
            'Linear SVM':[dict(C=c,max_iter=config['linear_svm']['max_iter']) for c in config['linear_svm']['C']]}


def score(train,target_x,cols,family,parameters,seed,cache,threads=2):
    if 'y' in target_x:raise ValueError('Target labels must not reach model scoring')
    if set(train.student_id)&set(target_x.student_id):raise ValueError('Source-target learner leakage')
    if train.y.nunique()!=2:raise ValueError('Training requires both classes; no constant-model substitution')
    h=hashlib.sha256(json.dumps(['published-family-v1',sklearn.__version__,family,parameters,seed,cols],sort_keys=True).encode())
    for frame in [train[['record_id','y']+cols],target_x[['record_id']+cols]]:
        h.update(pd.util.hash_pandas_object(frame,index=False).to_numpy().tobytes())
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);path=cache/(h.hexdigest()+'.npz')
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:return saved['score']
    if family=='RF':
        estimator=RandomForestClassifier(**parameters,random_state=seed,n_jobs=threads)
        model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),estimator)
    elif family=='Linear SVM':
        estimator=LinearSVC(**parameters,dual=False,random_state=seed)
        model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),estimator)
    else:raise ValueError('Unrecognized published model family')
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always');model.fit(train[cols],train.y)
    values=model.predict_proba(target_x[cols])[:,1] if family=='RF' else model.decision_function(target_x[cols])
    if not np.isfinite(values).all():raise ValueError('Non-finite model scores')
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,score=values);tmp.replace(path)
    if messages:io.write_json(path.with_suffix('.warnings.json'),{'warnings':[str(m.message) for m in messages]})
    return values


def tune(train,cols,cfg,config,cache):
    rows=[]
    for course,fit,val in quality.source_folds(train,cfg,middle=True):
        if course in set(fit.course) or set(fit.student_id)&set(val.student_id):raise ValueError('Invalid validation fold')
        for family,candidates in specifications(config).items():
            for index,parameters in enumerate(candidates):
                values=score(fit,val.drop(columns='y'),cols,family,parameters,cfg['seed'],cache,config['threads'])
                m=metrics.course_metrics(val,values,config['primary_budget'])
                if not np.isfinite(m['recall']) or not np.isfinite(m['ap']):raise ValueError('Undefined source metric')
                rows.append(dict(validation_course=str(course),family=family,candidate=index,recall=m['recall'],ap=m['ap']))
    evidence=pd.DataFrame(rows)
    if evidence.empty or evidence.validation_course.nunique()<2:raise ValueError('Insufficient source validation courses')
    means=evidence.groupby(['family','candidate'],as_index=False)[['recall','ap']].mean();selected={}
    for family,candidates in specifications(config).items():
        d=means.loc[means.family==family];d=d.loc[d.recall>=d.recall.max()-1e-12];d=d.loc[d.ap>=d.ap.max()-1e-12]
        winner=min(d.candidate.astype(int),key=lambda i:-candidates[i]['min_samples_leaf'] if family=='RF' else candidates[i]['C'])
        selected[family]=candidates[winner]
    return selected,evidence


def protocol_for(config):
    protocol=io.read_json(ROOT/'configs/decision_principle.json')
    protocol.update(seeds=config['seeds'],budgets=config['budgets'],threads=config['threads'])
    return protocol


def run(config,out,pilot=False):
    protocol=protocol_for(config);entries=[];cache=ROOT/'.cache/published_baseline_scores'
    # All source parameter selections finish before outer target evaluation.
    for uid,tag,train,target,cols,cfg in opt.tasks(protocol,config['datasets'],pilot=pilot):
        marker=out/'tuning'/(uid+'.complete.json')
        if not io.check_done(out,marker):
            print('BASELINE TUNE START '+uid,flush=True)
            selected,evidence=tune(train,cols,cfg,config,cache)
            f=out/'tuning'/(uid+'.frozen.json');e=out/'tuning'/(uid+'.source.csv')
            io.write_json(f,dict(**tag,selected=selected,target_labels_used=False));io.write_csv(e,evidence)
            io.write_json(marker,dict(complete=True,files={str(p.relative_to(out)):sha256(p) for p in [f,e]}))
            print('BASELINE TUNE DONE '+uid,flush=True)
        f=out/'tuning'/(uid+'.frozen.json');entries.append(dict(path=str(f.relative_to(out)),sha256=sha256(f)))
    if not entries:raise ValueError('No eligible outer tasks')
    io.write_json(out/'all_frozen.json',dict(complete=True,tasks=len(entries),files=entries))
    for item in entries:
        if sha256(out/item['path'])!=item['sha256']:raise ValueError('Frozen parameters modified')
    count=0
    for uid,tag,train,target,cols,cfg in opt.tasks(protocol,config['datasets'],pilot=pilot):
        marker=out/'evaluation'/(uid+'.complete.json')
        if io.check_done(out,marker):count+=1;continue
        parameters=io.read_json(out/'tuning'/(uid+'.frozen.json'))['selected']
        prediction=target[['record_id','student_id','course','cohort','y']].copy();rows=[]
        for family,setting in parameters.items():
            values=score(train,target.drop(columns='y'),cols,family,setting,cfg['seed'],cache,config['threads'])
            prediction['score__'+family]=values
            for b in cfg['budgets']:
                for cohort_name,cohort in target.groupby('cohort',sort=True):
                    ix=target.index.get_indexer(cohort.index)
                    m=metrics.cohort_metrics(cohort.y,values[ix],cohort.record_id,b)
                    rows.append(dict(**tag,method=family,budget=b,cohort=str(cohort_name),**m))
        f=out/'evaluation'/(uid+'.metrics.csv');p=out/'predictions'/(uid+'.csv.gz')
        io.write_csv(f,pd.DataFrame(rows));io.write_csv(p,prediction)
        io.write_json(marker,dict(complete=True,files={str(path.relative_to(out)):sha256(path) for path in [f,p]}))
        count+=1;io.write_json(out/'progress.json',dict(stage='evaluate',completed=count,expected=len(entries),latest=uid))
        print(f'BASELINE EVAL DONE {count}/{len(entries)} '+uid,flush=True)
    if count!=len(entries):raise ValueError('Incomplete evaluations')
    io.write_json(out/'complete.json',dict(complete=True,tasks=count,pilot=pilot,source_only_tuning=True))


def summarize(config,out):
    status=io.read_json(out/'complete.json')
    parts=[]
    for marker in (out/'evaluation').glob('*.complete.json'):
        if not io.check_done(out,marker):raise ValueError('Evaluation output incomplete')
        parts.append(pd.read_csv(marker.with_name(marker.name.replace('.complete.json','.metrics.csv'))))
    if len(parts)!=status['tasks']:raise ValueError('Coverage mismatch')
    frame=pd.concat(parts,ignore_index=True);keys=['dataset','seed','test_course','budget','method']
    c=frame.groupby(keys,as_index=False)[['recall','precision','ap','lift']].mean().merge(
        frame.groupby(keys,as_index=False)[['tp','n','selected','positives']].sum(),on=keys,validate='one_to_one')
    io.write_csv(out/'analysis/course_metrics.csv',c)
    keys=['dataset','seed','budget','method'];seed=c.groupby(keys,as_index=False)[['recall','precision','ap','lift']].mean().merge(
        c.groupby(keys,as_index=False)[['tp','n','selected','positives']].sum(),on=keys,validate='one_to_one')
    io.write_csv(out/'analysis/new_baseline_seed_metrics.csv',seed)
    if status['pilot']:
        print('PILOT ONLY: software/data-path check; not a paper-ready complete benchmark');return
    old=pd.read_csv(ROOT/'outputs/metric_refinement_v1/analysis/seed_metrics.csv')
    old=old.loc[(old.experiment_family=='primary')&old.dataset.isin(config['datasets'])&old.seed.isin(config['seeds'])&old.method.isin(['random','tuned_lr','tuned_hgb','tuned_ours'])].copy()
    old['method']=old.method.map(dict(random='Random',tuned_lr='LR',tuned_hgb='HGB',tuned_ours='Ours'))
    combined=pd.concat([old[seed.columns],seed],ignore_index=True)
    order={'Random':0,'LR':1,'HGB':2,'RF':3,'Linear SVM':4,'Ours':5}
    combined['method_order']=combined.method.map(order)
    combined=combined.sort_values(['dataset','seed','budget','method_order']).drop(columns='method_order')
    io.write_csv(out/'analysis/paper_all_seeds.csv',combined)
    paper=combined.loc[(combined.seed==20261032)&np.isclose(combined.budget,.2)].copy()
    if len(paper)!=len(config['datasets'])*6:raise ValueError('Incomplete primary table')
    io.write_csv(out/'analysis/paper_main_20261032.csv',paper)
    means=combined.groupby(['dataset','budget','method'])[['recall','precision','ap','lift','tp']].agg(['mean','std'])
    means.columns=['_'.join(v) for v in means.columns]
    io.write_csv(out/'analysis/mean_sd.csv',means.reset_index())
    comparisons=[]
    oldcourses=[]
    for name in config['datasets']:
        path=ROOT/'outputs/metric_refinement_v1/runs'/f'{name}_primary_{28 if name=="oulad" else 14}_behavior/metric_analysis/course_metrics.csv'
        d=pd.read_csv(path);oldcourses.append(d.loc[(d.method=='tuned_ours')&d.seed.isin(config['seeds'])])
    ours=pd.concat(oldcourses,ignore_index=True)
    for (name,budget),part in c.groupby(['dataset','budget']):
        left=ours.loc[(ours.dataset==name)&np.isclose(ours.budget,budget),['seed','test_course','recall']].rename(columns={'recall':'ours'})
        for family in ['RF','Linear SVM']:
            joined=part.loc[part.method==family].merge(left,on=['seed','test_course'],validate='one_to_one')
            delta=(joined.ours-joined.recall).groupby(joined.test_course).mean()
            lo,hi=io.bootstrap_interval(delta.to_numpy(),2000,20261032)
            comparisons.append(dict(dataset=name,budget=budget,comparator=family,delta_recall_pp=100*delta.mean(),ci_low_pp=100*lo,ci_high_pp=100*hi))
    io.write_csv(out/'analysis/paired_checks.csv',pd.DataFrame(comparisons))
    print(paper.to_string(index=False));print('PAPER DATA READY '+str(out/'analysis/paper_main_20261032.csv'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['plan','pilot','run','summarize','all'],default='plan')
    p.add_argument('--config',type=Path,default=ROOT/'configs/published_baselines.json')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/published_baselines_v1')
    a=p.parse_args();config=io.read_json(a.config)
    if a.stage=='plan':
        print(json.dumps(dict(datasets=config['datasets'],seeds=config['seeds'],outer_tasks=230,
            new_algorithms=list(specifications(config)),capacities=config['budgets'],source_candidates_per_family=3,
            new_metrics='Recall, Precision, TP, AP, Lift',training_started=False),indent=2));return
    paths=[Path(__file__),a.config,ROOT/'support_selection/quality.py',ROOT/'selector_optimization.py',ROOT/'metric_refinement.py',ROOT/'outputs/metric_refinement_v1/analysis/seed_metrics.csv']
    protocol=protocol_for(config)
    paths.extend(ROOT/protocol['datasets'][name]['processed'] for name in config['datasets'])
    manifest=dict(config=config,pilot=a.stage=='pilot',hashes={str(path.relative_to(ROOT)):sha256(path) for path in paths},
        versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__))
    out=io.lock_output(a.out/ 'pilot' if a.stage=='pilot' else a.out,manifest)
    with threadpool_limits(limits=config['threads']):
        if a.stage in ['pilot','run','all']:run(config,out,pilot=a.stage=='pilot')
        if a.stage in ['pilot','summarize','all']:summarize(config,out)


if __name__=='__main__':main()

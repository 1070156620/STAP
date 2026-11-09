"""Extend metrics, reselect parameters using SOURCE recall then AP, refreeze, evaluate.

Read-only access to the preceding replacement suite. Candidate scoring is cache-only:
missing cache entries stop execution instead of retraining models silently.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
import selector_optimization as opt
from support_selection import quality
from support_selection.data import ROOT,sha256
from support_selection.experiment import select_top,stable_hash
io=opt.gate


def cached_scores(train,target,cols,cfg,cache):
    """Exact existing quality.score_pool key, without a training fallback."""
    # Existing inner-evidence helper passes a validation frame with labels. Strip
    # them immediately; cache construction and reading never access validation Y.
    target=target.drop(columns='y',errors='ignore')
    if set(train.student_id)&set(target.student_id):raise ValueError('Student overlap')
    specs=quality.candidate_specs(cfg);h=hashlib.sha256()
    h.update(json.dumps([quality.VERSION,sklearn.__version__,specs,cols,cfg['seed']],sort_keys=True).encode())
    for frame in [train[['record_id','y']+cols],target[['record_id']+cols]]:
        h.update(pd.util.hash_pandas_object(frame,index=False).to_numpy().tobytes())
    path=Path(cache)/(h.hexdigest()+'.npz')
    if not path.is_file():
        raise FileNotFoundError('Required continuous score cache missing; no models retrained: '+str(path))
    with np.load(path,allow_pickle=False) as z:scores={name:z[name] for name in z.files}
    if set(scores)!=set(specs) or any(len(v)!=len(target) for v in scores.values()):
        raise ValueError('Candidate cache does not match the existing scoring specification')
    return scores,True


def cohort_metrics(y,scores,ids,budget):
    y=np.asarray(y,dtype=int);scores=np.asarray(scores,dtype=float)
    if not np.isfinite(scores).all():raise ValueError('Non-finite continuous risk score')
    mask=select_top(scores,ids,budget);tp=int(y[mask].sum());positives=int(y.sum())
    precision=tp/int(mask.sum());prevalence=float(y.mean())
    return dict(n=len(y),positives=positives,tp=tp,selected=int(mask.sum()),
        recall=tp/positives if positives else np.nan,precision=precision,
        ap=float(average_precision_score(y,scores)) if positives else np.nan,
        lift=precision/prevalence if positives else np.nan)


def course_metrics(frame,scores,budget):
    values=[]
    for _,cohort in frame.groupby('cohort',sort=True):
        ix=frame.index.get_indexer(cohort.index)
        values.append(cohort_metrics(cohort.y,np.asarray(scores)[ix],cohort.record_id,budget))
    df=pd.DataFrame(values)
    return {**{c:float(df[c].mean()) for c in ['recall','precision','ap','lift']},
            **{c:int(df[c].sum()) for c in ['n','positives','tp','selected']}}


def choose_with_ap(summary,theta_map,policy,allowed=None):
    candidates=summary.loc[np.isclose(summary.budget,policy['primary_budget'])].copy()
    if allowed is not None:candidates=candidates.loc[candidates.theta_id.isin(allowed)]
    if candidates.empty or candidates[['recall','ap']].isna().any().any():
        raise ValueError('Undefined source Recall/AP: no silent parameter exclusion')
    tol=policy['tie_tolerance']
    candidates=candidates.loc[candidates.recall>=candidates.recall.max()-tol]
    candidates=candidates.loc[candidates.ap>=candidates.ap.max()-tol]
    def simplicity(row):
        theta=theta_map[row.theta_id]
        return (['source_validation','structural','full'].index(theta['features']),
            row.switch_rate,-theta['alpha'],-theta['margin'],0 if theta['pool']=='full' else 1,row.theta_id)
    row=min(candidates.itertuples(),key=simplicity)
    return theta_map[row.theta_id]


def case_config(case):
    config=io.read_json(ROOT/case['config'])
    protocol=io.read_json(ROOT/config['base_protocol'])
    protocol.update(seeds=config['seeds'],budgets=config['budgets'],threads=config['threads'])
    config['dataset_names']=[case['dataset']]
    return protocol,config


def refine_case(case,policy,source_root,out):
    protocol,config=case_config(case)
    old=ROOT/case['result'];old_frozen=opt.frozen_index(old)
    new=out/'runs'/case['name'];new.mkdir(parents=True,exist_ok=True)
    theta_map={opt.theta_id(t):t for t in opt.parameter_grid(config)}
    entries=[]
    for uid,tag,train,target,cols,cfg in opt.tasks(protocol,[case['dataset']]):
        marker=new/'tuning'/(uid+'.complete.json')
        if io.check_done(new,marker):
            print('METRIC SOURCE REUSE '+case['name']+' '+uid,flush=True)
        else:
            oldmarker=old/'tuning'/(uid+'.complete.json')
            if not io.check_done(old,oldmarker):raise ValueError('Prior source search incomplete')
            original=io.read_json(old_frozen[uid])
            grid=pd.read_csv(old/'tuning'/(uid+'.source_grid.csv'))
            if set(grid.theta_id)!=set(theta_map):raise ValueError('Prior grid differs from all 142 specified configurations')
            bank={}
            for course,fit,val in quality.source_folds(train,cfg,middle=True):
                if course in set(fit.course) or set(fit.student_id)&set(val.student_id):
                    raise ValueError('Contaminated middle validation fold')
                scores,_=cached_scores(fit,val.drop(columns='y'),cols,cfg,ROOT/'.cache/quality_scores')
                for b in cfg['budgets']:
                    for candidate in scores:
                        bank[(str(course),float(b),candidate)]=course_metrics(val,scores[candidate],b)
            enriched=[]
            for row in grid.to_dict('records'):
                key=(str(row['validation_course']),float(row['budget']),row['candidate'])
                if key not in bank:raise ValueError('Source metric pairing failed')
                metrics=bank[key]
                if not np.isclose(row['recall'],metrics['recall'],atol=1e-12,rtol=0):
                    raise ValueError('Recomputed source recall differs from original source search')
                enriched.append({**row,**metrics})
            frame=pd.DataFrame(enriched)
            summary=frame.groupby(['theta_id','budget'],as_index=False).agg(
                recall=('recall','mean'),precision=('precision','mean'),ap=('ap','mean'),lift=('lift','mean'),
                tp=('tp','sum'),n=('n','sum'),switch_rate=('switched_from_fixed','mean'),
                validation_courses=('validation_course','nunique'))
            winner=choose_with_ap(summary,theta_map,policy)
            feature_best={mode:choose_with_ap(summary,theta_map,policy,
                [key for key,theta in theta_map.items() if theta['features']==mode]) for mode in config['feature_modes']}
            frozen={**original,'tuned':winner,'feature_best':feature_best,
                'previous_tuned':original['tuned'],'objective':'Source Recall@20%, then source AP, then predefined simplicity',
                'target_labels_used':False,'target_scores_computed':False,
                'old_freeze_sha256':sha256(old_frozen[uid]),'metric_policy':policy}
            base=new/'tuning'/uid
            files=[Path(str(base)+'.frozen.json'),Path(str(base)+'.source_grid.csv'),Path(str(base)+'.source_summary.csv')]
            io.write_json(files[0],frozen);io.write_csv(files[1],frame);io.write_csv(files[2],summary)
            io.write_json(marker,dict(complete=True,files={str(p.relative_to(new)):sha256(p) for p in files}))
            print('METRIC SOURCE DONE '+case['name']+' '+uid+' '+json.dumps(winner),flush=True)
        path=new/'tuning'/(uid+'.frozen.json')
        entries.append(dict(task=uid,freeze=str(path.relative_to(new)),sha256=sha256(path)))
        io.write_json(out/'progress.json',dict(stage='source_metric_selection',case=case['name'],completed=len(entries),expected=case['tasks']))
    if len(entries)!=case['tasks']:raise ValueError('Refinement source task count mismatch')
    io.write_json(new/'tuning_complete.json',dict(complete=True,tasks=len(entries),frozen=entries,
        configurations_per_task=len(theta_map),target_labels_used=False))


def extend_evaluation(case,out):
    protocol,config=case_config(case);new=out/'runs'/case['name']
    for uid,tag,train,target,cols,cfg in opt.tasks(protocol,[case['dataset']]):
        marker=new/'metric_evaluation'/(uid+'.complete.json')
        if io.check_done(new,marker):continue
        scores,_=cached_scores(train,target.drop(columns='y'),cols,cfg,ROOT/'.cache/quality_scores')
        scores['fixed_blend']=(rankdata(scores['lr_C1'])+rankdata(scores['rule_inactivity']))/(2*len(target))
        scores['random']=np.array([stable_hash(record,cfg['seed']) for record in target.record_id],dtype=float)
        decisions=io.read_json(new/'evaluation'/(uid+'.decisions.json'))
        if decisions['target_y_sent_to_scoring'] or decisions['outer_student_overlap']:
            raise ValueError('Invalid prior target-scoring boundary')
        frame=pd.read_csv(new/'evaluation'/(uid+'.metrics.csv'),dtype={'cohort':str})
        additions=[]
        for decision in decisions['choices']:
            b=decision['budget']
            for method,candidate in decision['mapping'].items():
                for cohort_name,cohort in target.groupby('cohort',sort=True):
                    ix=target.index.get_indexer(cohort.index)
                    result=cohort_metrics(cohort.y,scores[candidate][ix],cohort.record_id,b)
                    row=frame.loc[(frame.method==method)&np.isclose(frame.budget,b)&(frame.cohort==str(cohort_name))]
                    if len(row)!=1 or int(row.tp.iloc[0])!=result['tp'] or not np.isclose(row.precision.iloc[0],result['precision'],atol=1e-12):
                        raise ValueError('Extended metrics do not match saved selections')
                    additions.append(dict(method=method,budget=b,cohort=str(cohort_name),ap=result['ap'],lift=result['lift']))
        extended=frame.merge(pd.DataFrame(additions),on=['method','budget','cohort'],validate='one_to_one')
        file=new/'metric_evaluation'/(uid+'.extended.csv')
        io.write_csv(file,extended)
        io.write_json(marker,dict(complete=True,files={str(file.relative_to(new)):sha256(file)}))
        print('METRIC TARGET DONE '+case['name']+' '+uid,flush=True)


def summarize_case(case,out):
    new=out/'runs'/case['name'];protocol,config=case_config(case)
    parts=[]
    for p in sorted((new/'metric_evaluation').glob('*.complete.json')):
        if not io.check_done(new,p):raise ValueError('Incomplete extended metrics')
        parts.append(pd.read_csv(p.with_name(p.name.replace('.complete.json','.extended.csv'))))
    if len(parts)!=case['tasks']:raise ValueError('Extended task count mismatch')
    frame=pd.concat(parts,ignore_index=True)
    keys=['dataset','seed','test_course','source_courses','draw','budget','method']
    mean_cols=['recall','precision','ap','lift']+[c for c in frame if c.startswith(('jaccard_','extra_per_1000_'))]
    sum_cols=['tp','n','positives','selected']+[c for c in frame if c.startswith('net_tp_')]
    course=frame.groupby(keys,as_index=False)[mean_cols].mean().merge(
        frame.groupby(keys,as_index=False)[sum_cols].sum(),on=keys,validate='one_to_one')
    output=new/'metric_analysis'
    io.write_csv(output/'course_metrics.csv',course)
    # Keep source-course counts and draws separate: do not pool repeated learning-curve targets.
    seedkeys=['dataset','seed','source_courses','draw','budget','method']
    seed=course.groupby(seedkeys,as_index=False)[mean_cols].mean().merge(
        course.groupby(seedkeys,as_index=False)[sum_cols].sum(),on=seedkeys,validate='one_to_one')
    io.write_csv(output/'seed_metrics.csv',seed)
    means=seed.groupby(['dataset','source_courses','draw','budget','method'])[['recall','precision','ap','lift','tp']].agg(['mean','std'])
    means.columns=['_'.join(c) for c in means.columns]
    io.write_csv(output/'absolute_mean_sd.csv',means.reset_index())
    diagnostic=pd.concat([pd.read_csv(p) for p in sorted((new/'evaluation').glob('*.diagnostics.csv'))],ignore_index=True)
    io.write_csv(output/'mechanism_course_records.csv',diagnostic)
    io.write_csv(output/'mechanism_summary.csv',diagnostic.groupby(
        ['dataset','source_courses','draw','budget'],as_index=False)[
        ['H_pp','G_pp','remaining_pp','oracle_match','switched_from_fixed','harmful_switch','regret_pp']].mean())
    parameters=[]
    for p in sorted((new/'tuning').glob('*.frozen.json')):
        f=io.read_json(p)
        parameters.append({**{k:f[k] for k in ['dataset','seed','test_course','source_courses','draw']},**f['tuned']})
    io.write_csv(output/'frozen_parameters.csv',pd.DataFrame(parameters))
    comparisons=[]
    for (dataset,sources,draw,budget),part in course.groupby(['dataset','source_courses','draw','budget']):
        for metric in ['recall','ap','precision','lift']:
            pivot=part.pivot(index=['seed','test_course'],columns='method',values=metric)
            if pivot.isna().any().any():raise ValueError('Undefined or unpaired course metric')
            for comparator in ['source_fixed','existing_gate','tuned_rule','tuned_lr','tuned_hgb','fixed_blend','random']:
                delta=(pivot.tuned_ours-pivot[comparator]).groupby('test_course').mean()
                lo,hi=io.bootstrap_interval(delta.to_numpy(),config['bootstrap'],config['seeds'][0])
                scale=100 if metric in ['recall','precision'] else 1
                comparisons.append(dict(dataset=dataset,source_courses=sources,draw=draw,budget=budget,
                    metric=metric,comparator=comparator,mean_delta=scale*delta.mean(),
                    ci_low=scale*lo,ci_high=scale*hi,unit='pp' if scale==100 else 'raw',
                    higher=int((delta>1e-12).sum()),tied=int((abs(delta)<=1e-12).sum()),lower=int((delta< -1e-12).sum())))
    io.write_csv(output/'paired_metrics.csv',pd.DataFrame(comparisons))
    io.write_json(output/'complete.json',dict(complete=True,tasks=case['tasks'],
        final_method='tuned_ours',original_control_excluded_from_final_tables=True,
        tp_aggregation='Sum records within seed; mean/SD across seeds, not a sum across seeds',
        all_methods_preserved_in_internal_outputs=True,external_validation=False))


def run_cases(stage,cases,out,source_root,policy):
    for index,case in enumerate(cases,1):
        print(f'METRIC {stage.upper()} CASE {index}/{len(cases)} '+case['name'],flush=True)
        if stage=='select':refine_case(case,policy,source_root,out)
        elif stage=='evaluate':
            protocol,config=case_config(case)
            new=out/'runs'/case['name']
            # Existing evaluator remains unchanged; monkeypatch only this new process.
            with patch.object(quality,'score_pool',cached_scores):opt.evaluate(protocol,config,[case['dataset']],new)
            extend_evaluation(case,out)
        elif stage=='summarize':summarize_case(case,out)


def make_manifest(source_root,policy_path,cases):
    original=io.read_json(source_root/'suite_manifest.json')
    for relative,digest in {**original['hashes'],**original['input_hashes']}.items():
        if sha256(ROOT/relative)!=digest:
            raise ValueError('Original suite code/config/input changed: '+relative)
    paths=[Path(__file__),policy_path,ROOT/'selector_optimization.py',ROOT/'decision_principle_experiments.py',
           ROOT/'support_selection/quality.py',ROOT/'support_selection/experiment.py',
           source_root/'suite_manifest.json',source_root/'suite_complete.json']
    for case in cases:
        paths.extend([ROOT/case['config'],ROOT/case['protocol'],ROOT/case['result']/'tuning_complete.json'])
        for p in (ROOT/case['result']/'tuning').glob('*.complete.json'):
            if not io.check_done(ROOT/case['result'],p):raise ValueError('Prior source artifact incomplete')
            paths.append(p)
    return dict(policy=io.read_json(policy_path),source_suite=str(source_root),cases=cases,
        hashes={str(p.relative_to(ROOT)):sha256(p) for p in paths},
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['plan','select','evaluate','summarize','all'],default='plan')
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/selector_replacement_suite_v1')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/metric_refinement_v1')
    parser.add_argument('--policy',type=Path,default=ROOT/'configs/metric_refinement.json')
    args=parser.parse_args()
    source=args.source.resolve();out=args.out.resolve();policy=io.read_json(args.policy)
    suite=io.read_json(source/'suite_manifest.json');cases=suite['cases']
    if out==source or out.is_relative_to(source):raise ValueError('Do not write into the running/archived source suite')
    if args.stage=='plan':
        print(json.dumps(dict(source_suite_complete=(source/'suite_complete.json').exists(),cases=len(cases),
            source_reselection_tasks=sum(c['tasks'] for c in cases),target_reevaluation_tasks=sum(c['tasks'] for c in cases),
            selection=policy['selection_order'],cache_only=True,training_started=False),indent=2));return
    if not (source/'suite_complete.json').is_file() or not io.read_json(source/'suite_complete.json')['complete']:
        raise ValueError('Wait until the current replacement suite finishes; do not pause or restart it')
    manifest=make_manifest(source,args.policy,cases)
    out=io.lock_output(out,manifest)
    with threadpool_limits(limits=2):
        if args.stage in ['select','all']:
            run_cases('select',cases,out,source,policy)
            freezes=[]
            for case in cases:
                new=out/'runs'/case['name'];opt.frozen_index(new)
                path=new/'tuning_complete.json';freezes.append(dict(path=str(path.relative_to(out)),sha256=sha256(path)))
            io.write_json(out/'all_parameters_frozen.json',dict(complete=True,freezes=freezes,target_labels_used=False))
        if args.stage in ['evaluate','all']:
            index=io.read_json(out/'all_parameters_frozen.json')
            for item in index['freezes']:
                path=out/item['path']
                if sha256(path)!=item['sha256']:raise ValueError('Metric-selected parameter freeze changed')
                opt.frozen_index(path.parent)
            run_cases('evaluate',cases,out,source,policy)
        if args.stage in ['summarize','all']:
            run_cases('summarize',cases,out,source,policy)
            for filename in ['absolute_mean_sd.csv','paired_metrics.csv','seed_metrics.csv','mechanism_summary.csv','frozen_parameters.csv']:
                parts=[]
                for case in cases:
                    frame=pd.read_csv(out/'runs'/case['name']/'metric_analysis'/filename)
                    for key in ['name','family','cutoff','mode']:frame['experiment_'+key]=case[key]
                    parts.append(frame)
                io.write_csv(out/'analysis'/filename,pd.concat(parts,ignore_index=True))
            primary=pd.read_csv(out/'analysis/absolute_mean_sd.csv')
            keep=['random','tuned_rule','tuned_lr','tuned_hgb','fixed_blend','source_fixed','existing_gate','tuned_ours']
            paper=primary.loc[(primary.experiment_family=='primary')&primary.method.isin(keep)].copy()
            paper['method_order']=paper.method.map({method:i for i,method in enumerate(keep)})
            paper['display_method']=paper.method.map(dict(random='Random',tuned_rule='Tuned rule',
                tuned_lr='Tuned LR',tuned_hgb='Tuned HGB',fixed_blend='Fixed blend',source_fixed='Source-fixed',
                existing_gate='Existing Gate',tuned_ours='Ours'))
            paper=paper.sort_values(['dataset','budget','method_order'])
            io.write_csv(out/'analysis/paper_performance.csv',paper.drop(columns='method_order'))
            io.write_json(out/'complete.json',dict(complete=True,final_ours='tuned_ours',all_results_retained=True,
                manuscript_modified=False,scope=policy['scope']))
            print('METRIC REFINEMENT COMPLETE '+str(out/'analysis'),flush=True)


if __name__=='__main__':main()

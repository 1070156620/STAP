"""Source-only parameter search, immutable per-task freeze, then target evaluation.

Reuses content-keyed candidate score caches. Does not edit manuscript or old runs.
Previously inspected populations remain exploratory. No runtime benchmark.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'2')
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import decision_principle_experiments as gate
from support_selection import quality
from support_selection.data import ROOT, sha256
from support_selection.experiment import descriptor, select_top, stable_hash
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib_selector_optimization'))


def parameter_grid(config):
    grid=[]
    for pool in config['pools']:
        grid.append(dict(pool=pool,features='source_validation',alpha=10.,margin=0.))
        for features,alpha,margin in itertools.product(
                [f for f in config['feature_modes'] if f!='source_validation'],
                config['alphas'],config['margins']):
            grid.append(dict(pool=pool,features=features,alpha=alpha,margin=margin))
    return grid


def theta_id(theta):
    return hashlib.sha256(json.dumps(theta,sort_keys=True).encode()).hexdigest()[:16]


def restricted(evidence,theta,config):
    x,u,tags,names=evidence
    if theta['pool']=='core' and not set(config['core_candidates']).issubset(names):
        raise ValueError('Core candidate identities do not match the scoring pool')
    allowed=names if theta['pool']=='full' else [n for n in names if n in config['core_candidates']]
    if not allowed or not any(n.startswith('rule_') for n in allowed):
        raise ValueError('Empty candidate pool or no fallback rule')
    ix=[names.index(n) for n in allowed]
    return x,u[:,ix],tags,allowed


def utility_estimates(evidence,query,budget,theta,cfg,config):
    x,u,tags,names=restricted(evidence,theta,config)
    ix=[i for i,t in enumerate(tags) if np.isclose(t['budget'],budget)]
    if not ix: raise ValueError('No source evidence at requested capacity')
    means=u[ix].mean(axis=0)
    fixed=names[int(np.argmax(means))]
    rules=[n for n in names if n.startswith('rule_')]
    rule=max(rules,key=lambda n:means[names.index(n)])
    fallback=len({t['course'] for t in tags})<cfg['min_meta_courses']
    if theta['features']=='source_validation' or fallback:
        estimates=dict(zip(names,means.astype(float)))
    else:
        if theta['features']=='structural': x,query=x[:,:4],query[:4]
        counts=pd.Series([t['course'] for t in tags]).value_counts()
        weights=np.array([1/counts[t['course']] for t in tags])
        # Multi-output Ridge is algebraically equivalent to one Ridge per candidate.
        model=make_pipeline(StandardScaler(),Ridge(alpha=theta['alpha']))
        model.fit(x,u,ridge__sample_weight=weights)
        estimates=dict(zip(names,model.predict(query[None,:])[0].astype(float)))
    return dict(fixed=fixed,rule=rule,estimates=estimates,fallback=fallback,
                source_only=theta['features']=='source_validation')


def apply_margin(info,margin):
    if info['source_only'] or info['fallback']:return info['fixed']
    winner=max(info['estimates'],key=info['estimates'].get)
    return winner if info['estimates'][winner]-info['estimates'][info['rule']]>=margin else info['rule']


def choice(evidence,query,budget,theta,cfg,config):
    info=utility_estimates(evidence,query,budget,theta,cfg,config)
    return apply_margin(info,theta['margin']),info


def source_bank(source,cols,cfg,cache):
    bank=[]
    for course,fit,val in quality.source_folds(source,cfg,middle=True):
        if course in set(fit.course) or set(fit.student_id)&set(val.student_id):
            raise ValueError('Middle fold course/student leakage')
        evidence=quality.build_evidence(fit,cols,cfg,cache)
        if not len(evidence[0]) or course in {t['course'] for t in evidence[2]}:
            raise ValueError('Empty or contaminated inner evidence')
        scores,_=quality.score_pool(fit,val.drop(columns='y'),cols,cfg,cache)
        recalls={b:{n:gate.course_recall(val,scores[n],b) for n in evidence[3]} for b in cfg['budgets']}
        if any(v is None for byname in recalls.values() for v in byname.values()):
            raise ValueError('Undefined middle-course recall; no silent skipping')
        bank.append(dict(course=str(course),evidence=evidence,
            query={b:descriptor(fit,val.drop(columns='y'),b,cols) for b in cfg['budgets']},
            recalls=recalls,training_courses=sorted(map(str,fit.course.unique())),student_overlap=0))
    if len(bank)<2: raise ValueError('Fewer than two eligible middle courses')
    return bank


def search_source(bank,grid,cfg,config):
    rows=[]
    # Share each utility regression across seven margins; base scores are reused too.
    memo={}
    for theta in grid:
        uid=theta_id(theta)
        for fold in bank:
            for b in cfg['budgets']:
                key=(fold['course'],b,theta['pool'],theta['features'],theta['alpha'])
                if key not in memo:
                    memo[key]=utility_estimates(fold['evidence'],fold['query'][b],b,theta,cfg,config)
                info=memo[key]
                selected=apply_margin(info,theta['margin'])
                rows.append(dict(theta_id=uid,validation_course=fold['course'],budget=b,
                    recall=fold['recalls'][b][selected],candidate=selected,
                    switched_from_fixed=selected!=info['fixed'],fallback=info['fallback'],**theta))
    frame=pd.DataFrame(rows)
    summary=frame.groupby(['theta_id','budget'],as_index=False).agg(
        recall=('recall','mean'),switch_rate=('switched_from_fixed','mean'),
        validation_courses=('validation_course','nunique'))
    theta_map={theta_id(t):t for t in grid}
    def best(subset):
        eligible=summary.loc[np.isclose(summary.budget,config['primary_budget']) & summary.theta_id.isin(subset)]
        if eligible.empty:raise ValueError('No eligible parameter settings')
        top=eligible.recall.max()
        eligible=eligible.loc[eligible.recall>=top-1e-12]
        def tie(row):
            t=theta_map[row.theta_id]
            return (row.switch_rate,['source_validation','structural','full'].index(t['features']),
                    -t['alpha'],-t['margin'],0 if t['pool']=='full' else 1)
        win=min(eligible.itertuples(),key=tie)
        return theta_map[win.theta_id]
    selected=best(theta_map)
    feature_best={f:best([k for k,t in theta_map.items() if t['features']==f]) for f in config['feature_modes']}
    # Original alpha is source-tuned at the designated primary capacity, using its original grid.
    originals=[t for t in grid if t['pool']=='full' and t['features']=='full'
               and t['alpha'] in cfg['selector_alphas'] and np.isclose(t['margin'],cfg['gate_margin'])]
    orig_scores=[]
    for t in originals:
        r=summary.loc[(summary.theta_id==theta_id(t)) & np.isclose(summary.budget,config['primary_budget'])].iloc[0]
        orig_scores.append((float(r.recall),t['alpha'],t))
    original=max(orig_scores,key=lambda item:(item[0],item[1]))[2]
    return selected,original,feature_best,frame,summary


def tasks(protocol,names,pilot=False):
    for name in names:
        df=gate.dataset_frame(ROOT/protocol['datasets'][name]['processed'])
        df,cols=quality.transform_features(df,protocol['feature_mode'])
        for seed in (protocol['seeds'][:1] if pilot else protocol['seeds']):
            cfg=gate.experiment_config(protocol,name,seed)
            for index,(course,nsource,draw,train,test) in enumerate(quality.plan_runs(df,cfg)):
                if pilot and index>0:break
                uid=gate.task_id(name,seed,course,nsource,draw)
                tag=dict(dataset=name,seed=seed,test_course=str(course),source_courses=nsource,draw=draw)
                if set(train.student_id)&set(test.student_id) or set(train.course)&set(test.course):
                    raise ValueError('Outer leakage')
                if len(train)<cfg['min_train'] or len(test)<cfg['min_test'] or train.y.nunique()<2:
                    raise ValueError('Ineligible outer task')
                yield uid,tag,train,test,cols,cfg


def optimize(protocol,config,names,out,pilot=False):
    grid=parameter_grid(config)
    entries=[]
    folder=out/('pilot' if pilot else 'tuning')
    for uid,tag,train,target,cols,cfg in tasks(protocol,names,pilot):
        marker=folder/(uid+'.complete.json')
        if gate.check_done(out,marker):
            print('TUNE REUSE '+uid,flush=True)
        else:
            print('TUNE START '+uid,flush=True)
            bank=source_bank(train,cols,cfg,ROOT/'.cache/quality_scores')
            selected,original,features,rows,summary=search_source(bank,grid,cfg,config)
            base=folder/uid
            files=[Path(str(base)+'.source_grid.csv'),Path(str(base)+'.source_summary.csv'),Path(str(base)+'.frozen.json')]
            gate.write_csv(files[0],rows)
            gate.write_csv(files[1],summary)
            gate.write_json(files[2],dict(**tag,tuned=selected,original=original,feature_best=features,
                source_training_courses=sorted(map(str,train.course.unique())),
                middle_audit=[{k:f[k] for k in ['course','training_courses','student_overlap']} for f in bank],
                objective='source middle-course macro recall at 20 percent capacity',
                target_labels_used=False,target_scores_computed=False,source_configs=len(grid)))
            gate.write_json(marker,dict(complete=True,files={str(p.relative_to(out)):sha256(p) for p in files}))
            print('TUNE DONE '+uid+' '+json.dumps(selected),flush=True)
        entries.append(dict(task=uid,freeze=str((folder/(uid+'.frozen.json')).relative_to(out)),
                            sha256=sha256(folder/(uid+'.frozen.json'))))
        gate.write_json(out/'progress.json',dict(stage='pilot' if pilot else 'tune',completed=len(entries),latest=uid))
    if not entries:raise ValueError('No tasks selected')
    gate.write_json(out/('pilot_complete.json' if pilot else 'tuning_complete.json'),dict(
        complete=True,tasks=len(entries),configurations_per_task=len(grid),frozen=entries,
        target_labels_used=False,no_target_based_shortlist=True))


def frozen_index(out):
    info=gate.read_json(out/'tuning_complete.json')
    if not info['complete']:raise ValueError('All tasks must finish tuning before evaluation')
    result={}
    for item in info['frozen']:
        path=out/item['freeze']
        if sha256(path)!=item['sha256']:raise ValueError('Frozen parameter file was modified')
        result[item['task']]=path
    return result


def evaluate(protocol,config,names,out):
    freezes=frozen_index(out)  # Verify all configurations before reading any target performance.
    completed=0
    for uid,tag,train,target,cols,cfg in tasks(protocol,names):
        if uid not in freezes:raise ValueError('Task has no frozen source-only configuration')
        marker=out/'evaluation'/(uid+'.complete.json')
        if gate.check_done(out,marker):
            completed+=1
            print(f'EVAL REUSE {completed}/{len(freezes)} '+uid,flush=True)
            continue
        frozen=gate.read_json(freezes[uid])
        print(f'EVAL START {completed+1}/{len(freezes)} '+uid,flush=True)
        cache=ROOT/'.cache/quality_scores'
        evidence=quality.build_evidence(train,cols,cfg,cache)
        # Structural guard: target labels do not cross the scoring/selection boundary.
        target_x=target.drop(columns='y')
        scores,_=quality.score_pool(train,target_x,cols,cfg,cache)
        scores['fixed_blend']=(rankdata(scores['lr_C1'])+rankdata(scores['rule_inactivity']))/(2*len(target))
        scores['random']=np.array([stable_hash(i,cfg['seed']) for i in target.record_id],dtype=float)
        bank=source_bank(train,cols,cfg,cache)
        rows,decisions,diagnostics=[],[],[]
        prediction=target[['record_id','student_id','course','cohort','y']].copy()
        for b in cfg['budgets']:
            query=descriptor(train,target_x,b,cols)
            tuned,tinfo=choice(evidence,query,b,frozen['tuned'],cfg,config)
            original,_=choice(evidence,query,b,frozen['original'],cfg,config)
            rule=gate.select_family(evidence,b,'rule')
            model=gate.select_family(evidence,b,'model')
            source_gains=[]
            for f in bank:
                r=gate.select_family(f['evidence'],b,'rule');m=gate.select_family(f['evidence'],b,'model')
                source_gains.append(f['recalls'][b][m]-f['recalls'][b][r])
            stats=gate.lower_source_quantile(source_gains,protocol,cfg['seed'])
            mapping=dict(tuned_rule=rule,tuned_lr=gate.select_family(evidence,b,'lr'),
                tuned_hgb=gate.select_family(evidence,b,'hgb'),source_fixed=gate.select_family(evidence,b,'all'),
                original_ours=original,tuned_ours=tuned,
                existing_gate=gate.evidence_gate(rule,model,stats,protocol['primary_delta']),
                fixed_blend='fixed_blend',random='random')
            no_margin=copy.deepcopy(frozen['tuned']);no_margin['margin']=0
            mapping['tuned_no_margin']=choice(evidence,query,b,no_margin,cfg,config)[0]
            for features,theta in frozen['feature_best'].items():
                mapping['feature_'+features]=choice(evidence,query,b,theta,cfg,config)[0]
            masks={name:np.zeros(len(target),dtype=bool) for name in mapping}
            for _,cohort in target.groupby('cohort',sort=True):
                ix=target.index.get_indexer(cohort.index)
                local={name:select_top(scores[candidate][ix],cohort.record_id,b) for name,candidate in mapping.items()}
                if len({int(mask.sum()) for mask in local.values()})!=1:raise ValueError('Capacity mismatch')
                y=cohort.y.to_numpy()
                for name,mask in local.items():
                    masks[name][ix]=mask
                    tp=int(y[mask].sum());positive=int(y.sum());k=int(mask.sum())
                    row=dict(**tag,budget=b,cohort=str(cohort.cohort.iloc[0]),method=name,
                        candidate=mapping[name],n=len(y),positives=positive,selected=k,tp=tp,
                        recall=tp/positive if positive else np.nan,precision=tp/k)
                    for comparator in ['source_fixed','original_ours','existing_gate','tuned_rule']:
                        baseline=local[comparator]
                        row['net_tp_vs_'+comparator]=tp-int(y[baseline].sum())
                        row['jaccard_vs_'+comparator]=float((mask&baseline).sum()/(mask|baseline).sum())
                        row['extra_per_1000_vs_'+comparator]=1000*row['net_tp_vs_'+comparator]/len(y)
                    rows.append(row)
            for name,mask in masks.items():prediction[f'selected__{b:g}__{name}']=mask.astype(np.uint8)
            recalls={n:gate.course_recall(target,scores[n],b) for n in evidence[3]}
            if any(r is None for r in recalls.values()):raise ValueError('Target course has undefined recall')
            oracle=max(recalls.values());fixed_r=recalls[mapping['source_fixed']];tuned_r=recalls[tuned]
            diagnostics.append(dict(**tag,budget=b,oracle_recall=oracle,source_fixed_recall=fixed_r,
                tuned_recall=tuned_r,H_pp=100*(oracle-fixed_r),G_pp=100*(tuned_r-fixed_r),
                remaining_pp=100*(oracle-tuned_r),oracle_match=abs(oracle-tuned_r)<=1e-12,
                switched_from_fixed=tuned!=mapping['source_fixed'],
                harmful_switch=tuned!=mapping['source_fixed'] and tuned_r<fixed_r-1e-12,
                regret_pp=100*(oracle-tuned_r),target_label_diagnostic_only=True))
            decisions.append(dict(budget=b,mapping=mapping,tuned_theta=frozen['tuned'],
                fallback=tinfo['fallback'],source_only=tinfo['source_only'],
                estimates=tinfo['estimates'],target_labels_used_for_selection=False))
        base=out/'evaluation'/uid
        files=[Path(str(base)+'.metrics.csv'),Path(str(base)+'.decisions.json'),
               Path(str(base)+'.diagnostics.csv'),out/'predictions'/(uid+'.csv.gz')]
        gate.write_csv(files[0],pd.DataFrame(rows))
        gate.write_json(files[1],dict(**tag,choices=decisions,outer_student_overlap=0,
            target_y_sent_to_scoring=False,frozen_parameter_sha256=sha256(freezes[uid])))
        gate.write_csv(files[2],pd.DataFrame(diagnostics))
        gate.write_csv(files[3],prediction)
        gate.write_json(marker,dict(complete=True,files={str(p.relative_to(out)):sha256(p) for p in files}))
        completed+=1
        gate.write_json(out/'progress.json',dict(stage='evaluate',completed=completed,expected=len(freezes),latest=uid))
        print(f'EVAL DONE {completed}/{len(freezes)} '+uid,flush=True)
    if completed!=len(freezes):raise ValueError('Incomplete target task coverage')
    gate.write_json(out/'run_complete.json',dict(complete=True,tasks=completed,all_outcomes_retained=True))


def summarize(config,out):
    if not gate.read_json(out/'run_complete.json')['complete']:raise ValueError('Evaluation incomplete')
    parts=[pd.read_csv(p) for p in sorted((out/'evaluation').glob('*.metrics.csv'))]
    df=pd.concat(parts,ignore_index=True)
    keys=['dataset','seed','test_course','budget','method']
    numeric=[c for c in df if c in ['recall','precision'] or c.startswith(('jaccard_','extra_per_1000_'))]
    courses=df.groupby(keys,as_index=False)[numeric].mean()
    counts=df.groupby(keys,as_index=False)[['n','positives','selected','tp']+[c for c in df if c.startswith('net_tp_')]].sum()
    courses=courses.merge(counts,on=keys,validate='one_to_one')
    output=out/'analysis'
    gate.write_csv(output/'course_metrics.csv',courses)
    seed=courses.groupby(['dataset','seed','budget','method'],as_index=False)[numeric].mean()
    gate.write_csv(output/'seed_metrics.csv',seed)
    mean_sd=seed.groupby(['dataset','budget','method'])[['recall','precision']].agg(['mean','std'])
    mean_sd.columns=['_'.join(col) for col in mean_sd.columns]
    gate.write_csv(output/'absolute_mean_sd.csv',mean_sd.reset_index())
    contrasts=[]
    for (dataset,budget),part in courses.groupby(['dataset','budget']):
        pivot=part.pivot(index=['seed','test_course'],columns='method',values='recall')
        if pivot.isna().any().any():raise ValueError('Unpaired/missing course metrics')
        for comparator in ['source_fixed','original_ours','existing_gate','tuned_rule','tuned_lr','tuned_hgb','fixed_blend','random']:
            delta=(pivot['tuned_ours']-pivot[comparator]).groupby('test_course').mean()
            lo,hi=gate.bootstrap_interval(delta.to_numpy(),config['bootstrap'],config['seeds'][0])
            contrasts.append(dict(dataset=dataset,budget=budget,comparator=comparator,
                mean_delta_pp=100*delta.mean(),ci_low_pp=100*lo,ci_high_pp=100*hi,
                higher=int((delta>1e-12).sum()),tied=int((abs(delta)<=1e-12).sum()),lower=int((delta< -1e-12).sum())))
    gate.write_csv(output/'paired_comparisons.csv',pd.DataFrame(contrasts))
    recipientcols=[c for c in courses if c.startswith(('net_tp_','jaccard_','extra_per_1000_'))]
    recipients=courses.groupby(['dataset','seed','budget','method'])[recipientcols].agg(
        {c:'sum' if c.startswith('net_tp_') else 'mean' for c in recipientcols}).reset_index()
    gate.write_csv(output/'recipient_effects_per_seed.csv',recipients)
    diagnostic=pd.concat([pd.read_csv(p) for p in sorted((out/'evaluation').glob('*.diagnostics.csv'))],ignore_index=True)
    gate.write_csv(output/'mechanism_course_records.csv',diagnostic)
    gate.write_csv(output/'mechanism_summary.csv',diagnostic.groupby(['dataset','budget'],as_index=False)[
        ['H_pp','G_pp','remaining_pp','oracle_match','switched_from_fixed','harmful_switch','regret_pp']].mean())
    downside=[]
    for (dataset,budget),part in diagnostic.groupby(['dataset','budget']):
        switches=part.loc[part.switched_from_fixed]
        downside.append(dict(dataset=dataset,budget=budget,course_seed_tasks=len(part),
            switched_tasks=len(switches),harmful_switches=int(part.harmful_switch.sum()),
            loss_rate_given_switch=float(switches.harmful_switch.mean()) if len(switches) else None,
            min_gain_pp=float(part.G_pp.min()),gain_10th_percentile_pp=float(part.G_pp.quantile(.1)),
            mean_loss_pp=float(np.minimum(part.G_pp,0).mean())))
    gate.write_csv(output/'switch_downside_summary.csv',pd.DataFrame(downside))
    frozen=[gate.read_json(p) for p in sorted((out/'tuning').glob('*.frozen.json'))]
    theta_rows=[{**{k:f[k] for k in ['dataset','seed','test_course']},**f['tuned']} for f in frozen]
    gate.write_csv(output/'frozen_parameters.csv',pd.DataFrame(theta_rows))
    source=pd.concat([pd.read_csv(p).assign(dataset=p.name.split('_')[0]) for p in sorted((out/'tuning').glob('*.source_grid.csv'))],ignore_index=True)
    gate.write_csv(output/'source_parameter_sensitivity.csv',source.groupby(
        ['dataset','features','pool','alpha','margin','budget'],as_index=False).recall.mean())
    # Optional vector-only plots; no bitmap assets, no manuscript mutation.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'font.size':9})
    for field in ['margin','alpha']:
        fig,axes=plt.subplots(1,len(config['dataset_names']),figsize=(4*len(config['dataset_names']),3.4),squeeze=False)
        for ax,name in zip(axes[0],config['dataset_names']):
            primary=source.loc[(source.dataset==name)&np.isclose(source.budget,config['primary_budget'])
                &(source.pool=='full')&(source.features=='full')]
            # One-factor plots hold the other variable at the original setting.
            primary=primary.loc[np.isclose(primary.alpha,10) if field=='margin' else np.isclose(primary.margin,.01)]
            by=primary.groupby(field).recall.mean()
            ax.plot(by.index*(100 if field=='margin' else 1),100*by.values,'o-',label='Source validation')
            if field=='alpha':ax.set_xscale('log')
            ax.set(title=name,xlabel='Margin (pp)' if field=='margin' else 'Ridge alpha',ylabel='Source recall (%)')
            ax.grid(alpha=.2)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper right')
        fig.tight_layout(rect=(0,0,1,.88))
        (output/'figures').mkdir(exist_ok=True)
        fig.savefig(output/'figures'/f'source_{field}_sensitivity.pdf',bbox_inches='tight')
        plt.close(fig)
    gate.write_json(output/'report_status.json',dict(complete=True,compiled=False,
        all_results_retained=True,bootstrap_unit='course after averaging seeds',
        primary_budget=config['primary_budget'],scope=config['scope'],
        external_validation=False,model_weights_or_thresholds_selected_on_target=False))
    print('SUMMARY COMPLETE '+str(output),flush=True)
    print(pd.DataFrame(contrasts).query("budget == 0.2 and comparator == 'source_fixed'").to_string(index=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['plan','pilot','tune','evaluate','summarize','all'],default='plan')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/selector_optimization.json')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/selector_optimization_v1')
    parser.add_argument('--datasets',nargs='+',choices=['oulad','kdd2015','mooccubex'])
    args=parser.parse_args()
    config=gate.read_json(args.config)
    protocol=gate.read_json(ROOT/config['base_protocol'])
    protocol.update(seeds=config['seeds'],budgets=config['budgets'],threads=config['threads'])
    names=args.datasets or list(protocol['datasets'])
    if len(set(names))!=len(names):raise ValueError('Repeated datasets')
    config['dataset_names']=names
    counts={name:int(gate.dataset_frame(ROOT/protocol['datasets'][name]['processed']).course.nunique()) for name in names}
    total=sum(counts.values())*len(config['seeds'])
    if args.stage=='plan':
        print(json.dumps(dict(courses=counts,source_configurations_per_task=len(parameter_grid(config)),
            pilot_source_tasks=len(names),full_source_tuning_tasks=total,target_evaluation_tasks=total,
            budgets=config['budgets'],target_methods=13,training_started=False,scope=config['scope']),indent=2))
        return
    manifest=gate.build_manifest(protocol,ROOT/config['base_protocol'],names)
    manifest['optimization_config']=config
    manifest['hashes']['selector_optimization.py']=sha256(Path(__file__))
    manifest['hashes']['configs/selector_optimization.json']=sha256(args.config)
    out=gate.lock_output(args.out,manifest)
    with threadpool_limits(limits=config['threads']):
        if args.stage in ['pilot','tune','all']:optimize(protocol,config,names,out,pilot=args.stage=='pilot')
        if args.stage in ['evaluate','all']:evaluate(protocol,config,names,out)
        if args.stage in ['summarize','all']:summarize(config,out)


if __name__=='__main__':main()

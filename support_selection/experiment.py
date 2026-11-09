"""Nested, student-purged strategy comparison; CPU only, bounded parallelism."""
from pathlib import Path
import hashlib
import math
import time
import json
import platform
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from .data import ROOT, FEATURES, dump_json, sha256

RULES=["rule_inactivity","rule_low_activity","rule_low_recent_share"]
MODELS=["logistic","hist_gradient"]
CANDIDATES=RULES+MODELS

def stable_hash(value, seed=0):
    return int(hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()[:16],16)/2**64

def select_top(scores, ids, budget):
    scores=np.asarray(scores,float)
    if not np.isfinite(scores).all():raise ValueError("Non-finite ranking scores")
    if not 0 < budget <= 1:raise ValueError("Budget must be in (0,1]")
    n=len(scores); k=min(n,max(1,math.ceil(n*budget)))
    # Deterministic identifier hash resolves ties; outcome is never a tiebreaker.
    tie=np.array([stable_hash(x) for x in ids])
    order=np.lexsort((tie,-scores)); selected=np.zeros(n,bool);selected[order[:k]]=True
    return selected

def metrics(y, scores, ids, budget, probability=False):
    y=np.asarray(y,int); m=select_top(scores,ids,budget)
    tp=int(y[m].sum()); p=int(y.sum());k=int(m.sum());fp=k-tp
    return dict(n=len(y),positives=p,selected=k,tp=tp,fp=fp,fn=p-tp,
                recall=tp/p if p else np.nan,precision=tp/k,
                roc_auc=roc_auc_score(y,scores) if 0<p<len(y) else np.nan,
                pr_auc=average_precision_score(y,scores) if p else np.nan,
                brier=brier_score_loss(y,scores) if probability else np.nan)

def feature_columns(mode):
    if mode=="behavior":return FEATURES
    if mode=="activity_only":return ["log_clicks","log_events","log_active_days"]
    raise ValueError(f"Unknown feature mode: {mode}")

def rule_scores(df):
    return {"rule_inactivity":df.inactivity_fraction.to_numpy(float),
            "rule_low_activity":-df.log_clicks.to_numpy(float),
            "rule_low_recent_share":-df.recent_fraction.to_numpy(float)}

def scores_for(train, test, cols, cfg):
    scores=rule_scores(test); fitted={};times={}
    for name in MODELS:
        if train.y.nunique()<2:
            scores[name]=np.repeat(float(train.y.mean()),len(test));continue
        model=(make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),
                             StandardScaler(),LogisticRegression(C=1,max_iter=500,random_state=cfg['seed']))
               if name=="logistic" else HistGradientBoostingClassifier(max_iter=cfg['hgb_iterations'],
                     max_leaf_nodes=15,l2_regularization=1,early_stopping=False,random_state=cfg['seed']))
        start=time.perf_counter();model.fit(train[cols],train.y);times[name]=time.perf_counter()-start
        scores[name]=model.predict_proba(test[cols])[:,1];fitted[name]=model
    return scores,fitted,times

def descriptor(train, target, budget, cols=None):
    """Unlabelled cohort descriptor. Explicitly does not access target.y."""
    cols=FEATURES if cols is None else cols
    mu=train[cols].mean().to_numpy();sd=train[cols].std().fillna(0).to_numpy()+1e-6
    tm=target[cols].mean().to_numpy()
    z=np.nan_to_num((tm-mu)/sd,nan=0,posinf=0,neginf=0)
    return np.r_[np.log1p(len(target)),budget,target[cols].isna().mean().mean(),
                 (target.log_clicks==0).mean(),np.clip(z,-10,10)]

def purge_students(train, test):
    return train.loc[~train.student_id.isin(test.student_id)].copy()

def outer_splits(df, regime):
    if regime=="course":
        for course in sorted(df.course.unique()):
            test=df.loc[df.course==course].copy()
            train=purge_students(df.loc[df.course!=course],test)
            yield f"course={course}",train,test
    elif regime=="temporal":
        for cohort in sorted(df.cohort.unique()):
            test=df.loc[df.cohort==cohort].copy()
            if test.start.nunique()!=1:raise ValueError("A cohort has multiple starts")
            # Transfer within the same course, earlier batches only; no label look-ahead.
            decision_date=test.start.iloc[0]+pd.Timedelta(days=int(test.cutoff.iloc[0]))
            train=purge_students(df.loc[(df.course==test.course.iloc[0]) & (df.start < test.start.iloc[0]) &
                                        (df.label_available <= decision_date)],test)
            yield f"cohort={cohort}",train,test
    else: raise ValueError(regime)

def inner_evidence(train, cols, cfg, budgets):
    """Leave-one-cohort-out, but student purged. Only outer TRAIN outcomes are accessed."""
    xs=[]; utilities={s:[] for s in CANDIDATES};metadata=[]
    cohorts=sorted(train.cohort.unique())
    if cfg.get('max_inner'):cohorts=cohorts[:cfg['max_inner']]
    for cohort in cohorts:
        val=train.loc[train.cohort==cohort].copy()
        fit=purge_students(train.loc[train.cohort!=cohort],val)
        # Cohorts from the future of inner validation are excluded in temporal evaluation.
        if cfg.get('_regime')=='temporal':
            decision_date=val.start.iloc[0]+pd.Timedelta(days=int(val.cutoff.iloc[0]))
            fit=fit.loc[(fit.start < val.start.iloc[0])&(fit.label_available <= decision_date)].copy()
        if len(fit)<cfg['min_train'] or len(val)<cfg['min_test'] or val.y.sum()==0:continue
        scores,_,_=scores_for(fit,val,cols,cfg)
        for b in budgets:
            xs.append(descriptor(fit,val,b,cols))
            for s in CANDIDATES:utilities[s].append(metrics(val.y,scores[s],val.record_id,b)['recall'])
            metadata.append({'cohort':cohort,'budget':b,'fit_n':len(fit),'validation_n':len(val)})
    return np.array(xs),utilities,metadata

def choose_strategies(x, utilities, query, inner_meta, margin):
    # Each source validation cohort contributes one utility per budget. Distinct cohorts, not rows, determine eligibility.
    n_cohorts=len({m['cohort'] for m in inner_meta})
    global_best=max(CANDIDATES,key=lambda s:np.mean(utilities[s])) if len(x) else 'rule_inactivity'
    best_rule=max(RULES,key=lambda s:np.mean(utilities[s])) if len(x) else 'rule_inactivity'
    if n_cohorts<3:
        return {'global_validation':global_best,'selector':global_best,'selector_no_margin':global_best,'selector_no_shift':global_best},'fallback: fewer than 3 inner validation cohorts',{}
    estimates={}
    for s in CANDIDATES:
        # A small, predefined regularized utility regressor; not an optimality guarantee.
        m=make_pipeline(StandardScaler(),Ridge(alpha=10))
        m.fit(x,utilities[s]);estimates[s]=float(m.predict(query[None,:])[0])
    best=max(CANDIDATES,key=lambda s:estimates[s])
    selected=best if estimates[best]-estimates[best_rule]>=margin else best_rule
    estimates_no_shift={}
    for s in CANDIDATES:
        m=make_pipeline(StandardScaler(),Ridge(alpha=10));m.fit(x[:,:4],utilities[s])
        estimates_no_shift[s]=float(m.predict(query[None,:4])[0])
    no_shift=max(CANDIDATES,key=lambda s:estimates_no_shift[s])
    if estimates_no_shift[no_shift]-estimates_no_shift[best_rule]<margin:no_shift=best_rule
    return {'global_validation':global_best,'selector':selected,'selector_no_margin':best,'selector_no_shift':no_shift},'estimated from outer-train inner validation only',estimates

def validate_frame(df):
    required={'dataset','outcome','record_id','student_id','course','cohort','start','label_available','cutoff','y',*FEATURES}
    if not required.issubset(df):raise ValueError(f"Missing columns: {required-set(df.columns)}")
    if df.record_id.duplicated().any():raise ValueError("Duplicate record_id within one cutoff")
    if not df.y.isin([0,1]).all():raise ValueError("Non-binary label")
    if df[['student_id','course','cohort']].isna().any().any():raise ValueError("Missing group IDs")
    if df.cutoff.nunique()!=1 or df.dataset.nunique()!=1 or df.outcome.nunique()!=1:raise ValueError("Mixed data/endpoint/cutoff")

def run_experiments(paths, cfg, output):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError(f"Output directory not empty: {output}; use a new run name")
    output.mkdir(parents=True,exist_ok=True)
    dump_json(output/'config.json',cfg)
    dump_json(output/'environment.json',{'python':sys.version,'platform':platform.platform(),
              'versions':{n:__import__(n).__version__ for n in ['numpy','pandas','sklearn','scipy']},
              'input_hashes':{str(p):sha256(p) for p in paths},
              'scope':'Risk identification and strategy selection; not intervention-effect estimation.'})
    rows=[];prediction_files=[];decisions=[];skips=[];fold_audit=[];timings=[]
    with threadpool_limits(limits=cfg['threads']):
        for path in paths:
            df=pd.read_csv(path,dtype={c:str for c in ['record_id','student_id','course','cohort']},parse_dates=['start','label_available'])
            validate_frame(df)
            if int(df.cutoff.iloc[0]) not in cfg['cutoffs']:continue
            for mode in cfg['feature_modes']:
                cols=feature_columns(mode)
                for regime in cfg['regimes']:
                    accepted=0
                    for fold,train,test in outer_splits(df,regime):
                        tag=dict(dataset=df.dataset.iloc[0],outcome=df.outcome.iloc[0],cutoff=int(df.cutoff.iloc[0]),mode=mode,regime=regime,fold=fold)
                        if len(train)<cfg['min_train'] or len(test)<cfg['min_test'] or train.y.nunique()<2:
                            skips.append({**tag,'train_n':len(train),'test_n':len(test),'reason':'Insufficient training/test rows or training classes'});continue
                        if cfg.get('max_outer') and accepted>=cfg['max_outer']:break
                        accepted+=1
                        assert not set(train.student_id)&set(test.student_id)
                        assert not set(train.record_id)&set(test.record_id)
                        local={**cfg,'_regime':regime}
                        x,u,meta=inner_evidence(train,cols,local,cfg['budgets'])
                        scores,fitted,times=scores_for(train,test,cols,cfg)
                        scores['random']=np.array([stable_hash(i,cfg['seed']) for i in test.record_id])
                        scores['fixed_blend']=0.5*(rankdata(scores['logistic'])+rankdata(scores['rule_inactivity']))/len(test)
                        timings.append({**tag,**times})
                        fold_audit.append({**tag,'train_n':len(train),'test_n':len(test),'student_overlap':0,
                                           'inner_validation_cohorts':len({m['cohort'] for m in meta}),
                                           'train_last_start':str(train.start.max()),'test_first_start':str(test.start.min())})
                        if cfg.get('save_models'):
                            import joblib
                            modeldir=output/'models';modeldir.mkdir(exist_ok=True)
                            joblib.dump(fitted,modeldir/f"{len(fold_audit):04d}.joblib")
                        for b in cfg['budgets']:
                            choices,status,est=choose_strategies(x,u,descriptor(train,test,b,cols),meta,cfg['gate_margin'])
                            decisions.append({**tag,'budget':b,'choices':choices,'status':status,'estimated_source_utilities':est})
                            ss={**scores,**{name:scores[choice] for name,choice in choices.items()}}
                            for cohort,part in test.groupby('cohort',sort=True):
                                ix=test.index.get_indexer(part.index)
                                for method,all_scores in ss.items():
                                    pred=all_scores[ix]
                                    m=metrics(part.y,pred,part.record_id,b,method in MODELS)
                                    row={**tag,'test_cohort':cohort,'course':part.course.iloc[0],'budget':b,'method':method,**m}
                                    rows.append(row)
                                    # Store every prediction for paired and source-level inspection, append by fold to keep memory bounded.
                                    pr=part[['record_id','student_id','cohort','course','y']].copy()
                                    pr['score']=pred;pr['selected']=select_top(pred,part.record_id,b).astype(int)
                                    for k,v in {**tag,'budget':b,'method':method}.items():pr[k]=v
                                    name=output/'predictions'/f"{len(prediction_files):06d}.csv.gz";name.parent.mkdir(exist_ok=True)
                                    pr.to_csv(name,index=False);prediction_files.append(str(name.relative_to(output)))
                        print(f"Completed {tag} train={len(train)} test={len(test)}",flush=True)
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(timings).to_csv(output/'fit_times.csv',index=False)
    dump_json(output/'fold_audit.json',fold_audit);dump_json(output/'selector_decisions.json',decisions)
    dump_json(output/'skipped_folds.json',skips);dump_json(output/'prediction_files.json',prediction_files)
    if not rows:raise ValueError("No evaluable folds. See skipped_folds.json; do not treat this as a successful experiment.")
    summarize(output,cfg)
    dump_json(output/'run_complete.json',{'complete':True,'folds':len(fold_audit),'metric_rows':len(rows),
               'warning':'Completion means software finished; does not establish novelty or effectiveness.'})
    return output

def summarize(output,cfg):
    output=Path(output);df=pd.read_csv(output/'metrics.csv')
    keys=['dataset','outcome','regime','cutoff','mode','budget']
    summary=df.groupby(keys+['method'],dropna=False)[['recall','precision','roc_auc','pr_auc','brier']].mean().reset_index()
    summary.to_csv(output/'summary.csv',index=False)
    # Paired course-cluster bootstrap: repeated cohorts within a course stay together.
    rng=np.random.default_rng(cfg['seed']); comparisons=[]
    for group,q in df.groupby(keys,sort=True):
        index=['fold','test_cohort','course']
        pivot=q.pivot(index=index,columns='method',values='recall')
        for method in ['selector','selector_no_margin','selector_no_shift','global_validation','fixed_blend']:
            for baseline in ['global_validation','logistic','hist_gradient','rule_inactivity','rule_low_activity','rule_low_recent_share']:
                if method==baseline:continue
                v=pivot[[method,baseline]].dropna();delta=v[method]-v[baseline]
                by_course=delta.groupby(level='course').mean();n=len(by_course)
                boots=np.array([rng.choice(by_course.to_numpy(),size=n,replace=True).mean() for _ in range(cfg['bootstrap'])]) if n>=3 else np.array([])
                lo,hi=np.quantile(boots,[.025,.975]) if len(boots) else (np.nan,np.nan)
                comparisons.append({**dict(zip(keys,group)),'method':method,'baseline':baseline,
                   'macro_course_recall_difference':float(by_course.mean()),'courses':n,'ci_low':lo,'ci_high':hi,
                   'note':'Exploratory marginal 95% interval; no multiplicity correction. Few clusters make uncertainty fragile; CI omitted below 3 courses.'})
    pd.DataFrame(comparisons).to_csv(output/'paired_comparisons.csv',index=False)
    oracle=df[df.method.isin(CANDIDATES)].groupby(keys+['fold','test_cohort'])['recall'].max().rename('test_best_recall_upper_bound').reset_index()
    oracle['note']='Uses test labels: retrospective diagnostic only, never a deployable policy or a training input.'
    oracle.to_csv(output/'oracle_diagnostic_only.csv',index=False)
    # One primary predeclared comparison; no selection of best test baseline.
    report=['# Run report','', 'Scope: retrospective risk identification; no causal intervention benefit is estimated.',
            'The proposed selector is a regularized source-validation heuristic, not a proven optimal policy.',
            'Primary planned contrast: selector vs global_validation, day 28, budget 0.20, behavior features, course transfer.',
            'All other contrasts are exploratory; do not choose a primary contrast after seeing the results.','',
            'The two datasets have different outcomes. Replicate the method separately; do not pool their performance.',
            'See selector_decisions.json for fallback use; temporal folds may have too few prior cohorts to train a selector.',
            '', '## Outputs','summary.csv: cohort-macro performance; paired_comparisons.csv: course-macro differences.',
            'predictions/: full paired predictions; fold_audit.json: split accounting; skipped_folds.json: omissions.']
    (output/'report.md').write_text('\n'.join(report),encoding='utf-8')
    plot_results(output,summary)
    disagreement_analysis(output)

def plot_results(output,summary):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    subset=summary[(summary.cutoff==28)&(summary['mode']=='behavior')]
    for (dataset,outcome,regime),q in subset.groupby(['dataset','outcome','regime']):
        fig,ax=plt.subplots(figsize=(7,4))
        for method,g in q.groupby('method'):
            g=g.sort_values('budget');ax.plot(g.budget,g.recall,marker='o',label=method)
        ax.set(xlabel='Attention capacity fraction',ylabel='Recall of observed risk outcomes',ylim=(0,1))
        ax.legend(fontsize=7,ncol=2);fig.tight_layout()
        fig.savefig(output/f'capacity_{dataset}_{outcome}_{regime}.pdf');plt.close(fig)

def disagreement_analysis(output):
    paths=json.loads((output/'prediction_files.json').read_text(encoding='utf-8'))
    # Compare a fixed model and fixed rule; no test-best selection.
    candidates=[]
    for p in paths:
        q=pd.read_csv(output/p,dtype={'record_id':str,'student_id':str})
        if q.method.iloc[0] in ['logistic','rule_inactivity']:candidates.append(q)
    if not candidates:return
    df=pd.concat(candidates,ignore_index=True)
    keys=['dataset','outcome','regime','cutoff','mode','fold','cohort','budget','record_id']
    p=df.pivot(index=keys,columns='method',values='selected').reset_index()
    y=df.drop_duplicates(keys)[keys+['y']];p=p.merge(y,on=keys,validate='one_to_one')
    p['category']=np.select([(p.logistic==1)&(p.rule_inactivity==1),(p.logistic==1)&(p.rule_inactivity==0),
                             (p.logistic==0)&(p.rule_inactivity==1)],['both','model_only','rule_only'],default='neither')
    p.to_csv(output/'disagreement_records.csv.gz',index=False)
    p.groupby(['dataset','outcome','regime','cutoff','mode','budget','category']).agg(n=('y','size'),risk_outcomes=('y','sum'),risk_fraction=('y','mean')).reset_index().to_csv(output/'disagreement_summary.csv',index=False)

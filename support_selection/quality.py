"""Source-only hyperparameter selection with regenerated nested transfer evidence.

V2 is exploratory on the already-inspected OULAD data. Oracle diagnostics never
participate in training. Original v1 code/results are preserved.
"""
from pathlib import Path
import hashlib
import json
import time
import sys
import platform
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from .data import ROOT, FEATURES, sha256, dump_json
from .experiment import metrics, select_top, stable_hash, rule_scores, descriptor, purge_students, validate_frame

VERSION='quality-v2.0'

def transform_features(df, mode):
    df=df.copy()
    if mode=='activity_only':return df,['log_clicks','log_events','log_active_days']
    if mode=='behavior':return df,list(FEATURES)
    if mode=='rate_change':
        c=df.cutoff.to_numpy(float)
        recent=np.expm1(df.log_recent_clicks.to_numpy(float))
        earlier=np.maximum(0,np.expm1(df.log_clicks.to_numpy(float))-recent)
        df['log_recent_daily_rate']=np.log1p(recent/np.minimum(7,c+1))
        df['log_earlier_daily_rate']=np.log1p(earlier/np.maximum(c-6,1))
        df['log_rate_change']=df.log_recent_daily_rate-df.log_earlier_daily_rate
        return df,list(FEATURES)+['log_recent_daily_rate','log_earlier_daily_rate','log_rate_change']
    raise ValueError(mode)

def candidate_specs(cfg):
    specs={k:{'kind':'rule'} for k in rule_scores(pd.DataFrame({c:[0.] for c in FEATURES}))}
    for c in cfg['lr_C']:specs[f'lr_C{c:g}']={'kind':'lr','C':c}
    for leaves in cfg['hgb_leaves']:
        specs[f'hgb_L{leaves}']={'kind':'hgb','leaves':leaves,'iterations':cfg['hgb_iterations']}
    for w in cfg['rule_blend_weights']:specs[f'rule_blend_{w:g}']={'kind':'blend','weight':w}
    return specs

def score_pool(train, target, cols, cfg, cache):
    """Cache key includes train Y and train/target X, but excludes target Y."""
    assert not set(train.student_id)&set(target.student_id),'Student leakage'
    specs=candidate_specs(cfg);h=hashlib.sha256()
    import sklearn
    h.update(json.dumps([VERSION,sklearn.__version__,specs,cols,cfg['seed']],sort_keys=True).encode())
    for f in [train[['record_id','y']+cols],target[['record_id']+cols]]:
        h.update(pd.util.hash_pandas_object(f,index=False).to_numpy().tobytes())
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);key=h.hexdigest();path=cache/f'{key}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files},True
    scores=rule_scores(target)
    for name,spec in specs.items():
        kind=spec['kind']
        if kind=='rule':continue
        if kind=='blend':
            n=len(target);w=spec['weight']
            scores[name]=(w*rankdata(scores['rule_inactivity'])+(1-w)*rankdata(scores['rule_low_activity']))/n
            continue
        if train.y.nunique()<2:scores[name]=np.repeat(float(train.y.mean()),len(target));continue
        if kind=='lr':
            model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),
                  LogisticRegression(C=spec['C'],max_iter=700,random_state=cfg['seed']))
        else:
            model=HistGradientBoostingClassifier(max_iter=spec['iterations'],max_leaf_nodes=spec['leaves'],
                       learning_rate=.1,l2_regularization=1,early_stopping=False,random_state=cfg['seed'])
        model.fit(train[cols],train.y);scores[name]=model.predict_proba(target[cols])[:,1]
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**scores);tmp.replace(path)
    return scores,False

def source_folds(source,cfg,middle=False):
    groups=sorted(source.course.unique())
    limit=cfg.get('max_middle_courses') if middle else cfg.get('max_inner_courses')
    # Subsample source validation courses without inspecting their outcomes.
    if limit and len(groups)>limit:
        groups=sorted(groups,key=lambda g:stable_hash(g,cfg['seed']))[:limit]
    for course in groups:
        val=source.loc[source.course==course].copy()
        fit=purge_students(source.loc[source.course!=course],val)
        if len(fit)>=cfg['min_train'] and len(val)>=cfg['min_test']:
            yield course,fit,val

def build_evidence(source,cols,cfg,cache):
    """Regenerated inside every middle split; no held-out course enters base fitting."""
    xs=[];us=[];tags=[]
    specs=candidate_specs(cfg);names=list(specs)
    for course,fit,val in source_folds(source,cfg):
        scores,_=score_pool(fit,val,cols,cfg,cache)
        for b in cfg['budgets']:
            cohort_util=[]
            for _,q in val.groupby('cohort'):
                if q.y.sum()==0:continue
                ix=val.index.get_indexer(q.index)
                cohort_util.append([metrics(q.y,scores[n][ix],q.record_id,b)['recall'] for n in names])
            if not cohort_util:continue
            # One meta example per independent source COURSE and budget.
            xs.append(descriptor(fit,val,b,cols));us.append(np.mean(cohort_util,axis=0));tags.append({'course':course,'budget':b})
    return np.asarray(xs),np.asarray(us),tags,names

def select_names(evidence,query,budget,cfg,alpha=10,family='ridge',no_shift=False,margin=None):
    x,u,tags,names=evidence
    if not len(x):
        return {'chosen':'rule_inactivity','global':'rule_inactivity','rule':'rule_inactivity',
                'lr':next(n for n in names if n.startswith('lr_')),
                'hgb':next(n for n in names if n.startswith('hgb_')),'fallback':True,'estimates':{}}
    ix=[i for i,t in enumerate(tags) if t['budget']==budget]
    if not ix:raise ValueError('Requested budget has no source evidence')
    means=u[ix].mean(axis=0)
    def best(allowed):return max(allowed,key=lambda n:means[names.index(n)])
    global_name=best(names);rule=best([n for n in names if n.startswith('rule_')])
    result={'global':global_name,'rule':rule,'lr':best([n for n in names if n.startswith('lr_')]),
            'hgb':best([n for n in names if n.startswith('hgb_')])}
    if len({t['course'] for t in tags})<cfg['min_meta_courses']:
        return {**result,'chosen':global_name,'fallback':True,'estimates':{}}
    if no_shift:x=x[:,:4];query=query[:4]
    estimator=(Ridge(alpha=alpha) if family=='ridge' else DecisionTreeRegressor(max_depth=2,min_samples_leaf=2,random_state=cfg['seed']))
    import copy
    estimates={}
    for j,n in enumerate(names):
        model=make_pipeline(StandardScaler(),copy.deepcopy(estimator))
        # Budget rows within a source course are correlated. Each course has equal total fit weight.
        counts=pd.Series([t['course'] for t in tags]).value_counts()
        weights=np.array([1/counts[t['course']] for t in tags])
        model.fit(x,u[:,j],**{model.steps[-1][0]+'__sample_weight':weights})
        estimates[n]=float(model.predict(query[None,:])[0])
    winner=max(names,key=lambda n:estimates[n]);m=cfg['gate_margin'] if margin is None else margin
    chosen=winner if estimates[winner]-estimates[rule]>=m else rule
    return {**result,'chosen':chosen,'fallback':False,'estimates':estimates}

def tune_selector(source,cols,cfg,cache):
    """Middle leave-course-out; re-create all inner evidence without middle Y."""
    values={a:[] for a in cfg['selector_alphas']};audit=[]
    for course,fit,val in source_folds(source,cfg,middle=True):
        e=build_evidence(fit,cols,cfg,cache)
        scores,_=score_pool(fit,val,cols,cfg,cache)
        audit.append({'validation_course':course,'fit_courses':sorted(fit.course.unique()),
                      'validation_students_in_fit':len(set(fit.student_id)&set(val.student_id)),
                      'evidence_courses':sorted({t['course'] for t in e[2]})})
        for alpha in values:
            recall=[]
            for b in cfg['budgets']:
                d=select_names(e,descriptor(fit,val,b,cols),b,cfg,alpha=alpha)
                for _,q in val.groupby('cohort'):
                    if q.y.sum()==0:continue
                    ix=val.index.get_indexer(q.index)
                    recall.append(metrics(q.y,scores[d['chosen']][ix],q.record_id,b)['recall'])
            if recall:values[alpha].append(float(np.mean(recall)))
    eligible={a:np.mean(v) for a,v in values.items() if v}
    # Identical validation scores favor stronger regularization, predeclared.
    chosen=max(eligible,key=lambda a:(eligible[a],a)) if eligible else max(cfg['selector_alphas'])
    return chosen,{'alpha':chosen,'middle_course_scores':values,'middle_audit':audit,
                   'criterion':'Mean recall across middle validation courses; ties favor larger alpha. All evidence regenerated without the middle course.'}

def plan_runs(df,cfg):
    for course in sorted(df.course.unique()):
        test=df.loc[df.course==course].copy()
        source=purge_students(df.loc[df.course!=course],test)
        courses=sorted(source.course.unique())
        for requested in cfg['source_course_counts']:
            if requested is not None and requested>len(courses):continue
            count=len(courses) if requested is None else requested
            if requested is not None and count==len(courses) and None in cfg['source_course_counts']:continue
            draws=cfg.get('learning_draws',1) if requested is not None else 1
            for draw in range(draws):
                rng=np.random.default_rng(cfg['seed']+draw)
                chosen=sorted(rng.choice(courses,count,replace=False).tolist())
                train=source.loc[source.course.isin(chosen)].copy()
                yield course,count,draw,train,test

def run_quality(paths,cfg,output):
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new, empty quality output directory')
    out.mkdir(parents=True,exist_ok=True);dump_json(out/'config.json',cfg)
    dump_json(out/'environment.json',{'version':VERSION,'python':sys.version,'platform':platform.platform(),
        'versions':{n:__import__(n).__version__ for n in ['numpy','pandas','sklearn','scipy']},
        'input_hashes':{str(p):sha256(p) for p in paths},
        'interpretation':'Exploratory OULAD follow-up; independent external methodological replication only after protocol lock. No intervention effect.'})
    cache=ROOT/'.cache/quality_scores';metrics_rows=[];decisions=[];tuning=[];audits=[];skips=[]
    start=time.perf_counter();done=0
    with threadpool_limits(limits=cfg['threads']):
        for path in paths:
            frame=pd.read_csv(path,dtype={k:str for k in ['record_id','student_id','course','cohort']},parse_dates=['start','label_available'])
            validate_frame(frame)
            for mode in cfg['feature_modes']:
                df,cols=transform_features(frame,mode)
                for course,nsource,draw,train,test in plan_runs(df,cfg):
                    if cfg.get('max_outer') and done>=cfg['max_outer']:break
                    tag={'dataset':df.dataset.iloc[0],'outcome':df.outcome.iloc[0],'cutoff':int(df.cutoff.iloc[0]),
                         'mode':mode,'test_course':course,'source_courses':nsource,'draw':draw}
                    if len(train)<cfg['min_train'] or len(test)<cfg['min_test'] or train.y.nunique()<2:
                        skips.append({**tag,'reason':'Insufficient training rows/classes'});continue
                    print(f'Quality fold START {tag}; nested tuning can take time',flush=True)
                    alpha,tune=tune_selector(train,cols,cfg,cache)
                    tuning.append({**tag,**tune})
                    e=build_evidence(train,cols,cfg,cache)
                    scores,hit=score_pool(train,test,cols,cfg,cache)
                    scores['random']=np.array([stable_hash(i,cfg['seed']) for i in test.record_id])
                    fixed_lr='lr_C1' if 'lr_C1' in e[3] else next(n for n in e[3] if n.startswith('lr_'))
                    scores['fixed_blend']=(rankdata(scores[fixed_lr])+rankdata(scores['rule_inactivity']))/(2*len(test))
                    audits.append({**tag,'train_rows':len(train),'test_rows':len(test),'train_courses':sorted(train.course.unique()),
                        'student_overlap':len(set(train.student_id)&set(test.student_id)),
                        'inner_courses':len({t['course'] for t in e[2]}),'test_score_cache_hit':hit})
                    assert not set(train.student_id)&set(test.student_id)
                    probabilities=[n for n in e[3] if n.startswith(('lr_','hgb_'))]
                    pred=test[['record_id','student_id','cohort','course','y']+cols].copy()
                    for name,value in scores.items():pred['score__'+name]=value
                    for b in cfg['budgets']:
                        query=descriptor(train,test,b,cols)
                        d=select_names(e,query,b,cfg,alpha=alpha)
                        no_margin=select_names(e,query,b,cfg,alpha=alpha,margin=0)
                        no_shift=select_names(e,query,b,cfg,alpha=alpha,no_shift=True)
                        fixed=select_names(e,query,b,cfg,alpha=cfg['fixed_alpha'])
                        tree=select_names(e,query,b,cfg,family='tree')
                        mapping={'selector':d['chosen'],'global_validation':d['global'],'tuned_rule':d['rule'],
                                 'tuned_lr':d['lr'],'tuned_hgb':d['hgb'],'selector_fixed_alpha':fixed['chosen'],
                                 'selector_tree':tree['chosen'],'selector_no_shift':no_shift['chosen'],'selector_no_margin':no_margin['chosen']}
                        decisions.append({**tag,'budget':b,'alpha':alpha,'mapping':mapping,'fallback':d['fallback'],
                                          'predicted_utilities':d['estimates']})
                        policies={**scores,**{name:scores[candidate] for name,candidate in mapping.items()}}
                        for cohort,q in test.groupby('cohort'):
                            ix=test.index.get_indexer(q.index)
                            for name,values in policies.items():
                                m=metrics(q.y,values[ix],q.record_id,b,name in probabilities or mapping.get(name) in probabilities)
                                metrics_rows.append({**tag,'budget':b,'cohort':cohort,'method':name,**m})
                        for name in mapping:
                            pred[f'selected__{b:g}__{name}']=0
                            # Enforce the same capacity independently within each cohort.
                            for _,q in test.groupby('cohort'):
                                ix=test.index.get_indexer(q.index)
                                pred.loc[q.index,f'selected__{b:g}__{name}']=select_top(policies[name][ix],q.record_id,b).astype(int)
                    pred_dir=out/'predictions';pred_dir.mkdir(exist_ok=True)
                    pred.to_csv(pred_dir/f'{done:05d}.csv.gz',index=False)
                    done+=1
                    # Flush progress so a long run can be diagnosed without reading a live process.
                    dump_json(out/'progress.json',{'completed_outer_folds':done,'latest':tag,'elapsed_seconds':time.perf_counter()-start})
                    print(f'Quality fold COMPLETED {tag}; chosen alpha={alpha}',flush=True)
    if not metrics_rows:raise ValueError('No evaluable quality folds; check eligibility/configuration')
    pd.DataFrame(metrics_rows).to_csv(out/'metrics.csv',index=False)
    dump_json(out/'decisions.json',decisions);dump_json(out/'nested_tuning.json',tuning)
    dump_json(out/'fold_audit.json',audits);dump_json(out/'skipped_folds.json',skips)
    analyze_quality(out,cfg)
    dump_json(out/'run_complete.json',{'complete':True,'outer_folds':done,'elapsed_seconds':time.perf_counter()-start,
                                      'scope':'Software completion, not proof of effectiveness.'})
    return out

def analyze_quality(out,cfg):
    out=Path(out);df=pd.read_csv(out/'metrics.csv')
    keys=['dataset','outcome','cutoff','mode','source_courses','draw','budget']
    # Primary estimand: first average cohorts within held-out course, then macro courses.
    course=df.groupby(keys+['test_course','method'])[['recall','precision','roc_auc','pr_auc','brier']].mean().reset_index()
    course.to_csv(out/'course_metrics.csv',index=False)
    course.groupby(keys+['method'])[['recall','precision','roc_auc','pr_auc','brier']].mean().reset_index().to_csv(out/'summary.csv',index=False)
    pool=list(candidate_specs(cfg));headroom=[];paired=[];rng=np.random.default_rng(cfg['seed'])
    for group,q in course.groupby(keys):
        piv=q.pivot(index='test_course',columns='method',values='recall')
        oracle=piv[pool].max(axis=1)
        for c in piv.index:
            headroom.append({**dict(zip(keys,group)),'test_course':c,'oracle_course':oracle[c],
                'oracle_candidate':piv.loc[c,pool].idxmax(),
                'selector':piv.loc[c,'selector'],'global_validation':piv.loc[c,'global_validation'],
                'remaining_headroom':oracle[c]-piv.loc[c,'selector'],
                'matches_best_candidate_performance':bool(np.isclose(oracle[c],piv.loc[c,'selector'],rtol=0,atol=1e-12)),
                'note':'Test-label-dependent candidate-pool reference only, not a theoretical/global bound or deployable strategy.'})
        for method in ['selector','selector_fixed_alpha','selector_tree','selector_no_shift','selector_no_margin']:
            for baseline in ['global_validation','tuned_rule','tuned_lr','tuned_hgb','fixed_blend']:
                delta=(piv[method]-piv[baseline]).dropna().to_numpy();n=len(delta)
                samples=np.array([rng.choice(delta,n,replace=True).mean() for _ in range(cfg['bootstrap'])]) if n>=3 else []
                lo,hi=np.quantile(samples,[.025,.975]) if len(samples) else (np.nan,np.nan)
                paired.append({**dict(zip(keys,group)),'method':method,'baseline':baseline,'course_macro_delta':np.mean(delta),
                               'courses':n,'ci_low':lo,'ci_high':hi,
                               'note':'Exploratory marginal interval; correlated training folds and few courses limit inference; no multiplicity-adjusted significance claim.'})
    pd.DataFrame(headroom).to_csv(out/'candidate_headroom_diagnostic.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_comparisons.csv',index=False)
    # Training-course learning curves are descriptive; repeated draws are not independent samples.
    summary=course.groupby(keys+['method']).recall.mean().reset_index()
    summary.to_csv(out/'learning_curve_values.csv',index=False)
    disagreements=[]
    # File order matches fold audit order; never choose a favorable test baseline.
    audit=json.loads((out/'fold_audit.json').read_text(encoding='utf-8'))
    for i,tag in enumerate(audit):
        pred=pd.read_csv(out/'predictions'/f'{i:05d}.csv.gz')
        for b in cfg['budgets']:
            s=pred[f'selected__{b:g}__selector']==1;g=pred[f'selected__{b:g}__global_validation']==1
            category=np.select([s&g,s&~g,~s&g],['both','selector_only','global_only'],default='neither')
            pred['category']=category
            for name,part in pred.groupby('category'):
                disagreements.append({**{k:tag[k] for k in ['dataset','outcome','cutoff','mode','test_course','source_courses','draw']},
                                      'budget':b,'category':name,'n':len(part),'observed_risk':int(part.y.sum()),
                                      'risk_fraction':float(part.y.mean()),
                                      **{c:float(part[c].mean()) for c in FEATURES if c in part}})
    pd.DataFrame(disagreements).to_csv(out/'selection_disagreement.csv',index=False)
    (out/'report.md').write_text('# Quality experiment\n\nSee summary.csv (course macro), nested_tuning.json and decisions.json.\n'
        'candidate_headroom_diagnostic.csv uses test labels only for post-hoc candidate-space diagnosis.\n'
        'Training evidence is regenerated inside each middle course split. Test labels never tune alpha.\n'
        'KDD and OULAD have different outcomes and are reported separately.\n'
        'OULAD follow-up is exploratory after inspecting v1 results. A Q1 claim is not an output of this software.\n',encoding='utf-8')
    plot_quality(out,summary)

def plot_quality(out,summary):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for group,q in summary.groupby(['dataset','outcome','cutoff','mode','budget']):
        fig,ax=plt.subplots(figsize=(6,4))
        for method in ['selector','global_validation','tuned_rule','tuned_lr','tuned_hgb']:
            g=q[q.method==method].groupby('source_courses').recall.mean()
            ax.plot(g.index,g.values,marker='o',label=method)
        ax.set(xlabel='Number of source courses',ylabel='Course-macro risk recall',ylim=(0,1))
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/('learning_'+'_'.join(map(str,group))+'.pdf'));plt.close(fig)

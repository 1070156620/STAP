"""Post-hoc course comparisons and same-population list stability; no model training."""
import argparse,itertools,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from support_selection.data import ROOT
from support_selection.experiment import select_top

OUT=ROOT/'outputs/stap_course_stability_v1'
BASE=ROOT/'outputs/metric_refinement_v1/runs'
METHODS={'tuned_ours':'STAP','tuned_lr':'LR','tuned_hgb':'HGB'}
SEEDS=list(range(20261028,20261033));BUDGETS=[.1,.2,.3]
def load():
    blocks=[]
    for ds,day in [('oulad',28),('kdd2015',14)]:
        run=BASE/f'{ds}_primary_{day}_behavior'
        frame=pd.read_csv(run/'metric_analysis/course_metrics.csv')
        frame=frame[frame.method.isin(METHODS)].copy();frame['method']=frame.method.map(METHODS)
        blocks.append(frame)
    blocks.append(pd.read_csv(ROOT/'outputs/published_baselines_v1/analysis/course_metrics.csv'))
    d=pd.concat(blocks,ignore_index=True)
    assert set(d.method)=={'STAP','LR','HGB','RF','Linear SVM'}
    assert not d.duplicated(['dataset','seed','test_course','budget','method']).any()
    return d

def course_analysis():
    d=load();d.to_csv(OUT/'all_methods_course_metrics.csv',index=False)
    choices=pd.read_csv(ROOT/'outputs/stap_mechanism_v1/course_choices.csv')
    overlap=pd.read_csv(ROOT/'outputs/stap_mechanism_v1/course_list_overlap.csv')
    rows=[]
    for (ds,seed,course,b),g in d.groupby(['dataset','seed','test_course','budget']):
        assert len(g)==5
        ours=g[g.method=='STAP'].iloc[0]
        # This target-label maximum is an upper-reference diagnostic, never a deployable selector.
        ranked=g[g.method!='STAP'].sort_values(['recall','method'],ascending=[False,True])
        best=ranked.iloc[0]
        ch=choices[(choices.dataset==ds)&(choices.seed==seed)&(choices.test_course.astype(str)==str(course))&np.isclose(choices.budget,b)]
        ov=overlap[(overlap.dataset==ds)&(overlap.seed==seed)&(overlap.test_course.astype(str)==str(course))&np.isclose(overlap.budget,b)&(overlap.comparator==best.method)]
        if len(ch)!=1 or len(ov)!=1:raise ValueError('Course choice/list pairing missing')
        rows.append(dict(dataset=ds,seed=int(seed),test_course=course,budget=b,candidate=ch.iloc[0].candidate,selection_evidence=ch.iloc[0].features,
            stap_recall=ours.recall,stap_precision=ours.precision,stap_tp=int(ours.tp),stap_ap=ours.ap,
            posthoc_best_external=best.method,best_external_recall=best.recall,best_external_precision=best.precision,
            delta_recall_pp=100*(ours.recall-best.recall),jaccard_vs_best=ov.iloc[0].jaccard))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'course_seed_comparison.csv',index=False)
    frame[frame.seed==20261032].to_csv(OUT/'course_comparison_20261032.csv',index=False)
    avg=d.groupby(['dataset','test_course','budget','method'],as_index=False).agg(recall=('recall','mean'),precision=('precision','mean'))
    courses=[]
    for (ds,course,b),g in avg.groupby(['dataset','test_course','budget']):
        ours=g[g.method=='STAP'].iloc[0];best=g[g.method!='STAP'].sort_values(['recall','method'],ascending=[False,True]).iloc[0]
        courses.append(dict(dataset=ds,test_course=course,budget=b,stap_recall=ours.recall,stap_precision=ours.precision,
            posthoc_best_external=best.method,best_external_recall=best.recall,delta_recall_pp=100*(ours.recall-best.recall)))
    cm=pd.DataFrame(courses);cm.to_csv(OUT/'course_five_seed_comparison.csv',index=False)
    counts=[]
    for (ds,b),g in cm.groupby(['dataset','budget']):
        v=g.delta_recall_pp.to_numpy()
        counts.append(dict(dataset=ds,budget=b,courses=len(g),higher=int((v>1e-10).sum()),tied=int((abs(v)<=1e-10).sum()),lower=int((v < -1e-10).sum()),stap_recall_mean=g.stap_recall.mean(),posthoc_best_recall_mean=g.best_external_recall.mean()))
    pd.DataFrame(counts).to_csv(OUT/'course_win_tie_loss.csv',index=False)
    frame.groupby(['dataset','test_course','budget']).agg(recall_min=('stap_recall','min'),recall_max=('stap_recall','max'),recall_mean=('stap_recall','mean'),precision_min=('stap_precision','min'),precision_max=('stap_precision','max'),distinct_candidates=('candidate','nunique')).reset_index().to_csv(OUT/'course_coverage_range.csv',index=False)
    print('COURSE ANALYSIS COMPLETE: 230 tasks, 46 distinct courses',flush=True)

def read_prediction(run,uid):
    path=run/'predictions'/(uid+'.csv.gz')
    cols=['record_id','cohort','y']+[f'selected__{b:g}__tuned_ours' for b in BUDGETS]
    d=pd.read_csv(path,usecols=cols,dtype={'record_id':str,'cohort':str}).set_index('record_id').sort_index()
    if not d.index.is_unique:raise ValueError('Duplicate enrollment identifiers')
    return d

def stability(pilot=False):
    choices=pd.read_csv(ROOT/'outputs/stap_mechanism_v1/course_choices.csv')
    group_rows=[];capacity_rows=[];cohort_rows=[]
    for (ds,course),meta in choices.groupby(['dataset','test_course'],sort=True):
        if pilot and ds!='oulad':continue
        folder=OUT/'stability_tasks';folder.mkdir(exist_ok=True)
        key=hashlib.sha256(f'{ds}|{course}'.encode()).hexdigest()[:16]
        checkpoint=folder/(key+('.pilot.json' if pilot else '.json'))
        if checkpoint.exists():
            saved=json.loads(checkpoint.read_text());group_rows+=saved['course_pairs'];capacity_rows+=saved['capacity'];cohort_rows+=saved['cohort_pairs']
            print('STABILITY REUSE '+ds+' '+str(course),flush=True)
            if pilot:break
            continue
        run=BASE/f'{ds}_primary_{28 if ds=="oulad" else 14}_behavior'
        byseed={}
        for seed,g in meta.groupby('seed'):
            if g.task.nunique()!=1:raise ValueError('Multiple tasks per course/seed')
            byseed[int(seed)]=read_prediction(run,g.task.iloc[0])
        if sorted(byseed)!=SEEDS:raise ValueError('Missing seed')
        reference=byseed[SEEDS[0]]
        for seed,d in byseed.items():
            if not d.index.equals(reference.index) or not d[['cohort','y']].equals(reference[['cohort','y']]):raise ValueError('Different target populations, cohorts or outcomes across seeds')
        local=[];cp=[];co=[]
        for a,b in itertools.combinations(SEEDS,2):
            left,right=byseed[a],byseed[b]
            for budget in BUDGETS:
                values=[]
                col=f'selected__{budget:g}__tuned_ours'
                for cohort,g in reference.groupby('cohort'):
                    x=left.loc[g.index,col].to_numpy(bool);y=right.loc[g.index,col].to_numpy(bool)
                    if x.sum()!=y.sum():raise ValueError('Capacity mismatch across seeds')
                    j=float((x&y).sum()/(x|y).sum());values.append(j)
                    co.append(dict(dataset=ds,test_course=str(course),cohort=str(cohort),seed_a=a,seed_b=b,budget=budget,jaccard=j))
                local.append(dict(dataset=ds,test_course=str(course),seed_a=a,seed_b=b,budget=budget,jaccard=float(np.mean(values))))
        for seed,d in byseed.items():
            for small,large in itertools.combinations(BUDGETS,2):
                js=[];retention=[]
                for _,g in d.groupby('cohort'):
                    x=g[f'selected__{small:g}__tuned_ours'].to_numpy(bool);y=g[f'selected__{large:g}__tuned_ours'].to_numpy(bool)
                    js.append((x&y).sum()/(x|y).sum());retention.append((x&y).sum()/x.sum())
                cp.append(dict(dataset=ds,test_course=str(course),seed=seed,small_capacity=small,large_capacity=large,jaccard=float(np.mean(js)),smaller_list_retained=float(np.mean(retention))))
        checkpoint.write_text(json.dumps(dict(course_pairs=local,capacity=cp,cohort_pairs=co)),encoding='utf-8')
        group_rows+=local;capacity_rows+=cp;cohort_rows+=co
        print('STABILITY DONE '+ds+' '+str(course),flush=True)
        if pilot:break
    prefix='pilot_' if pilot else ''
    pairs=pd.DataFrame(group_rows);pairs.to_csv(OUT/(prefix+'cross_seed_course_pairs.csv'),index=False)
    pd.DataFrame(cohort_rows).to_csv(OUT/(prefix+'cross_seed_cohort_pairs.csv'),index=False)
    pd.DataFrame(capacity_rows).to_csv(OUT/(prefix+'cross_capacity_containment.csv'),index=False)
    summary=pairs.groupby(['dataset','test_course','budget'],as_index=False).agg(jaccard_mean=('jaccard','mean'),jaccard_min=('jaccard','min'),jaccard_max=('jaccard','max'),seed_pairs=('jaccard','size'))
    summary.to_csv(OUT/(prefix+'course_list_stability.csv'),index=False)
    if not pilot:
        assert len(pairs)==46*10*3 and len(summary)==46*3
        summary.groupby(['dataset','budget']).agg(courses=('test_course','nunique'),jaccard_course_mean=('jaccard_mean','mean'),jaccard_course_min=('jaccard_mean','min'),jaccard_course_max=('jaccard_mean','max')).reset_index().to_csv(OUT/'list_stability_summary.csv',index=False)

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    folder=ROOT.parent/'第二篇_论文材料/课程级与名单稳定性分析';folder.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(OUT/'course_five_seed_comparison.csv');d=d[np.isclose(d.budget,.2)]
    fig,ax=plt.subplots(figsize=(6.4,4.7),layout='constrained')
    for ds,label,color in [('oulad','OULAD','#1769A0'),('kdd2015','KDD2015','#DB8527')]:
        q=d[d.dataset==ds];ax.scatter(q.best_external_recall*100,q.stap_recall*100,label=label,c=color,s=30,alpha=.8)
    lim=[min(d.best_external_recall.min(),d.stap_recall.min())*100-1,max(d.best_external_recall.max(),d.stap_recall.max())*100+1]
    ax.plot(lim,lim,color='gray',linestyle='--',linewidth=1,label='Equal recall');ax.set(xlim=lim,ylim=lim,xlabel='Post-hoc strongest external recall (%)',ylabel='STAP recall (%)');ax.legend(loc='lower right',bbox_to_anchor=(1,1.02),ncol=3,frameon=True)
    fig.savefig(folder/'course_risk_coverage.pdf');plt.close(fig)
    from matplotlib.lines import Line2D
    d=pd.read_csv(OUT/'course_list_stability.csv');fig,axes=plt.subplots(1,2,figsize=(8,4),sharey=True)
    for ax,ds,title in zip(axes,['oulad','kdd2015'],['OULAD','KDD2015']):
        q=d[d.dataset==ds];ax.boxplot([q[np.isclose(q.budget,b)].jaccard_mean for b in BUDGETS],tick_labels=['10%','20%','30%'],showmeans=True,medianprops={'color':'#1769A0'},meanprops={'marker':'^','markerfacecolor':'#DB8527','markeredgecolor':'#DB8527'},flierprops={'marker':'o','markerfacecolor':'none','markeredgecolor':'gray'})
        ax.set(title=title,xlabel='Support capacity',ylim=(0,1.02))
    axes[0].set_ylabel('Mean cross-seed recipient Jaccard')
    fig.legend(handles=[Line2D([],[],color='#1769A0',label='Median'),Line2D([],[],marker='^',color='#DB8527',linestyle='none',label='Mean'),Line2D([],[],marker='o',markerfacecolor='none',color='gray',linestyle='none',label='Outlying course')],loc='upper right',ncol=3,frameon=True)
    fig.subplots_adjust(left=.10,right=.98,bottom=.15,top=.79,wspace=.15)
    fig.savefig(folder/'cross_seed_recipient_stability.pdf');plt.close(fig)
    print('PDF FIGURES: '+str(folder),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['plan','course','stability','figures','pilot','all'],default='plan');a=p.parse_args();OUT.mkdir(exist_ok=True,parents=True)
    if a.stage=='plan':print(json.dumps(dict(base_model_training=0,new_parameter_search=False,outer_tasks=230,distinct_courses=46,course_seed_capacity_comparisons=690,cross_seed_course_pairs=1380,capacity_course_pairs=690,posthoc_best_is_deployable=False),indent=2))
    if a.stage in ['course','all']:course_analysis()
    if a.stage in ['stability','all','pilot']:stability(a.stage=='pilot')
    if a.stage in ['figures','all']:figures()
    if a.stage=='all':(OUT/'complete.json').write_text(json.dumps(dict(complete=True,new_training=False,all_courses_retained=True,main_tables_modified=False,posthoc_reference='Best external recall chosen using target outcomes for diagnosis only; never used for deployment or STAP tuning.')),encoding='utf-8')

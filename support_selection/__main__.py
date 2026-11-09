import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import ROOT, FEATURES, download_oulad, prepare_oulad, prepare_kdd

def demo_data(cutoff=28):
    """Synthetic engineering fixture; never scientific evidence."""
    rng=np.random.default_rng(20261028);rows=[]
    for j in range(5):
        for t in range(3):
            for i in range(90):
                activity=rng.gamma(2,2)
                recent=rng.uniform(0,1)
                score=-activity/3+2*(1-recent)+0.2*j+rng.normal(0,.6)
                y=int(rng.random()<1/(1+np.exp(-score)))
                rows.append(dict(dataset='synthetic_demo',outcome='synthetic',record_id=f'{j}|{t}|{i}',
                    student_id=f's{j}_{t}_{i}',course=f'C{j}',cohort=f'C{j}_{t}',start=f'{2020+t}-02-01',label_available=f'{2020+t}-12-01',cutoff=cutoff,y=y,
                    log_clicks=activity,log_events=activity*.7,log_active_days=np.log1p(activity),
                    log_recent_clicks=activity*recent,recent_fraction=recent,inactivity_fraction=1-recent))
    p=ROOT/f'data/processed/demo_{cutoff}.csv.gz';p.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(p,index=False);return p

def main():
    ap=argparse.ArgumentParser(description='CPU-only, leakage-audited support selection research')
    sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('download',help='Download and safely extract official UCI OULAD archive')
    for cmd in ['prepare','run','audit']:
        s=sub.add_parser(cmd);s.add_argument('--dataset',choices=['oulad','kdd2015'],default='oulad')
        s.add_argument('--config',default='configs/pilot.json')
        s.add_argument('--outcome',choices=['withdrawal','unsuccessful'],default='withdrawal')
        if cmd=='prepare':s.add_argument('--raw',type=Path)
        if cmd=='run':s.add_argument('--name',required=True)
    s=sub.add_parser('demo');s.add_argument('--name',default='demo');
    s=sub.add_parser('quality-check');s.add_argument('--dataset',choices=['oulad','kdd2015'],required=True)
    s=sub.add_parser('quality');s.add_argument('--dataset',choices=['oulad','kdd2015'],required=True)
    s.add_argument('--stage',choices=['primary','learning','sensitivity'],default='primary');s.add_argument('--name',required=True)
    s=sub.add_parser('quality-demo');s.add_argument('--name',default='quality_demo')
    args=ap.parse_args()
    if args.command=='download':print(download_oulad());return
    if args.command=='quality-check':
        from .quality_cli import preflight
        result=preflight(args.dataset);print(json.dumps(result,ensure_ascii=False,indent=2))
        if not result['ready']:raise SystemExit(2)
        return
    if args.command=='quality':
        from .quality_cli import execute
        print(execute(args.dataset,args.stage,args.name));return
    if args.command=='quality-demo':
        from .quality_cli import load_config
        from .quality import run_quality
        cfg=load_config('oulad','primary')
        cfg.update({'min_train':30,'min_test':10,'max_outer':1,'max_middle_courses':2,'max_inner_courses':2,
                    'min_meta_courses':2,'lr_C':[1],'hgb_leaves':[7],'hgb_iterations':10,
                    'rule_blend_weights':[.5],'selector_alphas':[1,10],'bootstrap':30})
        print(run_quality([demo_data()],cfg,ROOT/'outputs'/args.name));return
    from .experiment import run_experiments, validate_frame
    cfg=json.loads((ROOT/(args.config if hasattr(args,'config') else 'configs/pilot.json')).read_text(encoding='utf-8'))
    if args.command=='demo':
        cfg.update({'cutoffs':[28],'bootstrap':50,'min_train':40,'min_test':20,'max_outer':2,'max_inner':4,'hgb_iterations':20})
        run_experiments([demo_data()],cfg,ROOT/'outputs'/args.name);return
    endpoint=args.outcome if args.dataset=='oulad' else 'competition_dropout'
    if args.command=='prepare':
        raw=args.raw or ROOT/'data/raw'/args.dataset
        if not raw.is_absolute():raw=ROOT/raw
        result=(prepare_oulad(raw,cfg['cutoffs'],args.outcome) if args.dataset=='oulad' else prepare_kdd(raw,cfg['cutoffs']))
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    paths=[ROOT/f'data/processed/{args.dataset}_{endpoint}_{c}.csv.gz' for c in cfg['cutoffs']]
    for p in paths:
        if not p.exists():raise FileNotFoundError(f'{p}; execute prepare with the same configuration first')
    if args.command=='audit':
        for p in paths:
            df=pd.read_csv(p,dtype={k:str for k in ['record_id','student_id','course','cohort']},parse_dates=['start','label_available'])
            validate_frame(df);print(p.name,'n=',len(df),'positive=',int(df.y.sum()),'courses=',df.course.nunique())
    elif args.command=='run':
        name=Path(args.name)
        if name.is_absolute() or len(name.parts)!=1 or args.name in ['.','..']:raise ValueError('Use a simple new run name')
        run_experiments(paths,cfg,ROOT/'outputs'/name)

if __name__=='__main__':main()

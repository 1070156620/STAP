from pathlib import Path
import json
import pandas as pd
from .data import ROOT, locate, sha256, dump_json, prepare_oulad, prepare_kdd

def load_config(dataset,stage):
    p=ROOT/f'configs/quality_{stage}.json'
    cfg=json.loads(p.read_text(encoding='utf-8'));overrides=cfg.pop('datasets')[dataset]
    cfg.update(overrides);cfg['dataset']=dataset;cfg['stage']=stage
    return cfg

def preflight(dataset):
    raw=ROOT/'data/raw'/dataset
    required=({'studentInfo.csv':['code_module','code_presentation','id_student','final_result'],
               'studentRegistration.csv':['code_module','code_presentation','id_student','date_registration','date_unregistration'],
               'studentVle.csv':['code_module','code_presentation','id_student','date','sum_click'],
               'courses.csv':['code_module','code_presentation','module_presentation_length']}
              if dataset=='oulad' else
              {'enrollment_train.csv':['enrollment_id','username','course_id'],
               'truth_train.csv':[], 'log_train.csv':['enrollment_id','time'],
               'date.csv':['course_id']})
    files={};errors=[]
    for name,cols in required.items():
        try:
            p=locate(raw,name)
            header=pd.read_csv(p,nrows=2)
            if not set(cols).issubset(header.columns):raise ValueError(f'Missing columns: {set(cols)-set(header.columns)}')
            if name=='date.csv' and not ({'from','to'}.issubset(header) or {'start','end'}.issubset(header)):
                raise ValueError('Missing explicit course start/end dates')
            if name=='truth_train.csv':
                lab=pd.read_csv(p,header=None,dtype=str)
                if lab.shape[1]!=2:raise ValueError('Expected exactly enrollment_id and binary label')
                if lab.iloc[0,0]=='enrollment_id':lab=lab.iloc[1:]
                if not lab.iloc[:,1].isin(['0','1']).all() or lab.iloc[:,1].nunique()!=2:
                    raise ValueError('Need genuine binary train labels containing both classes, not sampleSubmission placeholders')
                if lab.iloc[:,0].duplicated().any():raise ValueError('Duplicate train-label enrollment IDs')
            files[name]={'path':str(p),'bytes':p.stat().st_size,'sha256':sha256(p)}
        except (ValueError,FileNotFoundError,pd.errors.EmptyDataError) as e:errors.append(f'{name}: {e}')
    result={'dataset':dataset,'ready':not errors,'files':files,'errors':errors,
            'warning':'Only labelled train partition is used. sampleSubmission and released/unknown test labels are not ground truth.'}
    dest=ROOT/'quality_checks'/f'{dataset}_preflight.json';dump_json(dest,result)
    return result

def ensure_prepared(dataset,cfg):
    check=preflight(dataset)
    if not check['ready']:raise ValueError('Raw data incomplete:\n'+'\n'.join(check['errors']))
    endpoint=cfg['outcome']
    paths=[ROOT/f'data/processed/{dataset}_{endpoint}_{c}.csv.gz' for c in cfg['cutoffs']]
    audit=ROOT/f'data/processed/{dataset}_{endpoint}_audit.json' if dataset=='oulad' else ROOT/'data/processed/kdd2015_audit.json'
    valid=all(p.exists() for p in paths) and audit.exists()
    if valid:
        old=json.loads(audit.read_text(encoding='utf-8'))
        valid=all(old.get('source_hashes',{}).get(n)==v['sha256'] for n,v in check['files'].items())
        records={r['cutoff']:r for r in old.get('cohorts',[])}
        valid=valid and all(c in records and records[c].get('sha256')==sha256(p) for c,p in zip(cfg['cutoffs'],paths))
    if not valid:
        if dataset=='oulad':prepare_oulad(ROOT/'data/raw/oulad',cfg['cutoffs'],endpoint)
        else:prepare_kdd(ROOT/'data/raw/kdd2015',cfg['cutoffs'])
    return paths

def execute(dataset,stage,name):
    if Path(name).is_absolute() or len(Path(name).parts)!=1 or name in ['.','..']:raise ValueError('Use a simple new run name')
    cfg=load_config(dataset,stage);paths=ensure_prepared(dataset,cfg)
    from .quality import run_quality
    return run_quality(paths,cfg,ROOT/'outputs'/name)

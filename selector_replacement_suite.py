"""Rebuild the complete experiment family with source-tuned Ours.

All source searches across the suite finish before target evaluation begins.
Old result directories and manuscript assets are never overwritten.
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
import decision_principle_experiments as io
from support_selection.data import ROOT,sha256


def build_cases():
    original=io.read_json(ROOT/'configs/decision_principle.json')
    search=io.read_json(ROOT/'configs/selector_optimization.json')
    first_seed=search['seeds'][0]
    cases=[]
    windows={'oulad':[14,28,42],'kdd2015':[7,14,21],'mooccubex':[7,14,21]}
    primary={'oulad':28,'kdd2015':14,'mooccubex':14}
    for name in original['datasets']:
        modes=['behavior'] if name=='mooccubex' else ['activity_only','behavior','rate_change']
        for cutoff in windows[name]:
            for mode in modes:
                main=cutoff==primary[name] and mode=='behavior'
                cases.append(dict(name=f'{name}_{"primary" if main else "sensitivity"}_{cutoff}_{mode}',
                    dataset=name,cutoff=cutoff,mode=mode,seeds=search['seeds'] if main else [first_seed],
                    sources=original['datasets'][name]['source_course_counts'],draws=1,
                    family='primary' if main else 'sensitivity'))
    for name,sources in [('oulad',[3,4,5]),('kdd2015',[3,5,8])]:
        cases.append(dict(name=f'{name}_source_learning',dataset=name,cutoff=primary[name],mode='behavior',
            seeds=[first_seed],sources=sources,draws=2,family='source_learning'))
    cases.append(dict(name='kdd2015_full_sources',dataset='kdd2015',cutoff=14,mode='behavior',
        seeds=[first_seed],sources=[None],draws=1,family='full_sources'))
    return original,search,cases


def prepare(out):
    original,search,cases=build_cases()
    # Keep prepared protocols outside each result directory's immutable run lock.
    config_dir=out/'configurations'
    config_dir.mkdir(parents=True,exist_ok=True)
    schedule=[]
    count_cache={}
    for case in cases:
        name=case['dataset'];cutoff=case['cutoff']
        path=(f'outputs/mooccubex_v3/mooccubex_video_inactivity_14d_{cutoff}.csv.gz' if name=='mooccubex'
              else f'data/processed/{name}_{"withdrawal" if name=="oulad" else "competition_dropout"}_{cutoff}.csv.gz')
        if path not in count_cache:
            frame=io.dataset_frame(ROOT/path)
            count_cache[path]=int(frame.course.nunique())
        n=count_cache[path]
        protocol=copy.deepcopy(original)
        protocol['datasets'][name]['processed']=path
        protocol['datasets'][name]['source_course_counts']=case['sources']
        protocol['base']['learning_draws']=case['draws']
        protocol['feature_mode']=case['mode']
        protocol_file=config_dir/(case['name']+'.protocol.json')
        config_file=config_dir/(case['name']+'.search.json')
        candidate_search=copy.deepcopy(search)
        candidate_search['seeds']=case['seeds']
        candidate_search['base_protocol']=str(protocol_file.relative_to(ROOT)).replace('\\','/')
        # Deterministic preparation: do not silently change a prepared suite.
        for target,value in [(protocol_file,protocol),(config_file,candidate_search)]:
            if target.exists() and io.read_json(target)!=value:
                raise ValueError('Prepared configuration changed; use a new suite OutputRoot')
            if not target.exists():io.write_json(target,value)
        tasks=0
        for requested in case['sources']:
            if requested is not None and requested>n-1:continue
            tasks+=n*len(case['seeds'])*(case['draws'] if requested is not None else 1)
        schedule.append(dict(**case,courses=n,tasks=tasks,config=str(config_file.relative_to(ROOT)),
            protocol=str(protocol_file.relative_to(ROOT)),result=str((out/'runs'/case['name']).relative_to(ROOT))))
    manifest=dict(cases=schedule,source_only_configurations_per_task=142,
        source_tuning_tasks=sum(c['tasks'] for c in schedule),
        target_evaluation_tasks=sum(c['tasks'] for c in schedule),
        policy='Use source-selected Tuned Ours as final Ours in every experiment; Original Ours stays a control.',
        scope='Exploratory replacement suite on previously inspected datasets, not independent external validation.',
        hashes={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'selector_optimization.py',
            ROOT/'configs/selector_optimization.json',ROOT/'configs/decision_principle.json',
            ROOT/'support_selection/quality.py',ROOT/'support_selection/experiment.py']},
        input_hashes={path:sha256(ROOT/path) for path in count_cache})
    marker=out/'suite_manifest.json'
    if marker.exists() and io.read_json(marker)!=manifest:
        raise ValueError('Frozen suite code/input manifest differs; use a new OutputRoot')
    if not marker.exists():io.write_json(marker,manifest)
    return manifest


def execute(stage,manifest,out):
    if stage=='evaluate' and not (out/'all_source_searches_complete.json').is_file():
        raise ValueError('All suite source searches must finish before any target evaluation')
    for index,case in enumerate(manifest['cases'],1):
        print(f'SUITE {stage.upper()} {index}/{len(manifest["cases"])} {case["name"]}',flush=True)
        command=[sys.executable,'-u',str(ROOT/'selector_optimization.py'),'--stage',stage,
            '--config',str(ROOT/case['config']),'--out',str(ROOT/case['result']),'--datasets',case['dataset']]
        subprocess.run(command,cwd=ROOT,check=True)
        io.write_json(out/'suite_progress.json',dict(stage=stage,completed_cases=index,
            total_cases=len(manifest['cases']),latest=case['name']))
    if stage=='tune':
        freezes=[]
        for case in manifest['cases']:
            path=ROOT/case['result']/'tuning_complete.json'
            value=io.read_json(path)
            if not value['complete'] or value['tasks']!=case['tasks']:raise ValueError('Task coverage mismatch')
            freezes.append(dict(path=str(path.relative_to(ROOT)),sha256=sha256(path)))
        io.write_json(out/'all_source_searches_complete.json',dict(complete=True,freezes=freezes,
            tasks=manifest['source_tuning_tasks'],target_evaluation_started=False))


def collect(manifest,out):
    import pandas as pd
    output=out/'analysis'
    metrics=['absolute_mean_sd.csv','paired_comparisons.csv','mechanism_summary.csv',
             'switch_downside_summary.csv','recipient_effects_per_seed.csv','frozen_parameters.csv']
    for filename in metrics:
        frames=[]
        for case in manifest['cases']:
            result=ROOT/case['result']
            if not io.read_json(result/'analysis/report_status.json')['complete']:raise ValueError('Case incomplete')
            f=pd.read_csv(result/'analysis'/filename)
            for key in ['name','family','cutoff','mode']:f['experiment_'+key]=case[key]
            frames.append(f)
        io.write_csv(output/filename,pd.concat(frames,ignore_index=True))
    io.write_json(out/'suite_complete.json',dict(complete=True,cases=len(manifest['cases']),
        evaluated_tasks=manifest['target_evaluation_tasks'],all_results_retained=True,
        final_ours='tuned_ours',old_results_preserved=True,manuscript_updated=False))
    print('REPLACEMENT SUITE COMPLETE '+str(output),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['plan','tune','evaluate','summarize','all'],default='plan')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/selector_replacement_suite_v1')
    args=parser.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to((ROOT/'outputs').resolve()) or out==(ROOT/'outputs').resolve():
        raise ValueError('Use a new subdirectory under project outputs')
    out.mkdir(parents=True,exist_ok=True)
    manifest=prepare(out)
    if args.stage=='plan':
        print(json.dumps({k:v for k,v in manifest.items() if k not in ['hashes','input_hashes']},indent=2));return
    if args.stage in ['tune','all']:execute('tune',manifest,out)
    if args.stage in ['evaluate','all']:
        frozen=io.read_json(out/'all_source_searches_complete.json')
        for item in frozen['freezes']:
            if sha256(ROOT/item['path'])!=item['sha256']:raise ValueError('Suite freeze index changed')
        execute('evaluate',manifest,out)
    if args.stage in ['summarize','all']:
        execute('summarize',manifest,out)
        collect(manifest,out)


if __name__=='__main__':main()

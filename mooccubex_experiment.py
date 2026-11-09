"""Registered third-dataset task: future recorded video inactivity, not dropout."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta, timezone
import argparse
import json
import math
import os
import sqlite3
import time

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'analysis/.matplotlib'))
import numpy as np
import pandas as pd
from support_selection import quality as quality
from support_selection.data import dump_json,sha256
from support_selection.experiment import stable_hash

TZ=timezone(timedelta(hours=8))
OBS_START=datetime(2020,1,20,tzinfo=TZ).timestamp()
OBS_END=datetime(2020,11,1,tzinfo=TZ).timestamp()
GAPS=[(datetime(2020,6,20,tzinfo=TZ).timestamp(),datetime(2020,6,21,tzinfo=TZ).timestamp())]
CUTOFFS=[7,14,21]
HORIZON=14
OUTCOME='video_inactivity_14d'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def normalize_id(value,prefix):
    value=str(value)
    return value if value.startswith(prefix) else prefix+value


def epoch(text):
    try:
        value=datetime.fromisoformat(str(text))
        return (value.replace(tzinfo=TZ) if value.tzinfo is None else value).timestamp()
    except (ValueError,TypeError,OverflowError):return None


def summarize_events(times,enrollment,cutoff):
    """Only feature events <= decision; future events define the outcome separately."""
    decision=enrollment+cutoff*86400
    followup=decision+HORIZON*86400
    past=[t for t in times if enrollment<=t<=decision]
    if not past:return None
    recent=[t for t in past if t>decision-7*86400]
    active_days={int((t-enrollment)//86400) for t in past}
    future=any(decision<t<=followup for t in times)
    return {'log_clicks':math.log1p(len(past)),'log_events':math.log1p(len(past)),
            'log_active_days':math.log1p(len(active_days)),'log_recent_clicks':math.log1p(len(recent)),
            'recent_fraction':len(recent)/len(past),
            'inactivity_fraction':min(1,max(0,(decision-max(past))/86400/(cutoff+1))),
            'y':int(not future)}


def lock(out,raw):
    original=ROOT/'outputs/quality_kdd2015_primary_20261028_164729/config.json'
    protocol={'version':'mooccubex-v3','source':'https://github.com/THU-KEG/MOOCCubeX',
        'period_reference':'https://lfs.aminer.cn/misc/moocdata/publications/mooccubex.pdf, Section 3.1.2',
        'observed_video_period':'2020-01-19 through 2020-11-03',
        'conservative_eligibility_period':'[2020-01-20 00:00, 2020-11-01 00:00), UTC+08:00',
        'coverage_audit':'Full download: 2020-06-20 has no video records; 2020-11-01 has only 23 segments. Exclude enrollment windows crossing the missing day and the partial final day.',
        'timestamp_assumption':'Naive enrollment timestamps are UTC+08:00; local_start_time is Unix seconds. Validate ranges.',
        'cutoffs':CUTOFFS,'future_horizon_days':HORIZON,'primary_cutoff':14,'primary_capacity':.2,
        'target_course_cap':30,'minimum_initially_active_registrations_at_day14':100,
        'capacity_cohort':'Course by enrollment calendar month; >=30 initially active records per cutoff/cohort.',
        'eligibility':'Enroll in conservative period and complete 21+14-day follow-up; >=1 recorded video segment by each decision.',
        'sampling':'Eligible courses ordered by SHA256(seed=20261028, course_id), first 30; no outcome filtering.',
        'scope':'Video viewing only. No recorded future viewing is not administrative dropout or intervention benefit. '
                'Same-platform supplemental dataset; no frozen cross-dataset transfer.',
        'code_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'support_selection/quality.py',ROOT/'support_selection/experiment.py',original]},
        'input_hashes':{rel:sha256(raw/rel) for rel in ['entities/course.json','entities/user.json','relations/user-video.json']}}
    path=out/'protocol.json'
    if path.exists() and read(path)!=protocol:raise ValueError('Protocol/code/data changed; choose a new -OutputRoot')
    if not path.exists():dump_json(path,protocol)
    return protocol


def prepare(out,raw):
    if (out/'preparation_complete.json').exists():
        info=read(out/'preparation_complete.json')
        for path,digest in info['processed_hashes'].items():
            if sha256(path)!=digest:raise ValueError('Processed data changed')
        print('SKIP verified preparation',flush=True);return
    if (out/'registrations.sqlite').exists():
        raise ValueError('Incomplete preparation preserved. Choose a new -OutputRoot; no overwrite.')
    course_ids=set();video_course={};ambiguous=set()
    for line in (raw/'entities/course.json').open(encoding='utf-8'):
        row=json.loads(line);course_ids.add(row['id'])
        for item in row.get('resource',[]):
            video=item.get('resource_id','')
            if not video.startswith('V_'):continue
            if video in video_course and video_course[video]!=row['id']:ambiguous.add(video)
            else:video_course[video]=row['id']
    for video in ambiguous:video_course.pop(video,None)
    db=sqlite3.connect(out/'registrations.sqlite')
    db.execute('CREATE TABLE enroll(u TEXT,c TEXT,e REAL,PRIMARY KEY(u,c))')
    db.execute('CREATE TABLE features(cutoff INTEGER,c TEXT,u TEXT,cohort TEXT,payload TEXT,PRIMARY KEY(cutoff,c,u))')
    counts=Counter();batch=[];seen_users=set()
    for line in (raw/'entities/user.json').open(encoding='utf-8'):
        row=json.loads(line);counts['profiles']+=1
        courses=row.get('course_order',[]);dates=row.get('enroll_time',[])
        if len(courses)!=len(dates):counts['profile_array_mismatch']+=1;continue
        # Profile values other than IDs and registration dates are not used or exported.
        for course,date in zip(courses,dates):
            course=normalize_id(course,'C_')
            enrollment=epoch(date)
            if course not in course_ids or enrollment is None:counts['invalid_course_or_date']+=1;continue
            if not (OBS_START<=enrollment and enrollment+(max(CUTOFFS)+HORIZON)*86400<=OBS_END):
                counts['outside_complete_observation_period']+=1;continue
            if any(enrollment<end and enrollment+(max(CUTOFFS)+HORIZON)*86400>=begin for begin,end in GAPS):
                counts['crosses_missing_observation_day']+=1;continue
            batch.append((normalize_id(row['id'],'U_'),course,enrollment));counts['eligible_registration_entries']+=1
        if len(batch)>=10000:
            db.executemany('INSERT INTO enroll VALUES(?,?,?) ON CONFLICT(u,c) DO UPDATE SET e=MIN(e,excluded.e)',batch)
            db.commit();batch=[]
    if batch:db.executemany('INSERT INTO enroll VALUES(?,?,?) ON CONFLICT(u,c) DO UPDATE SET e=MIN(e,excluded.e)',batch)
    db.commit();counts['unique_eligible_registrations']=db.execute('SELECT COUNT(*) FROM enroll').fetchone()[0]
    if counts['unique_eligible_registrations']==0:raise ValueError('No eligible registrations')
    first_time=float('inf');last_time=float('-inf');rows=[]
    for line in (raw/'relations/user-video.json').open(encoding='utf-8'):
        row=json.loads(line);counts['video_user_rows']+=1
        uid=normalize_id(row['user_id'],'U_')
        if uid in seen_users:raise ValueError('Duplicate video user rows need explicit merging; no silent overwrite')
        seen_users.add(uid)
        enroll=dict(db.execute('SELECT c,e FROM enroll WHERE u=?',(uid,)).fetchall())
        if not enroll:continue
        activity={c:[] for c in enroll}
        for video in row.get('seq',[]):
            course=video_course.get(normalize_id(video.get('video_id'),'V_'))
            if course not in activity:continue
            for segment in video.get('segment',[]):
                try:t=float(segment['local_start_time'])
                except (KeyError,TypeError,ValueError):counts['invalid_event_times']+=1;continue
                if not np.isfinite(t):counts['invalid_event_times']+=1;continue
                first_time=min(first_time,t);last_time=max(last_time,t)
                if not (OBS_START<=t<OBS_END):counts['events_outside_conservative_period']+=1;continue
                activity[course].append(t)
        for course,times in activity.items():
            enrollment=enroll[course]
            for cutoff in CUTOFFS:
                values=summarize_events(times,enrollment,cutoff)
                if values is None:continue
                local=datetime.fromtimestamp(enrollment,TZ)
                start=local.replace(tzinfo=None)
                payload={'dataset':'mooccubex','outcome':OUTCOME,'record_id':f'mooccubex|{course}|{uid}',
                    'student_id':uid,'course':course,'cohort':f'{course}|{local.year}-{local.month:02d}',
                    'start':start.isoformat(),'label_available':(start+timedelta(days=cutoff+HORIZON)).isoformat(),
                    'cutoff':cutoff,**values}
                rows.append((cutoff,course,uid,payload['cohort'],json.dumps(payload)))
        if len(rows)>=5000:
            db.executemany('INSERT INTO features VALUES(?,?,?,?,?)',rows);db.commit();rows=[]
        if counts['video_user_rows']%10000==0:print('PREPARE video users',counts['video_user_rows'],flush=True)
    if rows:db.executemany('INSERT INTO features VALUES(?,?,?,?,?)',rows)
    db.commit()
    db.execute('DELETE FROM features WHERE (cutoff,c,cohort) NOT IN (SELECT cutoff,c,cohort FROM features GROUP BY cutoff,c,cohort HAVING COUNT(*)>=30)')
    db.commit()
    eligible=[r[0] for r in db.execute('SELECT c FROM features WHERE cutoff=14 GROUP BY c HAVING COUNT(*)>=100')]
    chosen=sorted(eligible,key=lambda c:stable_hash(c,20261028))[:30]
    if len(chosen)<10:raise ValueError(f'Only {len(chosen)} eligible courses; inspect coverage before training')
    dump_json(out/'selected_courses.json',{'courses':chosen,'eligible_course_count':len(eligible),
        'criterion':'Day14 past-activity count >=100; ID hash order; no outcome filtering'})
    hashes={};samples=[]
    for cutoff in CUTOFFS:
        records=[json.loads(payload) for course in chosen for (payload,) in db.execute('SELECT payload FROM features WHERE cutoff=? AND c=? ORDER BY u',(cutoff,course))]
        df=pd.DataFrame(records);quality.validate_frame(df)
        path=out/f'mooccubex_{OUTCOME}_{cutoff}.csv.gz';df.to_csv(path,index=False);hashes[str(path)]=sha256(path)
        samples.append({'cutoff':cutoff,'registrations':len(df),'positive':int(df.y.sum()),'courses':int(df.course.nunique())})
    db.close()
    dump_json(out/'preparation_complete.json',{'complete':True,'processed_hashes':hashes,'samples':samples,
        'counts':dict(counts),'ambiguous_videos_excluded':len(ambiguous),
        'mapped_event_timestamp_range':[first_time,last_time],
        'scope':f'Initially active, sufficiently followed registrations in {len(chosen)} courses selected by fixed ID order, capped at 30.'})
    print('PREPARATION COMPLETE',json.dumps(samples),flush=True)


def run(out,stage):
    if not (out/'preparation_complete.json').exists():raise ValueError('Execute prepare first')
    for path,digest in read(out/'preparation_complete.json')['processed_hashes'].items():
        if sha256(path)!=digest:raise ValueError('Processed data changed; preserve output and re-prepare in a new root')
    cfg=read(ROOT/'outputs/quality_kdd2015_primary_20261028_164729/config.json')
    cfg.update(dataset='mooccubex',outcome=OUTCOME,stage='third_'+stage,
        cutoffs=[14] if stage=='primary' else CUTOFFS,
        budgets=[.2] if stage=='primary' else [.1,.2,.3],feature_modes=['behavior'],source_course_counts=[8],
        note='Predefined third dataset; recorded future video inactivity, not administrative withdrawal. Same-platform replication, separately retrained.')
    destination=out/stage
    if destination.exists():
        if (destination/'run_complete.json').exists() and read(destination/'run_complete.json').get('complete') and read(destination/'config.json')==cfg:
            print('SKIP complete:',destination,flush=True);return
        raise ValueError('Incomplete training preserved; choose new output root')
    paths=[out/f'mooccubex_{OUTCOME}_{c}.csv.gz' for c in cfg['cutoffs']]
    quality.run_quality(paths,cfg,destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','primary','sensitivity','all'],default='all')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/mooccubex_v3')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    raw=ROOT/'data/raw/mooccubex'
    lock(out,raw)
    if args.stage in ['all','prepare']:prepare(out,raw)
    if args.stage in ['all','primary']:run(out,'primary')
    if args.stage in ['all','sensitivity']:run(out,'sensitivity')


if __name__=='__main__':main()

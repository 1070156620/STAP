"""Download official MOOCCubeX files or inspect bounded samples; never invent labels."""
from pathlib import Path
import argparse
import json
import shutil
import urllib.request
import time
from support_selection.data import sha256, dump_json

ROOT=Path(__file__).resolve().parent
BASE='https://lfs.aminer.cn/misc/moocdata/data/mooccube2/'
FILES={'course':'entities/course.json','user':'entities/user.json','video':'relations/user-video.json'}


def request(url,headers=None,method=None):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'EducationalResearchDataCheck/1.0',**(headers or {})},method=method),timeout=60)


def probe(out):
    result=[]
    for name,rel in FILES.items():
        url=BASE+rel
        with request(url,{'Range':'bytes=0-1048575'}) as response:
            chunk=response.read(1048576)
            content_range=response.headers.get('Content-Range')
            length=response.headers.get('Content-Length')
            info={'name':name,'url':url,'http_status':response.status,'sample_bytes':len(chunk),
                  'content_range':content_range,'content_length':length,
                  'etag':response.headers.get('ETag'),'last_modified':response.headers.get('Last-Modified')}
        sample=out/'samples'/f'{name}.partial.jsonl';sample.parent.mkdir(parents=True,exist_ok=True);sample.write_bytes(chunk)
        # Inspect schema only; do not print names, birth dates, profile values or labels.
        keys=[];seq_keys=[]
        for line in chunk.decode('utf-8',errors='ignore').splitlines()[:10]:
            try: record=json.loads(line)
            except json.JSONDecodeError: continue
            if isinstance(record,dict):
                keys=sorted(record)
                if isinstance(record.get('seq'),list) and record['seq'] and isinstance(record['seq'][0],dict):
                    seq_keys=sorted(record['seq'][0])
                break
        info.update(top_level_keys=keys,sequence_item_keys=seq_keys,sample_sha256=sha256(sample))
        result.append(info);print(name,'schema:',keys,'sequence fields:',seq_keys,flush=True)
    dump_json(out/'schema_probe.json',{'files':result,'ready_for_training':False,
        'reason':'Only bounded schema samples. No endpoint or observation-completeness verification yet.'})
    print('Probe saved:',out,flush=True)


def inspect_course(out):
    path=out/'entities/course.json'
    if not path.exists():return
    count=0;keys=set();videos={};unknown_resources=0
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);count+=1;keys.update(row)
            for resource in row.get('resource',[]):
                rid=resource.get('resource_id','')
                if rid.startswith('V_'):videos.setdefault(rid,set()).add(row['id'])
                elif not rid.startswith('Ex_'):unknown_resources+=1
    result={'courses':count,'course_keys':sorted(keys),
            'documented_time_fields':[k for k in sorted(keys) if any(t in k.lower() for t in ['time','date','start','end'])],
            'video_resources':len(videos),'ambiguous_video_resources':sum(len(v)>1 for v in videos.values()),
            'unknown_resource_prefixes':unknown_resources,'ready_for_training':False,
            'reason':'Need verified observation horizon and course-time interpretation before constructing outcomes.'}
    dump_json(out/'course_feasibility.json',result)
    print('Course feasibility:',json.dumps(result,ensure_ascii=False),flush=True)


def download(out):
    manifest=[]
    for name,rel in FILES.items():
        path=out/rel;path.parent.mkdir(parents=True,exist_ok=True)
        metadata=path.with_suffix(path.suffix+'.download.json')
        if path.exists():
            if not metadata.exists() or read(metadata).get('sha256')!=sha256(path):
                raise ValueError(f'Existing file not verified; preserved: {path}')
            print('SKIP verified:',path,flush=True);manifest.append(read(metadata));continue
        url=BASE+rel
        with request(url,method='HEAD') as response:
            total=int(response.headers['Content-Length'])
            current_tag=response.headers.get('ETag') or response.headers.get('Last-Modified')
        partial=path.with_suffix(path.suffix+'.part');resume_meta=partial.with_suffix(partial.suffix+'.json')
        offset=partial.stat().st_size if partial.exists() else 0
        if offset and (not resume_meta.exists() or read(resume_meta).get('server_tag')!=current_tag):
            raise ValueError(f'Partial file server version unknown/changed; preserved: {partial}')
        if offset>total:raise ValueError(f'Partial larger than source: {partial}')
        if shutil.disk_usage(out).free < max(0,total-offset)+1024**3:
            raise ValueError('Insufficient free space for remaining file plus 1 GiB reserve')
        dump_json(resume_meta,{'url':url,'server_tag':current_tag,'total_bytes':total})
        if offset<total:
            headers={'Range':f'bytes={offset}-'} if offset else {}
            with request(url,headers) as response:
                if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                    raise ValueError('Server did not honor resume range; partial preserved, no append')
                if current_tag and (response.headers.get('ETag') or response.headers.get('Last-Modified')) != current_tag:
                    raise ValueError('Source version changed between HEAD and GET')
                with partial.open('ab' if offset else 'wb') as stream:
                    last=time.monotonic()
                    while True:
                        chunk=response.read(4*1024**2)
                        if not chunk:break
                        stream.write(chunk);offset+=len(chunk)
                        if time.monotonic()-last>10:
                            print(f'DOWNLOAD {name}: {offset/1024**2:.1f}/{total/1024**2:.1f} MiB',flush=True);last=time.monotonic()
        if partial.stat().st_size!=total:raise ValueError('Incomplete download; rerun to resume')
        # Store checksum before promoting the downloaded file; existing destination never overwritten.
        digest=sha256(partial)
        if path.exists():raise FileExistsError(path)
        partial.rename(path)
        record={'url':url,'path':str(path),'bytes':total,'sha256':digest,'server_tag':current_tag}
        dump_json(metadata,record);manifest.append(record)
        print('DOWNLOADED:',path,flush=True)
    dump_json(out/'download_manifest.json',{'files':manifest,'ready_for_training':False,
        'reason':'Complete downloads do not establish reliable labels or observation horizons.'})


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['probe','download'],default='probe')
    parser.add_argument('--out',type=Path,default=ROOT/'data/raw/mooccubex')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    (probe if args.stage=='probe' else download)(args.out)
    inspect_course(args.out)


if __name__=='__main__':main()

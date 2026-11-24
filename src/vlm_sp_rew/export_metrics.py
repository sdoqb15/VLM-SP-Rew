"""Export balanced per-seed/per-pair SR and SPL from an episode CSV."""
import argparse,csv,json,math,statistics
from collections import defaultdict
from pathlib import Path

FIELDS=['record_kind','episode_id','benchmark','group','seed','pair_id','checkpoint_steps','success','spl']
def summarize(path,output,checkpoint_steps=400000):
    with Path(path).open(newline='',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    if not rows:raise ValueError('Episode CSV is empty')
    if not set(FIELDS)<=set(rows[0]):raise ValueError('Required columns: '+','.join(FIELDS))
    kinds={r['record_kind']for r in rows}
    if len(kinds)!=1 or not kinds<= {'measured','synthetic_reference'}:raise ValueError('One explicit record_kind required; do not mix kinds')
    rows=[r for r in rows if int(r['checkpoint_steps'])==checkpoint_steps]
    if not rows:raise ValueError('Requested checkpoint has no episodes')
    if len({r['episode_id']for r in rows})!=len(rows):raise ValueError('Duplicate episode_id')
    kind=next(iter(kinds));byseed=defaultdict(list);bypairseed=defaultdict(list)
    for r in rows:
        s=float(r['success']);spl=float(r['spl'])
        if s not in [0,1]or not math.isfinite(spl)or not 0<=spl<=s:raise ValueError('Require binary success and0<=SPL<=success')
        r=dict(r,success=int(s),spl=spl,seed=int(r['seed']))
        byseed[r['benchmark'],r['group'],r['seed']].append(r)
        bypairseed[r['benchmark'],r['group'],r['pair_id'],r['seed']].append(r)
    layout={}
    for (bench,group,seed),rr in byseed.items():
        signature=sorted((pair,sum(r['pair_id']==pair for r in rr))for pair in {r['pair_id']for r in rr})
        if bench in layout and layout[bench]!=signature:raise ValueError('Unmatched pair/episode counts within benchmark')
        layout[bench]=signature
    group_seeds={}
    for bench,group,seed in byseed:group_seeds.setdefault((bench,group),set()).add(seed)
    seedsets={}
    for (bench,group),ss in group_seeds.items():
        if len(ss)<2:raise ValueError('At least two fitted seeds required for sample SD')
        if bench in seedsets and seedsets[bench]!=ss:raise ValueError('Unmatched seeds across groups')
        seedsets[bench]=ss
    def metric(rr):
        return dict(episodes=len(rr),successes=sum(r['success']for r in rr),SR_percent=100*statistics.mean(r['success']for r in rr),SPL=statistics.mean(r['spl']for r in rr))
    perseed=[dict(record_kind=kind,benchmark=b,group=g,seed=s,**metric(rr))for (b,g,s),rr in sorted(byseed.items())]
    perpairseed=[dict(record_kind=kind,benchmark=b,group=g,pair_id=p,seed=s,**metric(rr))for (b,g,p,s),rr in sorted(bypairseed.items())]
    def across(rr):
        return dict(seeds=len(rr),episodes=sum(r['episodes']for r in rr),successes=sum(r['successes']for r in rr),SR_mean_percent=statistics.mean(r['SR_percent']for r in rr),SR_sample_SD_percent=statistics.stdev(r['SR_percent']for r in rr),SPL_mean=statistics.mean(r['SPL']for r in rr),SPL_sample_SD=statistics.stdev(r['SPL']for r in rr))
    groupseed=defaultdict(list);pairseed=defaultdict(list)
    for r in perseed:groupseed[r['benchmark'],r['group']].append(r)
    for r in perpairseed:pairseed[r['benchmark'],r['group'],r['pair_id']].append(r)
    summary=[dict(record_kind=kind,benchmark=b,group=g,**across(rr))for (b,g),rr in sorted(groupseed.items())]
    perpair=[dict(record_kind=kind,benchmark=b,group=g,pair_id=p,**across(rr))for (b,g,p),rr in sorted(pairseed.items())]
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for name,table in [('per_seed.csv',perseed),('per_pair_seed.csv',perpairseed),('per_pair.csv',perpair),('summary.csv',summary)]:
        with(output/name).open('w',newline='',encoding='utf-8')as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    manifest=dict(record_kind=kind,checkpoint_steps=checkpoint_steps,input_episodes=len(rows),per_seed_rows=len(perseed),per_pair_seed_rows=len(perpairseed),per_pair_rows=len(perpair),summary_rows=len(summary),statistics='Equalweight fitted-seed means; sample SD with ddof=1; matchedpair andepisodecounts withinbenchmark; no checkpoint selection by these outcomes.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return dict(manifest=manifest,summary=summary,per_seed=perseed,per_pair_seed=perpairseed,per_pair=perpair)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--out',required=True);p.add_argument('--checkpoint-steps',type=int,default=400000);a=p.parse_args()
    r=summarize(a.input,a.out,a.checkpoint_steps);print(json.dumps(r['manifest'],indent=2))
if __name__=='__main__':main()

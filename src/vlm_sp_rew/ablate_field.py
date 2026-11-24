"""Derive semantic signal ablations on exactly the original sampled graph."""
import argparse, copy, json, hashlib
from pathlib import Path
from vlm_sp_rew.reward import require
from vlm_sp_rew.annotation_cache import audit_existing_cache
from vlm_sp_rew.field_contract import PaperParameters, value_from_hops
import yaml


def derive(field, cache_audit, config, variant):
    require(variant in ('visibility','distance'),'Unknown semantic signal ablation')
    result=copy.deepcopy(field);p=PaperParameters.from_config(config)
    visible={(x['position']['x'],x['position']['z']):x['parsed']['goal_visible'] for x in cache_audit['points']}
    for point in result['points']:
        key=(point['position']['x'],point['position']['z'])
        require(key in visible,'Point does not match immutable annotation cache')
        point['value']=(p.alpha[0] if visible[key] else p.beta[0]) if variant=='visibility' else value_from_hops(False,point['graph_hops'],p)
    result['signal_variant']=variant
    result['provenance']['parent_field_sha256']=hashlib.sha256(json.dumps(field,sort_keys=True).encode()).hexdigest()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('field','source-cache','goal-reference','config','variant','out'):p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.out)
    audit=audit_existing_cache(a.source_cache,a.goal_reference)
    result=derive(json.loads(Path(a.field).read_text()),audit,yaml.safe_load(Path(a.config).read_text()),a.variant)
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')


if __name__=='__main__':main()

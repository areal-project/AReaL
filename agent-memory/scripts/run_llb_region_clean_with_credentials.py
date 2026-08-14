#!/usr/bin/env python3
"""Run a job-local patched LLB Region launcher with redacted Matrix credentials."""
from __future__ import annotations
import json
import os
from pathlib import Path


def install_credentials() -> None:
    import yaml
    from memrl.configs.config import MempConfig
    cfg_path = Path(os.environ.get('MEMRL_MATRIX_CREDENTIAL_CONFIG', '/storage/openpsi/users/yl/cfuse/rq3_v11_clean_20260719/config_multisurface_isolated.yaml'))
    payload = yaml.safe_load(cfg_path.read_text()) or {}
    maps = {str(x.get('model_name')): (x.get('litellm_params') or {}) for x in payload.get('model_list', []) if isinstance(x, dict)}
    def resolve(names):
        for name in names:
            value = maps.get(name, {}).get('api_key')
            if isinstance(value, str) and value.startswith('os.environ/'):
                value = os.environ.get(value.split('/', 1)[1])
            if value:
                return str(value)
        raise RuntimeError(f'Missing Matrix credential aliases: {names}')
    chat_key = resolve(('gpt-4o-2024-11-20', 'gpt-4o', 'gpt-5-mini'))
    embed_key = resolve(('text-embedding-3-large', 'text-embedding-3-small'))
    original = MempConfig.from_yaml.__func__
    @classmethod
    def patched_from_yaml(cls, path):
        config = original(cls, path)
        config.llm.api_key = chat_key
        config.embedding.api_key = embed_key
        return config
    def redacted_dump(self, *args, **kwargs):
        data = self.model_dump(mode='json')
        for key in ('llm', 'embedding'):
            if isinstance(data.get(key), dict):
                data[key]['api_key'] = '[REDACTED]'
        return json.dumps(data, ensure_ascii=False, indent=kwargs.get('indent'))
    MempConfig.from_yaml = patched_from_yaml
    MempConfig.model_dump_json = redacted_dump


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    source_path = project / 'run' / 'run_llb.py'
    source = source_path.read_text()
    # Different shared checkout revisions expose different subsets of Region
    # controls. Add only controls absent from the constructor body, immediately
    # after the stable propagation arguments, so this remains job-local.
    constructor_start = source.index('            region_manager = RegionManager(')
    constructor_end = source.index('            )\n            if args.shrinkage_confidence_k', constructor_start)
    constructor = source[constructor_start:constructor_end]
    missing = []
    if 'region_split_evidence_migration_mode=' not in constructor:
        missing.append('                region_split_evidence_migration_mode=os.environ.get("MEMRL_REGION_SPLIT_EVIDENCE_MIGRATION_MODE", "soft_source_conserving"),')
    if 'region_topology_updates_enabled=' not in constructor:
        missing.append('                region_topology_updates_enabled=os.environ.get("MEMRL_REGION_TOPOLOGY_UPDATES_ENABLED", "1").lower() not in {"0", "false", "no", "off"},')
    if 'region_evidence_sharpen_alpha=' not in constructor:
        missing.append('                region_evidence_sharpen_alpha=float(os.environ.get("MEMRL_REGION_EVIDENCE_SHARPEN_ALPHA", "2.0")),')
    if missing:
        anchor = '                propagation_sim_min=0.40,\n'
        if anchor not in constructor:
            raise RuntimeError('RegionManager propagation_sim_min patch anchor not found')
        injected = anchor + "\n".join(missing) + "\n"
        patched = source.replace(anchor, injected, 1)
        patch_mode = 'job-local compatibility patch'
    else:
        patched = source
        patch_mode = 'native'
    install_credentials()
    print(f'[Region clean] launcher mode={patch_mode}; source-conserving split + topology controlled by environment', flush=True)
    namespace = {'__name__': '__main__', '__file__': str(source_path), '__package__': None}
    exec(compile(patched, str(source_path), 'exec'), namespace)

if __name__ == '__main__':
    main()

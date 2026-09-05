#!/usr/bin/env python3
"""Fail-closed M5 qualification: Metal compile, native tests, then A/B timings."""
import argparse
import datetime
import json
import platform
import re
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,default=Path('results')/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    p.add_argument('--tokens',default='1,8,128,2048');p.add_argument('--dtype',default='bfloat16',choices=('float32','float16','bfloat16'))
    p.add_argument('--batch',type=int,default=1);a=p.parse_args()
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    status={'component_validation_passed':False,'performance_qualified':False,'compiler':'not_run','native_tests':'not_run','benchmark':'not_run'}
    def write(): (out/'status.json').write_text(json.dumps(status,indent=2))
    def run(name,cmd):
        print(name,flush=True)
        with (out/(name+'.log')).open('w') as log:
            proc=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:raise RuntimeError(f'{name} failed ({proc.returncode}); see {out/(name+".log")}')
    try:
        if platform.system()!='Darwin':raise RuntimeError('Native M5 checks require macOS, not a host shim')
        import mlx.core as mx
        if not mx.metal.is_available():raise RuntimeError('Metal is unavailable')
        info=mx.device_info(mx.gpu);status['device']=info
        name=str(info.get('device_name',info.get('name','')))
        chip=re.search(r'\bApple M(\d+)\b',name)
        if not chip or int(chip.group(1))<5:raise RuntimeError(f'Expected an M5-or-newer GPU, got {name!r}')
        run('emit',[sys.executable,str(root/'tools/emit_metal.py'),str(out/'gdn_prepare.metal')])
        run('compile',['xcrun','-sdk','macosx','metal','-std=metal3.2','-fno-fast-math','-c',str(out/'gdn_prepare.metal'),'-o',str(out/'gdn_prepare.air')])
        run('link',['xcrun','-sdk','macosx','metallib',str(out/'gdn_prepare.air'),'-o',str(out/'gdn_prepare.metallib')])
        status['compiler']='passed';write()
        run('native-tests',[sys.executable,str(root/'tests/test_native.py')])
        status['native_tests']='passed';write()
        run('benchmark',[sys.executable,str(root/'tools/benchmark.py'),'--tokens',a.tokens,'--dtype',a.dtype,'--batch',str(a.batch),'--output',str(out/'timings.json')])
        status['benchmark']='completed'
        status['component_validation_passed']=True
        status['qualification_scope']='Build and native component tests passed; preprocessing timings recorded. Whole-model correctness and performance remain unqualified.'
    except Exception as e:
        status['error']=str(e);write();print(str(e),file=sys.stderr);return 1
    write();print(out/'status.json');return 0

if __name__=='__main__':sys.exit(main())

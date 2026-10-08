"""Generate and run focused regressions using an existing Mojo M2NDP image.

Generate on the host with --generate; run inside the image with --run.
The image's MappedDomain spelling is adapted only at the host launch boundary.
An isolated copy of the image's driver adds libm for the host DFT reference.
No runtime installation or simulator/config change is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / 'verification/results/execution_codegen_fixes_2026-10-08'


def domain_bridge(source):
    source = source.replace('NDPTask, PooledRange,',
                            'NDPTask, Domain, MappedDomain, UTHREAD_BYTES,')
    bridge = '''
struct PooledRange:
    @staticmethod
    def over(data: UnsafePointer[Float32, MutAnyOrigin], count: Int) -> MappedDomain:
        return MappedDomain(data, Domain(count * size_of[Float32]() // UTHREAD_BYTES))
'''
    return source.replace('comptime W = VECTOR_WIDTH // size_of[Float32]()',
                          'comptime W = VECTOR_WIDTH // size_of[Float32]()\n' + bridge)


def generate():
    from dataclasses import replace
    from verification.verify_fft_execution_codegen_fixes import plain_plan
    from planning.core.target_profile import DEFAULT_TARGET_PROFILE as target
    from planning.execution.fft_plan_cooperative import make_cooperative_leaf_plan
    from planning.execution.fft_plan_persistent import make_persistent_leaf_plan, make_persistent_tail_hybrid_plan
    from codegen.fft_codegen import generate_fft_kernel
    from codegen.fft_cooperative_codegen import generate_cooperative_fft_kernel
    from codegen.fft_persistent_codegen import generate_persistent_fft_kernel, generate_persistent_tail_hybrid_kernel
    from planning.strategies.fft_plan_recursive import make_recursive_transpose_plan
    from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases = []
    def add(name, source, config='M2NDP'):
        (RESULTS / f'{name}.mojo').write_text(domain_bridge(source), encoding='utf-8')
        cases.append(dict(name=name, config=config,
                          execution=[json.loads(s.split(' ', 2)[2]) for s in source.splitlines()
                                     if s.startswith('# FFT_EXECUTION ')]))
    add('plain_cap1_n64_b64', generate_fft_kernel(plain_plan(batch=64), compute_lanes=4,
                                                narrow_middle_stages=True))
    coop = make_cooperative_leaf_plan(64, (4, 4, 4), workers_per_fft=8, total_ffts=65,
                                      spad_capacity_bytes=1024)
    add('coop_cap2_n64_b65', generate_cooperative_fft_kernel(coop, compute_lanes=4, in_place=True))
    for mode, workers, batch in (('physical', 16, 33), ('fused', 17, 33), ('wave', 10, 65)):
        p = make_persistent_leaf_plan(64, (4, 4, 4), num_logical_blocks=batch,
                                      workers_per_fft=workers, lowering_mode=mode)
        add(f'persistent_{mode}_w{workers}_b{batch}',
            generate_persistent_fft_kernel(p, num_logical_blocks=batch,
                                          copy_mode='vectorized_contiguous', vector_width=4))
    rtx = replace(target, num_ndp_units=28)
    p = make_persistent_leaf_plan(64, (4, 4, 4), num_logical_blocks=57,
                                  workers_per_fft=16, lowering_mode='wave', target=rtx, inverse=True)
    add('persistent_wave_rtx3060_inverse_b57',
        generate_persistent_fft_kernel(p, num_logical_blocks=57, target=rtx), 'M2NDP_RTX3060')
    p = make_persistent_tail_hybrid_plan(64, (4, 4, 4), num_logical_blocks=65,
                                         tail_strategy='cooperative_tail', tail_cooperative_workers=8)
    add('hybrid_cooperative_b65', generate_persistent_tail_hybrid_kernel(p))
    p = make_recursive_transpose_plan(960, batch=2, scratchpad_byte_budget=512,
                                      target=target, cooperative_workers=4, spread_across_units=True)
    add('recursive_cooperative_n960_b2', generate_recursive_fft_kernels(
        p, compute_lanes=4, narrow_middle_stages=True, spread_across_units=True))
    (RESULTS / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n', encoding='utf-8')
    print(f'Generated {len(cases)} cases in {RESULTS}')


def run(names):
    driver = Path('/tmp/fft-execution-driver')
    text = Path('/work/mojo-ndp').read_text()
    text = text.replace('REPO="$(cd "$(dirname "$0")" && pwd)"', 'REPO="/work"')
    text = text.replace('-Xlinker "$work/host_stubs.o"', '-Xlinker "$work/host_stubs.o" -Xlinker -lm')
    driver.write_text(text)
    cases = json.loads((RESULTS / 'cases.json').read_text())
    selected = [case for case in cases if not names or case['name'] in names]
    if names and len(selected) != len(set(names)):
        raise ValueError('Unknown simulation case')
    failed = False
    for case in selected:
        name = case['name']
        binary = '/tmp/fft-execution-' + name
        build = subprocess.run(['bash', str(driver), 'build', str(RESULTS / f'{name}.mojo'),
                                '-o', binary], cwd=ROOT, capture_output=True, text=True, timeout=660)
        (RESULTS / f'{name}.build.log').write_text(build.stdout + build.stderr)
        report = dict(name=name, config=case['config'], build_returncode=build.returncode,
                      host_api_bridge='PooledRange bytes -> MappedDomain microthread count; device code unchanged',
                      host_link_addition='libm for the independent DFT reference')
        if build.returncode == 0:
            result = subprocess.run(['bash', str(driver), 'run', binary, '--config', case['config'],
                                     '--out', str(RESULTS / 'simulator'), '--name', name], cwd=ROOT,
                                     capture_output=True, text=True, timeout=case.get('run_timeout',300))
            output = result.stdout + result.stderr
            (RESULTS / f'{name}.run.log').write_text(output)
            report.update(run_returncode=result.returncode,
                          verified=result.returncode == 0 and 'verification passed' in output
                          and 'mismatch' not in output and 'failed, exit' not in output)
        else:
            report['verified'] = False
        (RESULTS / f'{name}.result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        failed |= not report['verified']
    return 1 if failed else 0


def summarize():
    # Docker can create deep result paths even on hosts with MAX_PATH enabled.
    root = Path('\\\\?\\' + str(RESULTS)) if os.name == 'nt' else RESULTS
    cases = json.loads((root / 'cases.json').read_text())
    rows = []
    for case in cases:
        name = case['name']
        report = json.loads((root / f'{name}.result.json').read_text())
        if not report['verified']:
            raise ValueError(f'{name} did not pass verification')
        files = sorted((root / 'simulator' / name).rglob('run.json'))
        data = json.loads(files[-1].read_text())
        log = (root / f'{name}.run.log').read_text()
        rows.append(dict(name=name, config=case['config'], verified=True,
                         config_md5=data['config_md5'], core_mhz=data['core_mhz'],
                         cycles=data['cycles'], time_us=data['time_us'],
                         host_launches=len(data['tasks']), phase_launches=data['launches'],
                         kernel_exec_cycles=sum(k['finished'] - k['started'] for k in data['kernels']),
                         spill_warnings=re.findall(r'^.*spills.*$', log, re.MULTILINE),
                         source_sha256=hashlib.sha256((root / f'{name}.mojo').read_bytes()).hexdigest(),
                         execution=case['execution'], run_log=f'{name}.run.log'))
    summary = dict(image='ghcr.io/psal-postech/mojo-m2ndp:main', image_id='b8155d935ba7',
                   scope='Functional execution/codegen validation; distinct workloads, no GPU comparison',
                   host_compatibility='MappedDomain bridge; temporary driver links libm',
                   passed=len(rows), cases=rows)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for row in rows:
        print(row['name'], row['cycles'], row['time_us'], row['host_launches'], row['phase_launches'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--run', nargs='*')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.generate:
        generate()
    if args.run is not None:
        sys.exit(run(args.run))
    if args.summarize:
        summarize()

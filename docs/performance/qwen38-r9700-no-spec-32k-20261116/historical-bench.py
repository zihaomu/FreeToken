import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
PLAN = json.loads((ROOT / "plan.json").read_text())
PHASE = sys.argv[1]
SPEC = False
DIRECTORY = ROOT / PHASE
DIRECTORY.mkdir(exist_ok=False)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def get(origin, route):
    with urllib.request.urlopen(origin + route, timeout=5) as response:
        return json.load(response)


import torch
import triton
from freetoken.utils.hf import load_tokenizer

torch.cuda.init()
libs = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
               if 'libamdhip64.so' in line and line.split()[-1].startswith('/')})
assert len(libs) == 1, libs
hip = ctypes.CDLL(libs[0])
bus = ctypes.create_string_buffer(64)
assert hip.hipDeviceGetPCIBusId(bus, 64, 0) == 0
assert bus.value.decode() == PLAN['device']['pci'], bus.value
save(ROOT / f'{PHASE}-runtime.json', dict(torch=torch.__version__, hip=torch.version.hip,
     triton=triton.__version__, pci=bus.value.decode(), name=torch.cuda.get_device_name(0),
     arch=torch.cuda.get_device_properties(0).gcnArchName,
     environment_report=os.environ.get('DEEPINFER_ENVIRONMENT_REPORT'),
     image=os.environ.get('DEEPINFER_IMAGE_ID'), hip_visible=os.environ.get('HIP_VISIBLE_DEVICES'),
     cpu_affinity=sorted(os.sched_getaffinity(0)), hip_library=libs[0]))

model = Path(PLAN['model']).resolve()
tokenizer = load_tokenizer(str(model))
prompts = {}
for context in PLAN['contexts']:
    path = WORKSPACE / '.freetoken-zihao/fp8-kv-qwen36-hybrid-20261104' / f'prompt-{context}.txt'
    prompts[context] = path.read_text()
    assert len(tokenizer.encode(prompts[context], add_special_tokens=False)) == context


def request(origin, context, repetition):
    body = dict(model='qwen38-nvfp4', prompt=prompts[context], max_tokens=PLAN['output_tokens'],
                temperature=0, ignore_eos=True, stream=True, stream_options=dict(include_usage=True))
    save(DIRECTORY / f'{context}-{repetition}-request.json', body)
    req = urllib.request.Request(origin + '/v1/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    events, stamps, pieces = [], [], []
    usage = timings = finish = None
    done = False
    offset = (DIRECTORY / 'server.log').stat().st_size
    start_unix_ns = time.time_ns()
    start = time.perf_counter()
    try:
        response = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode(errors="replace")}') from error
    with response:
        for line in response:
            if time.perf_counter() - start > 900:
                raise TimeoutError('request exceeded 900 seconds')
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                done = True
                break
            event = json.loads(data)
            now = time.perf_counter()
            events.append(dict(elapsed_s=now-start, data=event))
            if 'error' in event:
                raise RuntimeError(event['error'])
            usage = event.get('usage') or usage
            timings = event.get('timings') or timings
            for choice in event.get('choices', []):
                if choice.get('text'):
                    stamps.append(now)
                    pieces.append(choice['text'])
                if choice.get('finish_reason') is not None:
                    finish = now
    elapsed = time.perf_counter() - start
    save(DIRECTORY / f'{context}-{repetition}-events.json', events)
    assert done and usage and stamps and finish is not None, (done, usage, len(stamps), finish)
    assert usage['completion_tokens'] == PLAN['output_tokens'] and usage['prompt_tokens'] == context, usage
    assert (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0) == 0, usage
    admitted = (DIRECTORY / 'server.log').read_bytes()[offset:].decode(errors='replace')
    batches = [(int(n), int(c)) for n, c in re.findall(
        r'Prefill batch, #new-seq: \d+, #new-token: (\d+), #cached-token: (\d+)', admitted)]
    assert sum(n for n, c in batches) == context and all(c == 0 for n, c in batches), batches
    row = dict(context=context, repetition=repetition, usage=usage, timings=timings,
               request_start_unix_ns=start_unix_ns, request_end_unix_ns=time.time_ns(),
               request_start_perf_s=start, first_text_perf_s=stamps[0], terminal_perf_s=finish,
               ttft_s=stamps[0]-start, e2e_s=elapsed,
               decode_tok_s=(PLAN['output_tokens']-1)/(finish-stamps[0]),
               finish_s=finish-start, text_chunks=len(stamps), text=''.join(pieces),
               prefill_batches=batches,
               logged_accept_lengths=[float(x) for x in re.findall(r'accept len: ([\d.]+)', admitted)])
    row['text_sha256'] = hashlib.sha256(row['text'].encode()).hexdigest()
    save(DIRECTORY / f'{context}-{repetition}.json', row)
    print(json.dumps({k: row[k] for k in ['context', 'repetition', 'decode_tok_s', 'ttft_s', 'timings']}), flush=True)
    return row


with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
origin = f'http://127.0.0.1:{port}'
cmd = json.loads((ROOT/'historical-command.json').read_text())['argv']
cmd[0]=sys.executable
for flag,value in [('--model',str(model)),('--port',str(port)),('--max-seq-len-override','34816')]:
    cmd[cmd.index(flag)+1]=value
cmd += ['--num-tokens','35072']
assert not any('speculative' in arg or 'dflash' in arg.lower() for arg in cmd)
os.environ['FT_INVENTORY_DIR']=str(DIRECTORY)
os.environ['PYTHONPATH']=str(ROOT/'inventory_hook')+os.pathsep+os.environ['PYTHONPATH']
save(DIRECTORY / 'command.json', dict(argv=cmd, model=str(model), draft=None,
     source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
     environment={k: os.environ.get(k) for k in ['HIP_VISIBLE_DEVICES', 'PYTHONPATH', 'HF_HUB_OFFLINE',
         'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR', 'TVM_FFI_CACHE_DIR', 'OMP_NUM_THREADS']},
     model_config_sha256=hashlib.sha256((model/'config.json').read_bytes()).hexdigest()))
with (DIRECTORY / 'server.log').open('x') as out:
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print(f'START {PHASE} pid={proc.pid} port={port}', flush=True)
    try:
        start = time.monotonic()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f'server exited {proc.returncode}: {DIRECTORY}/server.log')
            if time.monotonic()-start > 1200:
                raise TimeoutError('server startup exceeded 1200 seconds')
            try:
                health = get(origin, '/health')
                if health.get('status') == 'error':
                    raise RuntimeError(health)
                if health.get('maintenance') == 'serving':
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        save(DIRECTORY / 'ready.json', dict(startup_s=time.monotonic()-start, health=health,
             stats=get(origin, '/v1/stats'), cache=get(origin, '/v1/cache/status')))
        summaries = []
        for context in PLAN['contexts']:
            request(origin, context, 'warmup')
            rows = [request(origin, context, f'measure-{i+1}') for i in range(PLAN['measured_repeats'])]
            summary = dict(phase=PHASE, context=context, output_tokens=PLAN['output_tokens'],
                           repeats=len(rows), decode_tok_s=statistics.median(r['decode_tok_s'] for r in rows),
                           ttft_s=statistics.median(r['ttft_s'] for r in rows),
                           e2e_s=statistics.median(r['e2e_s'] for r in rows),
                           text_hashes=[r['text_sha256'] for r in rows],
                           samples=[{k: r[k] for k in ['decode_tok_s', 'ttft_s', 'e2e_s', 'timings']} for r in rows])
            summaries.append(summary)
            save(DIRECTORY / 'summary.json', summaries)
            print('SUMMARY '+json.dumps(summary), flush=True)
    except BaseException:
        save(DIRECTORY / 'failure.json', dict(traceback=traceback.format_exc()))
        raise
    finally:
        for sig, timeout in [(signal.SIGTERM, 30), (signal.SIGKILL, 10)]:
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=timeout)
                break
            except subprocess.TimeoutExpired:
                continue

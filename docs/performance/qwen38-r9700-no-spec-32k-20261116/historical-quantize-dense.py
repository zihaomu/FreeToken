"""Local experimental W8A16 derivative; never mutate the locked upstream checkpoint."""
import gc,hashlib,json,os,statistics,struct,time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from freetoken.layers.quantization.names import name_set
from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

torch.set_num_threads(1);torch.manual_seed(42)
src=Path('/home/zihaomu/zihaomu_workspace/freetoken_workspace/models/Qwen3.8-Flash-Next-NVFP4-fc694b54');dst=Path('/home/zihaomu/zihaomu_workspace/freetoken_workspace/.freetoken-zihao/qwen38-dense-fp8-32k-20261116-0703/derived-dense-fp8');dst.mkdir(exist_ok=True)
assert not (dst/'conversion-manifest.json').exists()
index=json.loads((src/'model.safetensors.index.json').read_text())
suffixes=('.linear_attn.in_proj_qkv.weight','.linear_attn.in_proj_z.weight','.linear_attn.out_proj.weight','.self_attn.o_proj.weight')
selected={k:v for k,v in index['weight_map'].items() if k=='lm_head.weight' or k.startswith('model.language_model.layers.') and k.endswith(suffixes)}
modules=[k[:-7] for k in selected]
affected=set(selected.values());manifest={'source_revision':'fc694b54fb0174e0913e6adf86691ef85a4ead47','algorithm':'per-output-row absmax/448 E4M3, BF16 activations, no calibration','selected':{},'microbench':{},'input_hashes':{}}
def bench(fn):
 for _ in range(3):fn()
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=stream):
  outputs=[fn() for _ in range(16)]
 torch.cuda.current_stream().wait_stream(stream)
 samples=[]
 for _ in range(7):
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  a.record();g.replay();b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000/16)
 return {'median_us':statistics.median(samples),'rounds_us':samples}

for f in src.iterdir():
 if f.is_file() and f.name not in affected and f.name not in ('config.json','hf_quant_config.json','model.safetensors.index.json'):
  assert (dst/f.name).exists(), 'Prepare unchanged hardlinks on the host before conversion'
seen=set()
for fname in sorted(affected):
 with safe_open(src/fname,framework='pt',device='cpu') as f:
  tensors={k:f.get_tensor(k) for k in f.keys()};metadata=f.metadata()
 for name in sorted(set(tensors)&set(selected)):
  w=tensors[name].cuda();assert w.dtype==torch.bfloat16 and w.ndim==2
  wf=w.float();scale=(wf.abs().amax(dim=1)/448).clamp_min(1e-12)
  q=(wf/scale[:,None]).clamp(-448,448).to(torch.float8_e4m3fn)
  restored=q.float()*scale[:,None]
  err=float((restored-wf).square().sum().sqrt()/wf.square().sum().sqrt())
  manifest['selected'][name]={'shape':list(w.shape),'weight_relative_rmse':err,'before_bytes':w.numel()*2,'after_bytes':q.numel()+scale.numel()*4}
  tensors[name]=q.cpu();sname=name[:-7]+'.weight_scale';tensors[sname]=scale.cpu();index['weight_map'][sname]=fname
  kind='lm_head' if name=='lm_head.weight' else next(s for s in suffixes if name.endswith(s))
  if kind not in seen:
   checks=[]
   for m in (1,16):
    x=torch.randn(m,w.shape[1],device='cuda',dtype=torch.bfloat16)*.1
    y=fp8_pertensor_linear(x,q,scale);ref=x.float()@restored.t();bf=x.float()@wf.t()
    rms=float(ref.square().mean().sqrt())
    torch.testing.assert_close(y.float(),ref,rtol=.02,atol=.02*rms)
    checks.append({'m':m,'kernel_relative_rmse':float((y.float()-ref).norm()/ref.norm()),'vs_bf16_weight_relative_rmse':float((y.float()-bf).norm()/bf.norm())})
   x=torch.randn(1,w.shape[1],device='cuda',dtype=torch.bfloat16)*.1
   manifest['microbench'][kind]={'name':name,'checks':checks,'bf16':bench(lambda:torch.nn.functional.linear(x,w)),'fp8':bench(lambda:fp8_pertensor_linear(x,q,scale))}
   print('MICRO',kind,json.dumps(manifest['microbench'][kind]),flush=True);seen.add(kind)
  del w,wf,q,scale,restored
 print('WRITE',fname,flush=True);save_file(tensors,dst/fname,metadata=metadata);del tensors;gc.collect()
 (dst/'conversion-manifest.partial.json').write_text(json.dumps(manifest,indent=2)+'\n')
for fname,key in [('config.json','quantization_config'),('hf_quant_config.json','quantization')]:
 obj=json.loads((src/fname).read_text());qcfg=obj[key]
 ignore_key='ignore' if fname=='config.json' else 'exclude_modules'
 old=qcfg.get(ignore_key,[]);qcfg[ignore_key]=[pat for pat in old if not any(name_set((pat,))(m) for m in modules)]
 for m in modules:qcfg['quantized_layers'][m]={'quant_algo':'FP8_PER_CHANNEL_PER_TOKEN'}
 if 'config_groups' in qcfg:
  qcfg['config_groups']['local_dense_w8a16']={'input_activations':None,'weights':{'dynamic':False,'num_bits':8,'type':'float','strategy':'channel'},'targets':modules}
 (dst/fname).write_text(json.dumps(obj,indent=2)+'\n');manifest['input_hashes'][fname]=hashlib.sha256((src/fname).read_bytes()).hexdigest()
index['metadata']['total_size']-=sum(v['before_bytes']-v['after_bytes'] for v in manifest['selected'].values())
(dst/'model.safetensors.index.json').write_text(json.dumps(index,indent=2)+'\n')
manifest['input_hashes']['model.safetensors.index.json']=hashlib.sha256((src/'model.safetensors.index.json').read_bytes()).hexdigest()
manifest['total_saved_weight_bytes']=sum(v['before_bytes']-v['after_bytes'] for v in manifest['selected'].values())
(dst/'conversion-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('DONE',len(selected),manifest['total_saved_weight_bytes'],flush=True)

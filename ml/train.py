"""Reproducible occupancy model. NumPy + Pillow only. No pretrained weights.
Run with --cache /absolute/path outside the submission repo.
Fixed dataset revision; split by capture day supplied by the mirror, audited here.
"""
import argparse, concurrent.futures, hashlib, json, time, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image

REV = '577e9c942e596819798d42fdc4d14d0b6646a43f'
BASE = f'https://huggingface.co/datasets/dronefreak/PKLot/resolve/{REV}/data/images'
ROOT = Path(__file__).resolve().parents[1]
def fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as r: data = r.read()
                path.write_bytes(data); break
            except Exception:
                if attempt == 3: raise
                time.sleep(1+attempt)
    return path

def sample_patch(rgb, box):
    x,y,w,h=box; H,W=rgb.shape[:2]
    xs=np.clip(np.floor(x+(np.arange(32)+.5)*w/32).astype(int),0,W-1)
    ys=np.clip(np.floor(y+(np.arange(32)+.5)*h/32).astype(int),0,H-1)
    return rgb[ys[:,None],xs[None,:],:]

def features(patch):
    # Identical nearest-neighbour sampling / feature order in dist/vision.js.
    rgb=patch.astype(np.float64)/255
    gray=rgb[:,:,0]*.299+rgb[:,:,1]*.587+rgb[:,:,2]*.114
    gx=gray[1:-1,2:]-gray[1:-1,:-2]; gy=gray[2:,1:-1]-gray[:-2,1:-1]
    mag=np.sqrt(gx*gx+gy*gy)
    bins=np.floor((np.arctan2(gy,gx)+np.pi)%np.pi*9/np.pi).astype(int)%9
    hist=np.zeros((4,4,9))
    for yy in range(30):
        for xx in range(30): hist[(yy+1)//8,(xx+1)//8,bins[yy,xx]]+=mag[yy,xx]
    hist=hist/(np.sqrt((hist*hist).sum(axis=2,keepdims=True))+.01)
    cells=gray.reshape(4,8,4,8).transpose(0,2,1,3).reshape(16,64)
    colors=rgb.reshape(4,8,4,8,3).transpose(0,2,1,3,4).reshape(16,64,3).mean(axis=1)
    return np.concatenate([hist.ravel(),cells.mean(axis=1),cells.std(axis=1),colors.ravel()])

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-35,35)))
def metrics(y,p):
    pred=p>=.5; tp=int(((y==1)&pred).sum()); tn=int(((y==0)&~pred).sum())
    fp=int(((y==0)&pred).sum()); fn=int(((y==1)&~pred).sum())
    return dict(n=len(y),accuracy=float((pred==y).mean()),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn),tp=tp,tn=tn,fp=fp,fn=fn)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True); args=ap.parse_args()
    rng=np.random.default_rng(42); selected={}; all_days={}
    for split,n in [('train',24),('valid',6),('test',9)]:
        meta=fetch(f'{BASE}/{split}/metadata.jsonl',args.cache/f'{split}.jsonl')
        rows=[json.loads(x) for x in meta.read_text().splitlines()]
        all_days[split]={r['file_name'].rsplit('_',3)[0] for r in rows}
        selected[split]=[]
        for cam in ['PUCPR','UFPR04','UFPR05']:
            group=sorted([r for r in rows if r['file_name'].startswith(cam+'_')],key=lambda r:r['file_name'])
            # Fixed seed sampling, independent of labels. Every camera represented.
            ids=sorted(rng.choice(len(group),min(n,len(group)),replace=False))
            selected[split].extend(group[i] for i in ids)
    assert not(all_days['train']&all_days['valid'] or all_days['train']&all_days['test'] or all_days['valid']&all_days['test'])
    jobs=[(split,r) for split,rows in selected.items() for r in rows]
    def download(job):
        split,r=job;name=r['file_name'];return fetch(f'{BASE}/{split}/{name}',args.cache/split/name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(download,jobs))
    print('Downloaded',len(jobs),'frames; capture-day splits audited',flush=True)
    sets={}; manifests=[]; start=time.perf_counter()
    for split,rows in selected.items():
        X=[];Y=[];cams=[]
        for r in rows:
            path=args.cache/split/r['file_name'];rgb=np.asarray(Image.open(path).convert('RGB'))
            manifests.append(dict(split=split,file=r['file_name'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source=f'{BASE}/{split}/{r["file_name"]}'))
            for box,label in zip(r['objects']['bbox'],r['objects']['categories']):
                X.append(features(sample_patch(rgb,box)));Y.append(label);cams.append(r['file_name'].split('_')[0])
        sets[split]=(np.asarray(X),np.asarray(Y),np.asarray(cams))
    print('Features', {s:len(v[1]) for s,v in sets.items()},'seconds',round(time.perf_counter()-start,2),flush=True)
    X,y,_=sets['train'];mean=X.mean(0);std=np.maximum(X.std(0),.03)
    sets={s:((v[0]-mean)/std,v[1],v[2]) for s,v in sets.items()};X,y,_=sets['train'];vx,vy,_=sets['valid']
    # One hidden layer; Adam, L2; early model selection uses validation only.
    params=[rng.normal(0,np.sqrt(2/X.shape[1]),(X.shape[1],48)),np.zeros(48),rng.normal(0,.1,48),np.zeros(1)]
    moments=[np.zeros_like(p) for p in params];variances=[np.zeros_like(p) for p in params];step=0;best=None;best_loss=1e9
    for epoch in range(160):
        for idx in np.array_split(rng.permutation(len(X)),max(1,len(X)//128)):
            xx=X[idx];yy=y[idx];w,b,v,c=params;h=np.maximum(0,xx@w+b);p=sigmoid(h@v+c)
            d=(p-yy)/len(idx);dh=d[:,None]*v[None,:]*(h>0)
            grads=[xx.T@dh+.0004*w,dh.sum(0),h.T@d+.0004*v,np.array([d.sum()])]
            step+=1
            for k,g in enumerate(grads):
                moments[k]=.9*moments[k]+.1*g;variances[k]=.999*variances[k]+.001*g*g
                params[k]-=.0015*(moments[k]/(1-.9**step))/(np.sqrt(variances[k]/(1-.999**step))+1e-8)
        w,b,v,c=params;p=sigmoid(np.maximum(0,vx@w+b)@v+c)
        loss=float(-(vy*np.log(p+1e-9)+(1-vy)*np.log(1-p+1e-9)).mean())
        if loss<best_loss:best_loss=loss;best=[p.copy() for p in params];best_epoch=epoch+1
    w,b,v,c=best
    def predict(x):return sigmoid(np.maximum(0,x@w+b)@v+c)
    # Temperature selected on validation labels; confidence remains uncalibrated OOD.
    logits=np.maximum(0,vx@w+b)@v+c
    temperatures=np.linspace(.7,2.5,19)
    temp=float(min(temperatures,key=lambda t:float(-(vy*np.log(sigmoid(logits/t)+1e-9)+(1-vy)*np.log(1-sigmoid(logits/t)+1e-9)).mean())))
    report=dict(seed=42,dataset_revision=REV,split='mirror camera-day disjoint; same cameras across splits',frames={s:len(r) for s,r in selected.items()},best_epoch=best_epoch,temperature=temp,conditions='Weather metadata unavailable in mirror; no local or night validation',metrics={},camera_metrics={})
    for s,(xx,yy,cc) in sets.items():
        p=sigmoid((np.maximum(0,xx@w+b)@v+c)/temp);accepted=(p<=.2)|(p>=.8)
        report['metrics'][s]={**metrics(yy,p), 'coverage_20_80':float(accepted.mean()),'accepted_accuracy_20_80':float(((p[accepted]>=.5)==yy[accepted]).mean())}
        if s=='test':report['camera_metrics']={cam:metrics(yy[cc==cam],p[cc==cam]) for cam in sorted(set(cc))}
    majority=int(y.mean()>=.5);ty=sets['test'][1];report['majority_baseline_test_accuracy']=float((ty==majority).mean())
    model=dict(version='baywatch-hog-mlp-v1',feature_size=len(mean),hidden=48,mean=mean.tolist(),std=std.tolist(),w=w.tolist(),b=b.tolist(),v=v.tolist(),c=float(c[0]),temperature=temp,thresholds=[.2,.8],seed=42,dataset_revision=REV)
    (ROOT/'dist/models/occupancy.json').write_text(json.dumps(model,separators=(',',':')))
    (ROOT/'dist/data/evaluation.json').write_text(json.dumps(report,indent=2))
    (ROOT/'docs/dataset-manifest.json').write_text(json.dumps(manifests,indent=2))
    # Demo frames chosen by fixed positions in test selection, never by performance.
    demos=[]
    for k,r in enumerate([selected['test'][4],selected['test'][13],selected['test'][22]]):
        original=args.cache/'test'/r['file_name'];rgb=np.asarray(Image.open(original).convert('RGB'))
        # Public demo at original resolution; aerial image has no readable identifying details.
        dest=ROOT/f'dist/assets/parking-{k+1}.jpg';dest.write_bytes(original.read_bytes())
        zones=[dict(id=f'P{i+1:02}',name=f'Место {i+1:02}',box=box,truth=int(lab),kind='parking') for i,(box,lab) in enumerate(zip(r['objects']['bbox'],r['objects']['categories']))]
        demos.append(dict(id=k+1,name=r['file_name'].split('_')[0],file=f'assets/parking-{k+1}.jpg',original=r['file_name'],width=rgb.shape[1],height=rgb.shape[0],zones=zones))
    (ROOT/'dist/data/demos.json').write_text(json.dumps(demos,ensure_ascii=False))
    # Cross-runtime parity fixture: raw sampled RGB and expected features/prediction.
    patch=sample_patch(rgb,zones[0]['box']);feat=features(patch);px=(feat-mean)/std
    fixture=dict(patch=patch.ravel().tolist(),features=feat.tolist(),probability=float(sigmoid((np.maximum(0,px@w+b)@v+c)/temp)[0]))
    (ROOT/'tests/parity.json').write_text(json.dumps(fixture))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()

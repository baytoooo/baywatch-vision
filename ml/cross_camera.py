"""Leave-one-camera-out stress audit. Never changes deployed model.
Same fixed frame subset, feature extractor and architecture. Each fold excludes
held-out camera from both training and validation; tested on its test days.
UFPR04 and UFPR05 view the same physical lot, so this is NOT leave-one-site-out.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from train import ROOT,features,sample_patch,sigmoid,metrics
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);args=p.parse_args()
manifest=json.loads((ROOT/'docs/dataset-manifest.json').read_text());sets={}
for split in ['train','valid','test']:
    rows={r['file_name']:r for r in map(json.loads,(args.cache/f'{split}.jsonl').read_text().splitlines())};X=[];Y=[];C=[]
    for row in manifest:
        if row['split']!=split:continue
        r=rows[row['file']];rgb=np.asarray(Image.open(args.cache/split/row['file']).convert('RGB'))
        for box,y in zip(r['objects']['bbox'],r['objects']['categories']):X.append(features(sample_patch(rgb,box)));Y.append(y);C.append(row['file'].split('_')[0])
    sets[split]=np.asarray(X),np.asarray(Y),np.asarray(C)
folds={}
for cam in ['PUCPR','UFPR04','UFPR05']:
    rng=np.random.default_rng(42);X,y,c=sets['train'];X,y=X[c!=cam],y[c!=cam];mean=X.mean(0);std=np.maximum(X.std(0),.03);X=(X-mean)/std
    vx,vy,vc=sets['valid'];vx,vy=(vx[vc!=cam]-mean)/std,vy[vc!=cam]
    tx,ty,tc=sets['test'];tx,ty=(tx[tc==cam]-mean)/std,ty[tc==cam]
    params=[rng.normal(0,np.sqrt(2/X.shape[1]),(X.shape[1],48)),np.zeros(48),rng.normal(0,.1,48),np.zeros(1)]
    mm=[np.zeros_like(p) for p in params];vv=[np.zeros_like(p) for p in params];step=0;bestloss=1e9
    for epoch in range(160):
        for idx in np.array_split(rng.permutation(len(X)),max(1,len(X)//128)):
            xx=X[idx];yy=y[idx];w,b,v,c=params;h=np.maximum(0,xx@w+b);p=sigmoid(h@v+c);d=(p-yy)/len(idx);dh=d[:,None]*v[None,:]*(h>0)
            grads=[xx.T@dh+.0004*w,dh.sum(0),h.T@d+.0004*v,np.array([d.sum()])];step+=1
            for k,g in enumerate(grads):mm[k]=.9*mm[k]+.1*g;vv[k]=.999*vv[k]+.001*g*g;params[k]-=.0015*(mm[k]/(1-.9**step))/(np.sqrt(vv[k]/(1-.999**step))+1e-8)
        w,b,v,c=params;p=sigmoid(np.maximum(0,vx@w+b)@v+c);loss=float(-(vy*np.log(p+1e-9)+(1-vy)*np.log(1-p+1e-9)).mean())
        if loss<bestloss:bestloss=loss;best=[p.copy() for p in params];best_epoch=epoch+1
    w,b,v,c=best;p=sigmoid(np.maximum(0,tx@w+b)@v+c);folds[cam]={**metrics(ty,p),'best_epoch':best_epoch,'train_samples':len(y),'validation_samples':len(vy)}
result=dict(protocol='Leave-one-camera-out; camera excluded from train AND validation. Same-lot overlap between UFPR04/UFPR05. No deployment model change. Threshold 0.5, no quality gate or rejection.',folds=folds,seed=42)
(ROOT/'dist/data/cross-camera.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

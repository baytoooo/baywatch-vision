// TensorFlow COCO-SSD adapter. Locally hosted Apache-2.0 runtime and weights.
export const VEHICLES=new Set(['car','truck','bus','motorcycle','bicycle']);
export const VEHICLE_NAMES={car:'Авто',truck:'Грузовой',bus:'Автобус',motorcycle:'Мото',bicycle:'Велосипед'};
export function iou(a,b){const [ax,ay,aw,ah]=a,[bx,by,bw,bh]=b;const overlap=Math.max(0,Math.min(ax+aw,bx+bw)-Math.max(ax,bx))*Math.max(0,Math.min(ay+ah,by+bh)-Math.max(ay,by));return overlap/Math.max(1e-9,aw*ah+bw*bh-overlap);}
export function nms(boxes,threshold=.45){const kept=[];for(const p of [...boxes].sort((a,b)=>b.score-a.score)){if(!kept.some(k=>iou(k.bbox,p.bbox)>threshold))kept.push(p);}return kept;}
export function tileRegions(width,height){const w=Math.ceil(width*.6),h=Math.ceil(height*.6);return [[0,0,w,h],[width-w,0,w,h],[0,height-h,w,h],[width-w,height-h,w,h]];}
export class Tracker{
 constructor(){this.reset();}reset(){this.next=1;this.tracks=[];}
 update(detections,now){const old=this.tracks.filter(t=>now-t.last<1500),used=new Set();const tracks=detections.map(d=>{let best=-1,overlap=.2;old.forEach((t,i)=>{const v=iou(t.bbox,d.bbox);if(!used.has(i)&&v>overlap){best=i;overlap=v;}});const id=best<0?this.next++:old[best].id;if(best>=0)used.add(best);return {...d,id,last:now};});this.tracks=[...tracks,...old.filter((_,i)=>!used.has(i))];return tracks;}
}
function script(src,globalName){if(window[globalName])return Promise.resolve();return new Promise((resolve,reject)=>{const el=document.createElement('script');el.src=src;el.onload=()=>window[globalName]?resolve():reject(Error('Библиотека не загрузилась'));el.onerror=()=>{el.remove();reject(Error('Не удалось загрузить '+src));};document.head.append(el);});}
export class VehicleDetector{
 constructor(){this.model=null;this.pending=null;}
 async load(onStatus=()=>{}){if(this.model)return this.model;if(this.pending)return this.pending;this.pending=(async()=>{onStatus('Загрузка детектора · 18 МБ');await script('vendor/tf.min.js','tf');await script('vendor/coco-ssd.min.js','cocoSsd');const tf=window.tf;try{await tf.setBackend('webgl');await tf.ready();}catch{await tf.setBackend('cpu');await tf.ready();}onStatus('Подготовка детектора…');try{this.model=await window.cocoSsd.load({base:'lite_mobilenet_v2',modelUrl:'models/coco/model.json'});}catch(e){if(tf.getBackend()==='cpu')throw e;await tf.setBackend('cpu');this.model=await window.cocoSsd.load({base:'lite_mobilenet_v2',modelUrl:'models/coco/model.json'});}onStatus('Детектор готов · '+tf.getBackend().toUpperCase());return this.model;})().catch(e=>{this.pending=null;throw e;});return this.pending;}
 async detect(frame,threshold=.4,detail=true,isCurrent=()=>true){await this.load();if(!isCurrent())return[];const w=frame.width,h=frame.height;let detections=(await this.model.detect(frame,100,threshold)).filter(p=>VEHICLES.has(p.class));if(detail&&isCurrent()){
 const tile=document.createElement('canvas'),c=tile.getContext('2d');
 for(const [x,y,tw,th]of tileRegions(w,h)){if(!isCurrent())return[];tile.width=tw;tile.height=th;c.drawImage(frame,x,y,tw,th,0,0,tw,th);const pred=await this.model.detect(tile,100,threshold);detections.push(...pred.filter(p=>VEHICLES.has(p.class)).map(p=>({...p,bbox:[p.bbox[0]+x,p.bbox[1]+y,p.bbox[2],p.bbox[3]]})));}}
 return nms(detections).filter(p=>p.bbox[2]>=8&&p.bbox[3]>=8);}
}

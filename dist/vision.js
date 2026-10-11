// Original BayWatch feature extractor and inference. Pure JS, no remote calls.
export function samplePatch(rgba, width, height, box) {
  const [x,y,w,h]=box, out=new Uint8Array(32*32*3);
  for(let yy=0;yy<32;yy++)for(let xx=0;xx<32;xx++){
    const sx=Math.max(0,Math.min(width-1,Math.floor(x+(xx+.5)*w/32)));
    const sy=Math.max(0,Math.min(height-1,Math.floor(y+(yy+.5)*h/32)));
    const i=(sy*width+sx)*4,j=(yy*32+xx)*3;
    out[j]=rgba[i];out[j+1]=rgba[i+1];out[j+2]=rgba[i+2];
  }return out;
}
export function features(patch){
  const gray=new Float64Array(1024),hist=new Float64Array(144),means=new Float64Array(16),sq=new Float64Array(16),colors=new Float64Array(48);
  for(let y=0;y<32;y++)for(let x=0;x<32;x++){
    const i=y*32+x,j=i*3,c=Math.floor(y/8)*4+Math.floor(x/8);
    const r=patch[j]/255,g=patch[j+1]/255,b=patch[j+2]/255;
    const v=.299*r+.587*g+.114*b;gray[i]=v;means[c]+=v/64;sq[c]+=v*v/64;
    colors[c*3]+=r/64;colors[c*3+1]+=g/64;colors[c*3+2]+=b/64;
  }
  for(let y=1;y<31;y++)for(let x=1;x<31;x++){
    const i=y*32+x,gx=gray[i+1]-gray[i-1],gy=gray[i+32]-gray[i-32];
    const angle=(Math.atan2(gy,gx)+Math.PI)%Math.PI,bin=Math.floor(angle*9/Math.PI)%9;
    hist[(Math.floor(y/8)*4+Math.floor(x/8))*9+bin]+=Math.hypot(gx,gy);
  }
  for(let c=0;c<16;c++){
    let norm=0;for(let k=0;k<9;k++)norm+=hist[c*9+k]**2;norm=Math.sqrt(norm)+.01;
    for(let k=0;k<9;k++)hist[c*9+k]/=norm;
  }
  return Float64Array.from([...hist,...means,...Array.from(sq,(v,i)=>Math.sqrt(Math.max(0,v-means[i]**2))),...colors]);
}
export function predict(feat,m){
  const h=Float64Array.from(m.b);
  for(let i=0;i<feat.length;i++){const f=(feat[i]-m.mean[i])/m.std[i];for(let k=0;k<h.length;k++)h[k]+=f*m.w[i][k];}
  let z=m.c;for(let k=0;k<h.length;k++)z+=Math.max(0,h[k])*m.v[k];
  return 1/(1+Math.exp(-Math.max(-35,Math.min(35,z/m.temperature))));
}
export function quality(patch){
  let mean=0,bright=0,dark=0,edge=0;const g=[];
  for(let i=0;i<1024;i++){const v=(.299*patch[i*3]+.587*patch[i*3+1]+.114*patch[i*3+2])/255;g.push(v);mean+=v/1024;if(v>.97)bright++;if(v<.035)dark++;}
  for(let y=1;y<31;y++)for(let x=1;x<31;x++){const i=y*32+x;edge+=Math.abs(g[i]*4-g[i-1]-g[i+1]-g[i-32]-g[i+32])/900;}
  const reasons=[];if(mean<.09||dark/1024>.72)reasons.push('Слишком темно');if(mean>.94||bright/1024>.72)reasons.push('Пересвет');if(edge<.008)reasons.push('Мало деталей');
  return {ok:reasons.length===0,reasons,mean,edge};
}
export function classify(p,q,box,threshold=.8){
  if(box[2]<16||box[3]<16)return 'review';
  if(!q.ok)return 'review';
  if(p>=threshold)return 'occupied';if(p<=1-threshold)return 'free';return 'review';
}
export class Stabilizer{
  constructor(required=3){this.required=required;this.states=new Map();}
  update(id,raw,now){
    let s=this.states.get(id)||{state:'review',candidate:null,count:0,since:now};
    if(raw==='review'){s={state:'review',candidate:null,count:0,since:now};}
    else{if(s.candidate===raw)s.count++;else{s.candidate=raw;s.count=1;}if(s.count>=this.required&&s.state!==raw){s.state=raw;s.since=now;}}
    this.states.set(id,s);return {...s};
  }
  clear(){this.states.clear();}
}
export function actionFor(zone,state,elapsedMs,limitMinutes){
  if(state==='review')return {priority:'review',text:'Проверить кадр и положение камеры'};
  if(state==='occupied'&&zone.kind==='restricted')return {priority:'high',text:'Проверить перекрытие проезда'};
  if(state==='occupied'&&zone.kind==='charging'&&elapsedMs>=limitMinutes*60000)return {priority:'high',text:'Проверить длительную стоянку у зарядки'};
  return null;
}
export function roi({events,minutes,hourly,cost,setup}){
  const saved=events*minutes/60*hourly*26,net=saved-cost;
  return {saved,net,payback:net>0?setup/net:null};
}

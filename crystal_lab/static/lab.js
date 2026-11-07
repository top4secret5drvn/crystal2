import * as THREE from './vendor/three.module.js';
const $=id=>document.getElementById(id), token=document.querySelector('meta[name="lab-token"]').content;
let currentTab='memory', active=0, cy=null, selectedNode=null, inspectedInstance=null, examples=[], renderer=null, spaceCy=null, disposeSpace=null;
const pretty=x=>JSON.stringify(x,null,2);
function fail(e){$('error').hidden=false;$('error').textContent=e.message||String(e);}
async function api(path,data){
  if(data)data={...data,_instance:Number($('instance').value)};
  const url=new URL('/api/'+path,location.origin);
  if(!data&&$('instance').value)url.searchParams.set('instance',$('instance').value);
  const response=await fetch(url,{method:data?'POST':'GET',headers:data?{'Content-Type':'application/json','X-Lab-Token':token}:{},body:data?JSON.stringify(data):undefined});
  const result=await response.json();if(!response.ok)throw new Error(result.error||response.statusText);return result;
}
function bind(id,fn){$(id).addEventListener('click',()=>fn().catch(fail));}
function activeOnly(){if(Number($('instance').value)!==active)throw new Error('Сначала сделайте просматриваемый экземпляр активным.');}
async function refresh(){
  $('error').hidden=true;
  const instances=await api('instances');const selected=$('instance').value;
  $('instance').replaceChildren(...instances.map(i=>new Option(`#${i.id+1} · ${i.dormant?'спящий':'бодрствует'}${i.active?' · рабочий':''}`,i.id)));
  active=instances.find(i=>i.active)?.id||0;
  $('instance').value=instances.some(i=>String(i.id)===selected)?selected:String(active);
  const s=await api('status');active=s.active;
  for(const key of ['nodes','relations','hypotheses','tick'])$(key).textContent=s[key].toLocaleString('ru');
  $('context').textContent=s.context;$('connection').textContent=`● Ядро подключено · рабочий #${active+1}${s.dormant?' · спящий':''}`;
  $('commands').textContent=s.commands.join(' · ')+' · думай: до 10 тактов; сон: до 500 узлов. Файлы — через панель снапшотов.';
  const files=await api('snapshots');$('snapshot').replaceChildren(...files.map(n=>new Option(n,n)));
  const list=$('population-list');list.replaceChildren();
  for(const i of instances){const c=document.createElement('div');c.className='card'+(i.active?' active':'');c.textContent=`#${i.id+1} · ${i.active?'РАБОЧИЙ · ':''}${i.dormant?'спящий':'бодрствует'} · ${i.nodes} узлов · такт ${i.tick}`;list.append(c);}
  const history=await api('history');$('history-list').replaceChildren();
  for(const event of history.slice().reverse()){const d=document.createElement('details');d.className='card';const summary=document.createElement('summary');summary.textContent=`${new Date(event.time*1000).toLocaleTimeString()} · ${event.action}${event.error?' · ОШИБКА':''}`;const pre=document.createElement('pre');pre.textContent=pretty(event);d.append(summary,pre);$('history-list').append(d);}
  $('transfers').textContent=pretty(history.flatMap(e=>e.transfer_events||[]));
  if(currentTab==='memory'){
    await loadGraph();
    if(selectedNode!==null&&inspectedInstance===$('instance').value){
      try{const d=await api('nodes/'+selectedNode);$('node-title').textContent=d.label;$('details').textContent=pretty(d);}
      catch{selectedNode=null;$('details').textContent='Узел удалён или недоступен.';$('neighbors').hidden=true;}
    }else{selectedNode=null;$('node-title').textContent='Выберите узел';$('details').textContent='Нет выбранного объекта';$('neighbors').hidden=true;}
  }
  if(currentTab==='geometry')await geometry();
  if(currentTab==='training')await training();
}
async function loadGraph(neighbor=null){
  const params=new URLSearchParams({q:$('search').value,type:$('edge-type').value,scope:$('scope').value,services:$('services').checked,limit:150});
  if(neighbor!==null)params.set('node',neighbor);
  const g=await api('graph?'+params);
  if(cy)cy.destroy();
  cy=cytoscape({container:$('graph'),maxZoom:1.6,minZoom:0.1,elements:[...g.nodes.map(n=>({data:n,position:{x:(Number(n.id)%12)*120,y:Math.floor(Number(n.id)/12)*120}})),...g.edges.map(e=>({data:e}))],
    style:[{selector:'node',style:{'label':'data(label)','color':'#bed2df','font-size':11,'text-valign':'bottom','text-margin-y':7,'background-color':'#567789','border-width':2,'border-color':'#67828e','width':22,'height':22}},
    {selector:'node[energy > 15]',style:{'background-color':'#67dbca','border-color':'#b9fff0','width':29,'height':29}},
    {selector:'node[?hypothesis]',style:{'border-style':'dashed','border-color':'#eead69'}},
    {selector:'node[status = "confirmed"]',style:{'border-color':'#95ebbd','border-width':4}},
    {selector:'edge',style:{'width':1.2,'line-color':'#354b60','target-arrow-color':'#354b60','target-arrow-shape':'triangle','curve-style':'bezier','label':'data(type)','font-size':8,'color':'#70899c','text-rotation':'autorotate','line-style':'dashed'}},
    {selector:'edge[?confirmed]',style:{'line-color':'#58918d','target-arrow-color':'#58918d','line-style':'solid'}},
    {selector:':selected',style:{'border-color':'#fff','line-color':'#fff','target-arrow-color':'#fff'}}],
    layout:{name:'cose',animate:false,randomize:false,nodeRepulsion:14000,idealEdgeLength:105,nodeDimensionsIncludeLabels:true,padding:45}});
  cy.on('tap','node',async e=>{try{selectedNode=e.target.id();inspectedInstance=$('instance').value;const detail=await api('nodes/'+selectedNode);$('node-title').textContent=detail.label;$('details').textContent=pretty(detail);$('neighbors').hidden=false;}catch(err){fail(err);}});
  cy.on('tap','edge',e=>{$('node-title').textContent=e.target.data('type');$('details').textContent=pretty(e.target.data());$('neighbors').hidden=true;});
  $('graph-info').textContent=`${g.nodes.length} / ${g.total_nodes} узлов · ${g.edges.length} связей${g.partial?' · ограниченная выборка':''} · топологическая раскладка`;
}
async function training(){
  const data=await api('training');examples=data.examples;
  const previous=$('example').value;$('example').replaceChildren(...examples.filter(e=>e.split==='train').map(e=>new Option(`${e.id} · ${e.query}`,e.id)));
  if(examples.some(e=>e.id===previous))$('example').value=previous;
  selectExample();$('training-report').textContent=pretty(data.state||{status:'Нет сохранённых параметров'});
}
function selectExample(){const e=examples.find(e=>e.id===$('example').value);$('gold').textContent=pretty(e?.relations||[]);$('reference').value=e?.references[0]||'';}
async function geometry(){
  const query=$('world').value?'?world='+encodeURIComponent($('world').value):'';
  const data=await api('geometry'+query);$('world').replaceChildren(...data.worlds.map(w=>new Option(w,w)));if(data.world)$('world').value=data.world;
  $('coordinates').textContent=pretty(data.senses);
  if(disposeSpace){disposeSpace();disposeSpace=null;}if(renderer){renderer.dispose();renderer=null;}if(spaceCy){spaceCy.destroy();spaceCy=null;}$('space').replaceChildren();
  const points=data.senses.filter(s=>s.coordinates!==null);
  const mode=$('geometry-view').value;
  const projection=data.dimensions===8?(mode==='2'?'8D → проекция осей 1, 2':'8D → проекция осей 1, 2, 3'):`${data.dimensions||'—'}D исходные позиции`;
  $('geometry-info').textContent=`${projection} · обучено ${points.length} · без актуальной позиции ${data.untrained}`;
  if(!points.length){$('space').textContent='В этом экземпляре нет обученных позиций. Координаты не создаются визуализатором.';return;}
  if(mode==='2'){
    $('geometry-info').textContent+=data.dimensions!==2?' · показана проекция XY':'';
    spaceCy=cytoscape({container:$('space'),elements:points.map(s=>({data:{id:s.id,label:s.label+' / '+s.meaning},position:{x:s.coordinates[0]*180,y:s.coordinates[1]*180}})),style:[{selector:'node',style:{label:'data(label)','font-size':11,color:'#cedfe9','background-color':'#67dbca',width:10,height:10,'text-margin-y':8,'text-valign':'bottom'}}],layout:{name:'preset',fit:true,padding:70}});return;
  }
  if(data.dimensions===2){$('space').textContent='Исходная модель 2D: третьей координаты нет. Выберите плоскость XY.';return;}
  if(mode==='8'&&data.dimensions!==8){$('space').textContent='В памяти нет 8D-модели. Размерность не преобразуется визуализатором.';return;}
  const width=$('space').clientWidth,height=440;
  try{renderer=new THREE.WebGLRenderer({antialias:true});}catch(e){$('space').textContent='WebGL недоступен. Исходные координаты приведены ниже.';return;}
  renderer.setSize(width,height);renderer.setClearColor(0x0d1721);$('space').append(renderer.domElement);
  const scene=new THREE.Scene(),group=new THREE.Group();scene.add(group);
  const buffer=new THREE.BufferGeometry();buffer.setAttribute('position',new THREE.Float32BufferAttribute(points.flatMap(s=>s.coordinates.slice(0,3)),3));
  buffer.computeBoundingSphere();const radius=Math.max(buffer.boundingSphere.radius,0.5),center=buffer.boundingSphere.center;
  group.add(new THREE.Points(buffer,new THREE.PointsMaterial({color:0x67dbca,size:radius/30})));buffer.translate(-center.x,-center.y,-center.z);
  scene.add(new THREE.AxesHelper(radius));const camera=new THREE.PerspectiveCamera(45,width/height,.001,1000);camera.position.set(0,0,radius*3+1);camera.lookAt(0,0,0);
  disposeSpace=()=>{scene.traverse(obj=>{obj.geometry?.dispose();if(Array.isArray(obj.material))obj.material.forEach(m=>m.dispose());else obj.material?.dispose();});renderer?.forceContextLoss();};
  const draw=()=>renderer?.render(scene,camera);draw();let dragging=false,lastX=0,lastY=0;
  renderer.domElement.onpointerdown=e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;e.target.setPointerCapture(e.pointerId);};
  renderer.domElement.onpointerup=()=>dragging=false;
  renderer.domElement.onpointermove=e=>{if(!dragging)return;group.rotation.y+=(e.clientX-lastX)/150;group.rotation.x+=(e.clientY-lastY)/150;lastX=e.clientX;lastY=e.clientY;draw();};
  renderer.domElement.onwheel=e=>{e.preventDefault();camera.position.z=Math.max(.1,Math.min(100,camera.position.z*(e.deltaY>0?1.1:.9)));draw();};
}
for(const button of document.querySelectorAll('.tab'))button.onclick=async()=>{currentTab=button.dataset.tab;for(const v of document.querySelectorAll('.view'))v.hidden=v.id!==currentTab;for(const b of document.querySelectorAll('.tab'))b.classList.toggle('active',b===button);$('title').textContent={memory:'Карта памяти',geometry:'Семантические пространства',population:'Популяция кристаллов',training:'Имитационное обучение',history:'История изменений'}[currentTab];try{await refresh();}catch(e){fail(e);}};
bind('refresh',refresh);bind('graph-load',()=>loadGraph());bind('graph-reset',async()=>{$('search').value='';$('edge-type').value='';$('scope').value='all';await loadGraph();});bind('neighbors',async()=>{$('search').value='';await loadGraph(selectedNode);});
$('instance').onchange=()=>refresh().catch(fail);
bind('activate',async()=>{await api('instances/select',{id:Number($('instance').value)});await refresh();});
bind('snapshot-load',async()=>{if(!$('snapshot').value)throw new Error('Поместите .cry в рабочий каталог.');const r=await api('snapshots/load',{name:$('snapshot').value});$('instance').value=String(r.instance);await refresh();$('instance').value=String(r.instance);await refresh();});
bind('snapshot-save',async()=>{activeOnly();await api('snapshots/save',{name:$('save-name').value,confirm_overwrite:$('overwrite').checked});await refresh();});
bind('examples-import',async()=>{activeOnly();await api('training/examples',{examples:JSON.parse($('examples-json').value)});await training();});
$('example').onchange=selectExample;
bind('train-run',async()=>{activeOnly();const r=await api('training/run',{id:$('example').value,reference:$('reference').value});await refresh();$('training-report').textContent=pretty(r);});
bind('train-rollback',async()=>{activeOnly();await api('training/rollback',{});await refresh();});
bind('geometry-load',geometry);
$('command-form').onsubmit=async e=>{e.preventDefault();const button=e.target.querySelector('button');button.disabled=true;try{activeOnly();const text=$('command').value;const r=await api('commands',{command:text});$('console-output').textContent+=`\n\n› ${text}\n${r.output}`;$('console-output').textContent=$('console-output').textContent.slice(-40000);$('command').value='';await refresh();}catch(err){fail(err);}finally{button.disabled=false;}};
refresh().catch(fail);

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
  for(const key of ['nodes','relations','hypotheses','tick'])$(key).textContent=(key==='hypotheses'?(s.hypotheses_total??s[key]):s[key]).toLocaleString('ru');
  $('context').textContent=s.context;$('connection').textContent=`● Ядро подключено · рабочий #${active+1}${s.dormant?' · спящий':''}`;
  $('commands').textContent=s.commands.join(' · ')+' · думай: до 10 тактов; сон: до 500 узлов. Файлы — через панель снапшотов.';
  const files=await api('snapshots');const selectedFile=$('snapshot').value;$('snapshot').replaceChildren(...files.map(n=>new Option(n,n)));if(files.includes(selectedFile))$('snapshot').value=selectedFile;
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
  if(currentTab==='imitation')await training();
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
for(const button of document.querySelectorAll('.tab'))button.onclick=async()=>{currentTab=button.dataset.tab;for(const v of document.querySelectorAll('.view'))v.hidden=v.id!==currentTab;for(const b of document.querySelectorAll('.tab'))b.classList.toggle('active',b===button);$('title').textContent={memory:'Карта памяти',geometry:'Семантические пространства',population:'Популяция кристаллов',imitation:'Обучение через подражание',study:'Автономное изучение',history:'История изменений'}[currentTab];try{await refresh();}catch(e){fail(e);}};
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

// Each message describes a finished core step on an isolated copy. No synthetic timing.
const liveJobs={};const jobGraphs={};const lastJobTrace={};const lastJobFrame={};
const stageNames={started:'Операция начата · временная копия',fetching:'Получение сведений',material:'Материал получен',observation:'Текстовое наблюдение',tick:'Такт мышления',sleep:'Сон',snapshot:'Временный снапшот',research_complete:'Исследовательский цикл завершён',analysis:'Разбор вопроса и эталона',initial_attempt:'Исходная попытка',candidate:'Проверен кандидат',round_complete:'Раунд завершён',completed:'Операция завершена',failed:'Ошибка — рабочая память сохранена',cancelled:'Остановлено — рабочая память сохранена'};
function drawJob(kind,graph,path=[],relation=[]){
  const container=$(kind+'-graph');if(!graph||!container)return;
  let view=jobGraphs[kind];
  if(!view){view=cytoscape({container,maxZoom:1.5,elements:[],style:[
    {selector:'node',style:{label:'data(label)','font-size':10,color:'#bdcedb','background-color':'#4f6a7a',width:18,height:18,'text-valign':'bottom','text-margin-y':5}},
    {selector:'node[energy > 15]',style:{'background-color':'#67dbca',width:25,height:25}},
    {selector:'node.focus',style:{'border-width':4,'border-color':'#ffffff'}},
    {selector:'node[?hypothesis]',style:{'border-width':2,'border-style':'dashed','border-color':'#e6a761'}},
    {selector:'edge',style:{width:1,'line-color':'#476076','line-style':'dashed','target-arrow-shape':'triangle','target-arrow-color':'#476076','curve-style':'bezier'}},
    {selector:'edge[?confirmed]',style:{'line-style':'solid','line-color':'#67a9a0'}},
    {selector:'edge[edge_origin = "causal_hypothesis"]',style:{'line-color':'#eead69','target-arrow-color':'#eead69',label:'data(type)','font-size':9,color:'#eead69'}},
    {selector:'edge.focus',style:{'line-color':'#ffffff','target-arrow-color':'#ffffff',width:3}}]});jobGraphs[kind]=view;}
  const elements=[...graph.nodes,...graph.edges];const ids=new Set(elements.map(e=>e.id));let added=false;
  view.batch(()=>{
    view.elements().forEach(e=>{if(!ids.has(e.id()))e.remove();});
    for(const data of elements){const existing=view.getElementById(data.id);if(existing.length)existing.data(data);else{added=true;view.add({data,position:{x:Number(data.id)%10*80||0,y:Math.floor(Number(data.id)/10)*80||0}});}}
    view.elements().removeClass('focus');for(const id of path)view.getElementById(String(id)).addClass('focus');
    if(path.length===2)view.edges().forEach(e=>{if(e.data('source')===String(path[0])&&e.data('target')===String(path[1])&&e.data('type')===relation[1])e.addClass('focus');});
  });
  if(added)view.layout({name:'cose',animate:false,randomize:false,padding:30,nodeRepulsion:10000}).run();
}
async function startLive(kind,data){
  activeOnly();if(liveJobs[kind])throw new Error('Остановите текущую операцию.');
  const route=kind==='study'?'study/start':'imitation/start';
  const stop=$(kind==='study'?'study-stop':'simple-stop');const start=$(kind==='study'?'study-start':'simple-start');
  start.disabled=true;let started;
  try{started=await api(route,data);}catch(error){start.disabled=false;throw error;}
  lastJobTrace[kind]=null;lastJobFrame[kind]=null;
  $(kind+'-events').replaceChildren();$(kind+'-summary').textContent='';stop.disabled=false;start.disabled=true;
  if(kind==='study')$('study-source').textContent='Материал ещё не получен.';
  if(kind==='imitation'){$('imitation-help').hidden=true;$('simple-limitation').hidden=true;}
  const stream=new EventSource('/api/jobs/'+started.id+'/events');liveJobs[kind]={id:started.id,stream,last:0};
  stream.onmessage=async message=>{
    const event=JSON.parse(message.data);const live=liveJobs[kind];if(!live||event.sequence<=live.last)return;live.last=event.sequence;
    $(kind+'-status').textContent=(stageNames[event.stage]||event.stage)+(event.round?' · раунд '+event.round:'')+(event.iteration?' · шаг '+event.iteration:'');
    if(event.attempt)lastJobTrace[kind]=event.attempt;
    if(event.result?.after)lastJobTrace[kind]=event.result.after;
    if(event.graph){lastJobFrame[kind]=event.graph;if(kind==='study')$('study-hypotheses').textContent=pretty({node_hypotheses:event.graph.node_hypotheses,causal_hypotheses:event.graph.causal_hypotheses});const trace=event.attempt||lastJobTrace[kind]||{};drawJob(kind,event.graph,trace.path||[],trace.relation||[]);$(kind+'-status').textContent+=` · такт ${event.graph.tick} · ${event.graph.total_nodes} узлов`;}
    if(event.material)$('study-source').textContent=pretty(event.material)+'\nСведения источника не являются доказательством истинности.';
    const card=document.createElement('details');card.className='step'+(event.stage==='candidate'&&!event.eligible?' rejected':'');
    if(event.stage==='candidate'){card.dataset.round=event.round;card.dataset.parameters=JSON.stringify(event.parameters);card.dataset.eligible=event.eligible;}
    const title=document.createElement('summary');title.textContent=`${event.sequence}. ${stageNames[event.stage]||event.stage}${event.iteration?' #'+event.iteration:''}${event.eligible===false?' · отклонён':event.eligible===true?' · прошёл контроли, ещё не применён':''}`;
    const content=document.createElement('pre');const detail={...event};delete detail.graph;content.textContent=pretty(detail);
    const technical=document.createElement('details');const technicalTitle=document.createElement('summary');technicalTitle.textContent='Технические данные: трасса, параметры, контрольные метрики';technical.append(technicalTitle,content);card.append(title);
    if(event.attempt){
      card.open=true;const text=document.createElement('p');text.textContent='Ответ: '+event.attempt.text;card.append(text);
      const comparison=document.createElement('p');const names={A:'ошибка языковой формы',B:'ошибка типа отношения',C:'ошибка выбора известного знания',D:'не хватает подтверждённых знаний',E:'неоднозначность',unknown:'причина не определена',correct:'смысл соответствует разметке'};
      const mark=x=>x===null||x===undefined?'не определено':x===1?'верно':'не совпадает';
      comparison.textContent='Диагноз: '+(names[event.diagnosis?.kind]||'не определён')+'. Смысл: '+mark(event.comparison?.semantic)+'. Форма: '+mark(event.comparison?.form)+'.';card.append(comparison);
      if(event.controls){const control=document.createElement('p');control.textContent='Проверено контрольных задач: '+event.controls.length+'. '+(event.eligible?'Кандидат прошёл проверки; выбор лучшего ещё не завершён.':'Мутация не допущена.');card.append(control);}
    }
    if(event.stage==='observation'){const text=document.createElement('p');text.textContent='Наблюдение '+event.sentence+'/'+event.total_sentences+': '+event.text;card.append(text);}
    card.append(technical);
    $(kind+'-events').append(card);
    if(event.attempt&&event.graph)card.addEventListener('toggle',()=>{if(card.open)drawJob(kind,lastJobFrame[kind]||event.graph,event.attempt.path,event.attempt.relation);});
    if(event.analysis&&(!event.analysis.reliable||event.analysis.controls===0)){$('imitation-help').hidden=false;$('simple-limitation').hidden=false;$('simple-limitation').textContent=event.analysis.limitation||'Нет независимых контрольных примеров из других подтверждённых знаний. Изменение не будет принято без проверок.';}
    if(event.result){const r=event.result;for(const candidate of $(kind+'-events').querySelectorAll('[data-parameters]')){if(Number(candidate.dataset.round)!==event.round)continue;const selected=r.accepted&&Object.entries(JSON.parse(candidate.dataset.parameters)).every(([k,v])=>r.after.parameters[k]===v);const t=candidate.querySelector('summary');t.textContent=t.textContent.replace(' · прошёл контроли, ещё не применён','')+(selected?' · выбран во временной копии':candidate.dataset.eligible==='true'?' · не выбран':'');}
$(kind+'-summary').textContent='Исходный ответ: '+r.before.text+'\nЭталон: '+r.references.join(' / ')+'\nОтвет после раунда: '+(r.after?.text||r.before.text)+'\n'+(r.accepted?'Изменение принято после контрольных проверок.':'Изменение не принято: '+({missing_validation_or_evidence:'недостаточно подтверждённых знаний, надёжной разметки или независимых контролей',no_supported_improvement:'кандидаты не дали улучшения без регрессий',already_consolidated:'ответ уже соответствует заданному образцу; новая мутация не нужна',budget:'исчерпан вычислительный бюджет'}[r.reason]||r.reason))+'\nИзменённые параметры: '+pretty(r.changed||{});$('training-report').textContent=pretty(r);if(lastJobFrame[kind]&&lastJobTrace[kind])drawJob(kind,lastJobFrame[kind],lastJobTrace[kind].path,lastJobTrace[kind].relation);if(!r.accepted)$('imitation-help').hidden=false;}
    if(['completed','cancelled','failed'].includes(event.stage)){
      stream.close();delete liveJobs[kind];stop.disabled=true;start.disabled=false;
      if(event.diff)$(kind+'-summary').textContent+=kind==='study'?'\nДо / после:\n'+pretty(event.diff):'\nУзлов добавлено: '+event.diff.totals.added_nodes+'; новых отношений: '+event.diff.totals.new_relations+'. Калибровка не добавляет знаний.';
      $(kind+'-status').textContent+=' · '+(event.committed?'изменения применены':'рабочее ядро не заменено');
      if(kind==='imitation'&&!event.committed)$('imitation-help').hidden=false;
      await refresh();
    }
  };
  stream.onerror=()=>{if(liveJobs[kind])$(kind+'-status').textContent='Переподключение к потоку событий… Последние шаги будут повторены по ID.';};
}
async function stopLive(kind){const job=liveJobs[kind];if(!job)return;await api('jobs/'+job.id+'/stop',{});$(kind+'-status').textContent='Остановка запрошена. Текущий шаг безопасно завершится, новые итерации не начнутся.';}
bind('study-start',()=>startLive('study',{topic:$('study-topic').value,material:$('study-material').value}));
bind('study-stop',()=>stopLive('study'));
bind('simple-start',()=>startLive('imitation',{question:$('simple-question').value,reference:$('simple-reference').value}));
bind('simple-stop',()=>stopLive('imitation'));
bind('simple-rollback',async()=>{activeOnly();await api('training/rollback',{});await refresh();$('imitation-status').textContent='Последняя принятая мутация отменена.';});
bind('offer-study',async()=>{$('study-topic').value=$('simple-question').value.replace(/^что такое\s+/i,'').replace(/[?!.]+$/,'');document.querySelector('[data-tab="study"]').click();});
bind('retry-imitation',async()=>{document.querySelector('[data-tab="imitation"]').click();});
bind('add-observation',async()=>{activeOnly();const result=await api('imitation/observation',{question:$('simple-question').value,reference:$('simple-reference').value});$('imitation-status').textContent=pretty(result)+' · добавлено наблюдение, не подтверждённый факт';await refresh();});

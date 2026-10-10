const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
let project = null;
let currentFileId = null;
let selectedPointId = null;
let pendingStart = null;
let calibrating = false;
let calibPoints = [];
let progressTimer = null;
let progressValue = 0;
let aiReady = false;
let aiInstallPolling = false;
let shotConfig = null;
let lastShotResult = null;
let lastServeResult = null;

const video = $('#video');
const overlay = $('#overlay');
const ctx = overlay.getContext('2d');
const statusEl = $('#status');
const statusProgressWrap = $('#statusProgressWrap');
const statusProgressFill = $('#statusProgressFill');
const statusProgressText = $('#statusProgressText');

const SHOT_KO = {serve:'서브',forehand:'포핸드',backhand:'백핸드',volley:'발리',smash:'스매시',lob:'로브',dropshot:'드롭샷',other:'기타',stroke:'스트로크'};
const SPIN_KO = {topspin:'탑스핀',slice:'슬라이스',flat:'플랫',kick:'킥',unknown:'모름'};
const DIR_KO = {T:'T',body:'바디',wide:'와이드',unknown:'-'};
const RESULT_KO = {in:'인',fault:'폴트',ace:'에이스',winner:'위너',error:'에러',unknown:'-'};
const CALL_KO = {in:'인',out:'아웃',net:'네트',close:'라인 근처'};
const FIX_KO = {serve_to_smash:'랠리 중 서브를 스매시로',before_serve:'서브 전 타구 제외',fault_return:'폴트 공 넘겨주기 제외',duplicate:'중복 타구 제거',side_flip:'친 쪽 수정',side_fill:'친 쪽 보완',return_volley:'리턴 발리/스매시를 기타로',result_mid_rally:'랠리 중간 위너·에러를 인으로',serve_result:'리턴된 서브를 인으로',result_vs_label:'승자 기록과 충돌해 결과 미확인',tracked_hit:'공 추적으로 찾은 타구',swing_side:'스윙 방향으로 포/백 결정',ball_no_bounce:'바운스 없음 → 발리',ball_bounced:'바운스 있음 → 스트로크',serve_mid_rally:'랠리 중 서브를 스매시로',second_serve_after_fault:'폴트 뒤 세컨드 서브',tracked_hit_type:'공·소리로 종류 결정',ai_saw_no_hit:'AI는 타구를 못 봄',added_by_tracking:'공 추적으로 추가'};

/* ── small helpers ─────────────────────────────── */
function fmt(t){ t=Math.max(0,Number(t)||0); const m=Math.floor(t/60), s=t-m*60; return `${String(m).padStart(2,'0')}:${s.toFixed(3).padStart(6,'0')}`; }
function fmtShort(t){ t=Math.round(Math.max(0,Number(t)||0)*10)/10; const m=Math.floor(t/60), s=t-m*60; return `${m}:${s.toFixed(1).padStart(4,'0')}`; }
function fmtClock(t){ t=Math.max(0,Number(t)||0); const h=Math.floor(t/3600), m=Math.floor((t%3600)/60), s=Math.floor(t%60); return h>0?`${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`:`${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`; }
function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }
function esc(s){ return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function kmh(v){ return v?`${v}`:'-'; }
function fileById(id){ return project?.files?.find(f=>f.id===id); }
function pointById(id){ return project?.points?.find(p=>p.id===id); }
function filePoints(){ return (project?.points||[]).filter(p=>p.file_id===currentFileId); }

/* ── status + progress (one place: top bar) ────── */
function setStatus(msg, bad=false){ statusEl.textContent=msg; statusEl.title=msg; statusEl.classList.toggle('bad',!!bad); }
function showProgress(percent, label){
  statusProgressWrap.classList.remove('hidden'); statusProgressWrap.setAttribute('aria-hidden','false');
  statusProgressFill.classList.remove('indeterminate');
  statusProgressFill.style.width=Math.max(0,Math.min(100,percent)).toFixed(1)+'%';
  statusProgressText.textContent=Math.max(0,Math.min(100,percent)).toFixed(0)+'%';
  if(label!=null) setStatus(label);
}
function setBusyProgress(active, label='진행 중…', opts={}){
  const {mode='pseudo', done=false}=opts;
  if(progressTimer){ clearInterval(progressTimer); progressTimer=null; }
  if(!active){
    progressValue=0; statusProgressWrap.classList.add('hidden'); statusProgressWrap.setAttribute('aria-hidden','true');
    statusProgressFill.classList.remove('indeterminate'); statusProgressFill.style.width='0%'; statusProgressText.textContent='0%';
    return;
  }
  setStatus(label,false);
  if(done){ showProgress(100); return; }
  if(mode==='indeterminate'){
    statusProgressWrap.classList.remove('hidden'); statusProgressFill.classList.add('indeterminate'); statusProgressText.textContent='';
    return;
  }
  progressValue=6; showProgress(progressValue);
  progressTimer=setInterval(()=>{ const step=progressValue<35?7:progressValue<60?4:progressValue<82?2:.6; progressValue=Math.min(94,progressValue+step); showProgress(progressValue); },550);
}
async function finishBusyProgress(message){ setBusyProgress(true,message,{done:true}); await sleep(350); setBusyProgress(false); setStatus(message); }
function clearBusyProgressOnError(message){ setBusyProgress(false); setStatus(message,true); }
function statusLink(url, text){ const a=document.createElement('a'); a.href=url; a.textContent=text; a.target='_blank'; statusEl.appendChild(a); }
function jobProgress(job, fallback){
  const parts=[job.message||fallback];
  if(job.chunk_count) parts.push(`구간 ${job.chunks_done||0}/${job.chunk_count}`);
  if(job.elapsed_seconds) parts.push(`경과 ${fmtClock(job.elapsed_seconds)}`);
  showProgress(Number(job.percent)||0, parts.join(' · '));
}

/* ── server ─────────────────────────────────────── */
async function api(path, data=null){
  const opt={method:data===null?'GET':'POST',headers:{},cache:'no-store'};
  if(data!==null){ opt.headers['Content-Type']='application/json'; opt.body=JSON.stringify(data); }
  let r;
  try{ r=await fetch(path,opt); }
  catch(err){ throw new Error(`서버에 연결할 수 없어요. run_windows.bat 창이 켜져 있는지 확인하세요. (${err.message})`); }
  const text=await r.text(); let j={};
  if(text.trim()){ try{ j=JSON.parse(text); }catch(err){ throw new Error(`서버 응답을 읽지 못했어요 (HTTP ${r.status}). 서버를 껐다가 run_windows.bat로 다시 켜 주세요.`); } }
  if(!r.ok||j.ok===false) throw new Error(j.error||`HTTP ${r.status}`);
  return j;
}
async function ensureProgressServer(){
  try{ const v=await api('/api/version'); if(Number(v.progress_api||0)<4) throw new Error('OLD_SERVER'); return v; }
  catch(err){
    if(/OLD_SERVER|unknown endpoint|HTTP 404/.test(String(err.message))) throw new Error('예전 버전 서버가 켜져 있어요. run_windows.bat 창을 닫고 다시 실행하세요.');
    throw err;
  }
}
async function refresh(){ const j=await api('/api/state'); project=j.project; render(); }

/* ── tabs ───────────────────────────────────────── */
function showTab(name){
  $$('.tab').forEach(t=>t.setAttribute('aria-selected',String(t.dataset.tab===name)));
  $$('.pane').forEach(p=>p.classList.toggle('hidden',p.dataset.pane!==name));
  try{ localStorage.setItem('ht_tab',name); }catch(e){}
}
$$('.tab').forEach(t=>t.addEventListener('click',()=>showTab(t.dataset.tab)));
try{ const saved=localStorage.getItem('ht_tab'); if(saved) showTab(saved); }catch(e){}

/* ── render ─────────────────────────────────────── */
function render(){
  if(!project) return;
  const fs=$('#fileSelect'); const prev=fs.value||currentFileId;
  fs.innerHTML='';
  project.files.forEach(f=>{ const o=document.createElement('option'); o.value=f.id; o.textContent=f.name; fs.appendChild(o); });
  if(!project.files.length){ const o=document.createElement('option'); o.textContent='영상 없음 — 영상 추가를 누르세요'; fs.appendChild(o); fs.disabled=true; currentFileId=null; }
  else { fs.disabled=false; currentFileId=project.files.some(f=>f.id===prev)?prev:project.files[0].id; fs.value=currentFileId; }
  renderSettings(); renderCurrentFile(); renderPoints(); renderScoreboard(); renderSteps(); renderTrack();
}

function updatePosLabels(){
  const singles=$('#mode').value==='singles';
  $$('.pos-label').forEach(el=>{
    el.textContent=singles?'선수':(el.dataset.pos==='ad'?'애드 코트':'듀스 코트');
    el.parentElement.classList.toggle('hidden',singles&&el.dataset.pos==='deuce');
  });
}
function renderSettings(){
  const s=project.settings||{};
  $('#mode').value=s.mode||'doubles';
  $('#a1').value=s.team_a?.[0]||''; $('#a2').value=s.team_a?.[1]||'';
  $('#b1').value=s.team_b?.[0]||''; $('#b2').value=s.team_b?.[1]||'';
  $('#rotation').value=(s.server_rotation||[]).join(',');
  $('#noAd').checked=!!s.no_ad;
  $('#leftHanded').value=(s.left_handed||[]).join(', ');
  $('#positionMode').value=s.scoreboard_position_mode||'court';
  $('#teamAStart').value=s.team_a_start_end||'near';
  $('#changeEnds').checked=s.change_ends!==false;
  updatePosLabels();
}

let shotLoadedFor=null;
const proxyRev={};  // bumped when a copy is rebuilt, so the player doesn't keep the old one
function renderCurrentFile(){
  const f=fileById(currentFileId);
  if(!f){
    video.removeAttribute('src'); video.dataset.src='';
    $('#videoEmpty').classList.remove('hidden'); $('#videoEmptyText').textContent='위의 “영상 추가”로 경기 영상을 불러오세요.';
    $('#fileSummary').textContent='영상을 추가하세요.'; shotLoadedFor=null; $('#shotResult').innerHTML=''; $('#serveResult').innerHTML='';
    return;
  }
  if(shotLoadedFor!==f.id){ loadShotResult(f.id); loadServeResult(f.id); renderShotLabels(); refreshLabelEval(); }
  const m=f.metadata||{};
  const mb=m.size?`${(m.size/1e9).toFixed(1)}GB`:'';
  $('#fileSummary').textContent=[f.name, m.width&&`${m.width}×${m.height}`, m.fps&&`${Math.round(m.fps)}fps`, m.duration&&fmtClock(m.duration), mb].filter(Boolean).join(' · ');
  $('#proxyStatus').textContent=f.proxy?'재생용 사본 있음':'';
  $('#btnProxy').textContent=f.proxy?'사본 다시 만들기':'재생용 사본 만들기';
  $('#btnProxy').classList.toggle('ghost',!!f.proxy); $('#btnProxy').classList.toggle('primary',!f.proxy);
  const src=f.proxy?`/media/${f.id}${proxyRev[f.id]?`?r=${proxyRev[f.id]}`:''}`:'';
  if(src && video.dataset.src!==src){ video.src=src; video.dataset.src=src; video.load(); }
  if(!src){ video.removeAttribute('src'); video.dataset.src=''; }
  $('#videoEmpty').classList.toggle('hidden',!!src);
  $('#videoEmptyText').textContent='원본은 크기가 커서 바로 재생되지 않아요. 준비 탭에서 “재생용 사본 만들기”를 먼저 눌러 주세요.';
  calibPoints=(f.calibration?.points||[]).map(p=>({...p})); drawOverlay();
  $('#btnCalibRevert').classList.toggle('hidden',!f.calibration?.manual_points);
  $('#btnCalibRefine').disabled=!(f.calibration?.points?.length>=4);
  $('#btnCalibrate').textContent=f.calibration?.points?.length>=4?'다시 찍기':'모서리 4곳 찍기';
}

function setStep(id, state, text){
  const el=$('#'+id); el.classList.remove('done','todo'); if(state) el.classList.add(state);
  $('#'+id+'State').textContent=text||'';
}
function renderSteps(){
  const f=fileById(currentFileId); const s=project?.settings||{};
  const video_ok=!!f?.proxy, court_ok=(f?.calibration?.points?.length||0)>=4;
  const players_ok=!!(s.team_a?.[0] && s.team_b?.[0]);
  const ai_ok=aiReady;
  setStep('stepVideo', !f?'todo':video_ok?'done':'todo', !f?'영상 없음':video_ok?'완료':'사본 필요');
  setStep('stepCourt', court_ok?'done':'todo', court_ok?'완료':'필요');
  setStep('stepPlayers', players_ok?'done':'todo', players_ok?'완료':'이름 입력');
  setStep('stepAI', ai_ok?'done':'todo', ai_ok?'완료':'설치 필요');
  const n=[video_ok,court_ok,players_ok,ai_ok].filter(Boolean).length;
  const b=$('#tabSetupBadge'); b.textContent=n===4?'':`${n}/4`; b.classList.toggle('todo',n<4);
  const pts=filePoints().length; $('#tabPointsBadge').textContent=pts?String(pts):'';
}

function renderPoints(){
  const wrap=$('#points'); const pts=filePoints();
  const all=project.points||[];
  if(!pts.length){
    wrap.innerHTML=`<div class="empty">아직 기록한 포인트가 없어요.<br>영상을 보면서 서브 직전에 <kbd>S</kbd>, 포인트가 끝나면 <kbd>E</kbd>를 누르세요.</div>`;
  }else{
    wrap.innerHTML=pts.map(p=>{
      const i=all.indexOf(p); const sc=p.score_before||{};
      const win=p.winner?`<span class="win ${p.winner.toLowerCase()}">${p.winner} 승</span>`:'<span class="win">미정</span>';
      return `<div class="pt-row${p.id===selectedPointId?' sel':''}" role="option" tabindex="0" aria-selected="${p.id===selectedPointId}" data-id="${p.id}">
        <span class="pt-no">#${i+1}</span>
        <span class="pt-time">${fmtShort(p.start)}<small>${(p.end-p.start).toFixed(1)}초</small></span>
        <span class="pt-score">${sc.set_games_a??0}-${sc.set_games_b??0} · ${sc.point_a||'0'}:${sc.point_b||'0'}</span>${win}</div>`;
    }).join('');
    wrap.querySelectorAll('.pt-row').forEach(row=>{
      const p=pointById(row.dataset.id);
      row.addEventListener('click',()=>selectPoint(p,true));
      row.addEventListener('dblclick',()=>openPoint(p));
      row.addEventListener('keydown',e=>{ if(e.key==='Enter') openPoint(p); });
    });
  }
  const wins=pts.filter(p=>p.winner).length;
  $('#pointSummary').textContent=pts.length?`${pts.length}개 · 승자 기록 ${wins}개`:'';
  const sel=pointById(selectedPointId);
  $('#selectedBox').classList.toggle('has',!!sel);
  $$('#selectedBox .selected-actions button').forEach(b=>b.disabled=!sel);
  if(sel){
    const i=all.indexOf(sel);
    $('#selectedInfo').innerHTML=`<b>포인트 #${i+1}</b> · ${fmtShort(sel.start)} → ${fmtShort(sel.end)} <span class="muted">(${(sel.end-sel.start).toFixed(1)}초) · 더블클릭하면 재생</span>`;
  }else{
    $('#selectedInfo').innerHTML='목록에서 포인트를 고르세요. 새 포인트는 영상 아래 <kbd>S</kbd> 시작 · <kbd>E</kbd> 끝으로 기록해요.';
  }
  $('#btnWinnerA').classList.toggle('on-a',sel?.winner==='A');
  $('#btnWinnerB').classList.toggle('on-b',sel?.winner==='B');
}
function selectPoint(p, seek){
  selectedPointId=p.id; renderPoints(); renderScoreboard(); renderTrack();
  if(seek && fileById(p.file_id)?.proxy){ video.pause(); video.currentTime=p.start; }
}

function renderScoreboard(){
  let st=null; const sel=pointById(selectedPointId);
  if(sel) st=sel.score_before;
  if(!st && project.points?.length) st=project.points[project.points.length-1].score_after;
  st=st||{sets_a:0,sets_b:0,set_games_a:0,set_games_b:0,point_a:'0',point_b:'0',server_id:(project.settings?.server_rotation||['A1'])[0],team_a_end:'near'};
  const s=project.settings||{};
  const ta=(s.team_a||[]).filter(Boolean).join(' · ')||'Team A', tb=(s.team_b||[]).filter(Boolean).join(' · ')||'Team B';
  let rows=[{team:'A',name:ta,sets:st.sets_a,games:st.set_games_a,pts:st.point_a},{team:'B',name:tb,sets:st.sets_b,games:st.set_games_b,pts:st.point_b}];
  if(s.scoreboard_position_mode==='court' && st.team_a_end==='far') rows.reverse();
  $('#scoreboard').innerHTML=`<div class="sb-row head"><span>${sel?'이 포인트 시작 전':'현재 스코어'}</span><span class="n">세트</span><span class="n">게임</span><span class="n">포인트</span></div>`
    +rows.map(r=>`<div class="sb-row"><span class="sb-name"><i class="srv${String(st.server_id||'').startsWith(r.team)?'':' off'}" title="서브"></i><span class="team-tag ${r.team.toLowerCase()}" style="width:22px;height:22px;font-size:11px">${r.team}</span>${esc(r.name)}</span><span class="n">${r.sets}</span><span class="n">${r.games}</span><span class="pts">${r.pts}</span></div>`).join('');
}

/* ── timeline track under the video ─────────────── */
function trackDuration(){ return video.duration||fileById(currentFileId)?.metadata?.duration||0; }
function renderTrack(){
  const bar=$('#trackBar'); const head=$('#trackHead');
  bar.querySelectorAll('.seg-pt,.mk').forEach(n=>n.remove());
  const dur=trackDuration(); if(!dur) return;
  const pct=t=>`${(t/dur*100).toFixed(3)}%`;
  const frag=document.createDocumentFragment();
  filePoints().forEach(p=>{
    const d=document.createElement('div');
    d.className='seg-pt'+(p.id===selectedPointId?' sel':'')+(p.winner?` win-${p.winner.toLowerCase()}`:'');
    d.style.left=pct(p.start); d.style.width=`max(3px, ${((p.end-p.start)/dur*100).toFixed(3)}%)`;
    d.title=`포인트 ${fmtShort(p.start)}–${fmtShort(p.end)}`; frag.appendChild(d);
  });
  const serves=(lastServeResult?.serves||[]).map(s=>s.contact_t);
  const hits=(lastShotResult?.events||[]).filter(e=>e.shot!=='serve').map(e=>e.t);
  hits.forEach(t=>{ const m=document.createElement('i'); m.className='mk'; m.style.left=pct(t); frag.appendChild(m); });
  serves.forEach(t=>{ const m=document.createElement('i'); m.className='mk sv'; m.style.left=pct(t); frag.appendChild(m); });
  bar.insertBefore(frag,head);
  head.style.left=pct(video.currentTime||0);
}
$('#trackBar').addEventListener('click',e=>{
  const dur=trackDuration(); if(!dur||!fileById(currentFileId)?.proxy) return;
  const r=e.currentTarget.getBoundingClientRect(); const t=(e.clientX-r.left)/r.width*dur;
  video.currentTime=t;
  const p=filePoints().find(p=>t>=p.start&&t<=p.end); if(p&&p.id!==selectedPointId){ selectedPointId=p.id; renderPoints(); renderScoreboard(); renderTrack(); }
});

function openPoint(p){
  if(currentFileId!==p.file_id){ currentFileId=p.file_id; $('#fileSelect').value=currentFileId; renderCurrentFile(); }
  if(!fileById(p.file_id)?.proxy){ setStatus('재생용 사본을 먼저 만들어 주세요 (준비 탭).',true); return; }
  video.currentTime=p.start; video.play();
  window.setTimeout(()=>{ if(video.currentTime>=p.end-0.15) video.pause(); },Math.max(100,(p.end-p.start)*1000));
}

/* ── overlay (court corners) ────────────────────── */
function resizeOverlay(){ const r=$('#videoWrap').getBoundingClientRect(); overlay.width=Math.round(r.width*devicePixelRatio); overlay.height=Math.round(r.height*devicePixelRatio); drawOverlay(); }
function drawOverlay(){
  const w=overlay.width,h=overlay.height; ctx.clearRect(0,0,w,h);
  if(!calibPoints.length || !(calibrating||showCourtLines)) return;
  const dpr=devicePixelRatio;
  ctx.lineWidth=2*dpr; ctx.strokeStyle='rgba(205,230,106,.9)'; ctx.fillStyle='#cde66a'; ctx.font=`600 ${13*dpr}px Pretendard, sans-serif`;
  if(calibPoints.length>=4){ ctx.beginPath(); calibPoints.slice(0,4).forEach((p,i)=>{ const x=p.x*w,y=p.y*h; i?ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.closePath(); ctx.stroke(); }
  calibPoints.forEach((p,i)=>{ const x=p.x*w,y=p.y*h; ctx.beginPath(); ctx.arc(x,y,5*dpr,0,Math.PI*2); ctx.fill(); ctx.fillText(String(i+1),x+8*dpr,y-8*dpr); });
}
let showCourtLines=false;

/* ── video events, transport ────────────────────── */
$('#mode').addEventListener('change',updatePosLabels);
$('#fileSelect').addEventListener('change',()=>{ currentFileId=$('#fileSelect').value; selectedPointId=null; renderCurrentFile(); renderPoints(); renderScoreboard(); renderSteps(); renderTrack(); });
video.addEventListener('timeupdate',()=>{ $('#currentTime').textContent=fmt(video.currentTime); const d=trackDuration(); if(d) $('#trackHead').style.left=`${(video.currentTime/d*100).toFixed(3)}%`; });
video.addEventListener('error',()=>{ if(video.dataset.src) setStatus('브라우저가 이 재생용 사본을 열지 못했어요. 준비 탭에서 “사본 다시 만들기”를 눌러 주세요.',true); });
video.addEventListener('loadedmetadata',()=>{ $('#duration').textContent=fmt(video.duration); resizeOverlay(); renderTrack(); });
window.addEventListener('resize',resizeOverlay);
function setRate(r){ video.playbackRate=r; $$('.rate .chip').forEach(c=>c.classList.toggle('on',Number(c.dataset.rate)===r)); }
$$('.rate .chip').forEach(c=>c.addEventListener('click',()=>setRate(Number(c.dataset.rate))));

/* ── setup actions ──────────────────────────────── */
$('#btnSelect').onclick=async()=>{ try{ setStatus('파일 선택 창을 여는 중…'); await api('/api/select_files',{}); await refresh(); setStatus('영상을 추가했어요. 준비 탭에서 재생용 사본을 만드세요.'); showTab('setup'); }catch(e){ setStatus(e.message,true); } };
async function waitForProxyJob(fid){
  while(true){
    const r=await api(`/api/proxy_status/${fid}`); const job=r.job||{};
    const out=Number(job.out_time)||0, dur=Number(job.duration)||0, eta=job.eta_seconds==null?null:Number(job.eta_seconds);
    const parts=[job.message||'재생용 사본 만드는 중'];
    if(dur>0) parts.push(`${fmtClock(out)} / ${fmtClock(dur)}`);
    if(eta!==null&&Number.isFinite(eta)) parts.push(`약 ${fmtClock(eta)} 남음`);
    showProgress(Number(job.percent)||0, parts.join(' · '));
    if(job.status==='done'){ proxyRev[fid]=Date.now(); await refresh(); await finishBusyProgress(`재생용 사본 완료 · ${fmtClock(job.elapsed_seconds||0)} 걸림`); return job; }
    if(job.status==='error'){ setBusyProgress(false); throw new Error(job.error||job.message||'사본 만들기 실패'); }
    await sleep(500);
  }
}
$('#btnProxy').onclick=async()=>{
  if(!currentFileId) return; const fid=currentFileId;
  try{ $('#btnProxy').disabled=true; showProgress(0,'서버 확인 중…'); await ensureProgressServer(); await api(`/api/proxy/${fid}`,{}); await waitForProxyJob(fid); }
  catch(e){ clearBusyProgressOnError(e.message); } finally{ $('#btnProxy').disabled=false; }
};

function renderAIStatus(job){
  const dep=job?.dependencies||{}; const el=$('#aiStatus'), install=$('#btnInstallAI');
  aiReady=!!dep.ready; el.className='hint';
  if(aiReady){ el.textContent='설치됨'; el.classList.add('ok'); install.classList.add('hidden'); $('#btnFormation').disabled=false; }
  else{
    $('#btnFormation').disabled=true; install.classList.remove('hidden');
    if(job?.status==='running'||job?.status==='queued'){ el.textContent=`설치 중 ${Math.round(Number(job?.percent)||0)}%`; el.classList.add('warn'); install.disabled=true; }
    else if(job?.status==='error'){ el.textContent='설치 실패'; el.classList.add('bad'); install.disabled=false; }
    else{ el.textContent='설치 필요'; el.classList.add('warn'); install.disabled=false; }
  }
  if(project) renderSteps();
}
async function refreshAIStatus(){ try{ const r=await api('/api/ai_status'); renderAIStatus(r.job||{}); return r.job||{}; }catch(e){ $('#aiStatus').textContent='확인 실패'; return null; } }
async function waitForAIInstall(){
  if(aiInstallPolling) return; aiInstallPolling=true;
  try{
    while(true){
      const r=await api('/api/ai_status'); const job=r.job||{};
      renderAIStatus(job);
      const parts=[job.message||'AI 패키지 설치 중'];
      if(job.package_count) parts.push(`${job.completed_packages||0}/${job.package_count} 패키지`);
      if(job.current_package) parts.push(job.current_package);
      showProgress(Number(job.percent)||0, parts.join(' · '));
      if(job.status==='done'&&job.dependencies?.ready){ await finishBusyProgress(`AI 설치 완료 · ${fmtClock(job.elapsed_seconds||0)} 걸림`); return; }
      if(job.status==='error'){ setBusyProgress(false); throw new Error(job.error||'AI 패키지 설치에 실패했어요.'); }
      await sleep(500);
    }
  }finally{ aiInstallPolling=false; }
}
$('#btnInstallAI').onclick=async()=>{
  try{ $('#btnInstallAI').disabled=true; await ensureProgressServer(); showProgress(0,'AI 설치 시작 중… 처음엔 몇 분 걸려요.'); const r=await api('/api/install_ai',{}); renderAIStatus(r.job||{}); await waitForAIInstall(); }
  catch(e){ clearBusyProgressOnError(e.message); await refreshAIStatus(); }
};
$('#btnFormation').onclick=async()=>{
  if(!currentFileId||!aiReady) return;
  try{ setBusyProgress(true,'선수 대형 분석 중… 처음엔 모델 다운로드로 시간이 걸려요.',{mode:'indeterminate'}); const j=await api(`/api/analyze_formation/${currentFileId}`,{}); await finishBusyProgress(`대형 분석 완료 · 특징 ${j.feature_count}개 · 변화 지점 ${j.change_points.length}개`); }
  catch(e){ clearBusyProgressOnError(e.message); }
};

function renderShotConfig(cfg){
  shotConfig=cfg; const el=$('#shotKeyStatus'); el.className='hint';
  if(cfg?.has_key){ el.textContent=`${cfg.model} · ${cfg.key_source==='env'?'환경변수 키':'키 저장됨'}`; el.classList.add('ok'); }
  else{ el.textContent='키 없음'; el.classList.add('warn'); }
  $('#shotModelInput').value=cfg?.model||'';
  if(project) renderSteps();
}
async function refreshShotConfig(){ try{ const r=await api('/api/shot_config'); renderShotConfig(r.config); }catch(e){ $('#shotKeyStatus').textContent='확인 실패'; } }
$('#btnShotKey').onclick=()=>{ $('#shotKeyForm').classList.toggle('hidden'); $('#shotKeyInput').focus(); };
$('#btnShotKeyClose').onclick=()=>$('#shotKeyForm').classList.add('hidden');
$('#btnShotKeySave').onclick=async()=>{
  try{
    const data={model:$('#shotModelInput').value.trim()}; const key=$('#shotKeyInput').value.trim();
    if(key||confirm('키 칸이 비어 있어요. 저장된 키를 지울까요?')) data.api_key=key;
    const r=await api('/api/shot_config',data); renderShotConfig(r.config); $('#shotKeyInput').value=''; $('#shotKeyForm').classList.add('hidden');
    setStatus(r.config.has_key?'Gemini 설정을 저장했어요.':'Gemini 키를 지웠어요.');
  }catch(e){ setStatus(e.message,true); }
};

$('#btnApplySettings').onclick=async()=>{
  const mode=$('#mode').value;
  const data={mode,team_a:[$('#a1').value,$('#a2').value].filter((x,i)=>mode==='doubles'||i===0),team_b:[$('#b1').value,$('#b2').value].filter((x,i)=>mode==='doubles'||i===0),
    server_rotation:$('#rotation').value.split(',').map(x=>x.trim()).filter(Boolean),no_ad:$('#noAd').checked,left_handed:$('#leftHanded').value.split(',').map(x=>x.trim()).filter(Boolean),
    scoreboard_position_mode:$('#positionMode').value,team_a_start_end:$('#teamAStart').value,change_ends:$('#changeEnds').checked};
  try{ await api('/api/settings',data); await refresh(); setStatus('선수와 규칙을 저장했어요.'); }catch(e){ setStatus(e.message,true); }
};

$('#btnCalibRefine').onclick=async()=>{
  if(!currentFileId) return;
  try{
    setBusyProgress(true,'코트 라인에 맞춰 보정 중…',{mode:'indeterminate'});
    const r=await api(`/api/calibration_refine/${currentFileId}`,{}); await refresh();
    const ref=r.calibration.refined;
    await finishBusyProgress(`보정 완료 · 라인과 어긋남 ${ref.line_error_px_before}px → ${ref.line_error_px_after}px. 분석을 다시 돌리면 반영돼요.`);
    flashCourt();
  }catch(e){ clearBusyProgressOnError(e.message); }
};
$('#btnCalibRevert').onclick=async()=>{ if(!currentFileId) return; try{ await api(`/api/calibration_revert/${currentFileId}`,{}); await refresh(); setStatus('직접 찍은 점으로 되돌렸어요.'); flashCourt(); }catch(e){ setStatus(e.message,true); } };
$('#btnCalibrate').onclick=()=>{
  if(!fileById(currentFileId)?.proxy) return setStatus('재생용 사본을 먼저 만들어 주세요.',true);
  calibrating=true; calibPoints=[]; $('#videoWrap').classList.add('calibrating'); $('#calibHelp').classList.remove('hidden'); video.pause(); drawOverlay();
};
$('#btnCancelCalib').onclick=()=>{ calibrating=false; $('#videoWrap').classList.remove('calibrating'); $('#calibHelp').classList.add('hidden'); renderCurrentFile(); };
overlay.addEventListener('click',async e=>{
  if(!calibrating) return;
  const r=overlay.getBoundingClientRect(); calibPoints.push({x:(e.clientX-r.left)/r.width,y:(e.clientY-r.top)/r.height}); drawOverlay();
  if(calibPoints.length===4){
    try{ await api(`/api/calibration/${currentFileId}`,{points:calibPoints}); calibrating=false; $('#videoWrap').classList.remove('calibrating'); $('#calibHelp').classList.add('hidden'); await refresh(); setStatus('코트 위치를 저장했어요.'); flashCourt(); }
    catch(err){ setStatus(err.message,true); }
  }
});
function flashCourt(){ showCourtLines=true; drawOverlay(); setTimeout(()=>{ showCourtLines=false; drawOverlay(); },2500); }

/* ── points ─────────────────────────────────────── */
$('#btnStart').onclick=()=>{ if(!fileById(currentFileId)?.proxy) return setStatus('재생용 사본을 먼저 만들어 주세요.',true); pendingStart=video.currentTime; setStatus(`포인트 시작 ${fmtShort(pendingStart)} — 끝나면 E`); };
$('#btnEnd').onclick=async()=>{
  if(pendingStart===null) return setStatus('먼저 S로 시작을 찍어 주세요.',true);
  try{ const end=video.currentTime; await api('/api/points',{file_id:currentFileId,start:pendingStart,end}); pendingStart=null; await refresh(); selectedPointId=filePoints().at(-1)?.id; renderPoints(); renderTrack(); setStatus('포인트를 추가했어요. A/B로 승자를 기록하세요.'); }
  catch(e){ setStatus(e.message,true); }
};
async function setWinner(w){ if(!selectedPointId) return setStatus('포인트를 먼저 고르세요.',true); try{ await api(`/api/points/${selectedPointId}/winner`,{winner:w}); await refresh(); setStatus(w?`${w} 승으로 기록했어요.`:'승자를 지웠어요.'); }catch(e){ setStatus(e.message,true); } }
$('#btnWinnerA').onclick=()=>setWinner('A'); $('#btnWinnerB').onclick=()=>setWinner('B'); $('#btnWinnerClear').onclick=()=>setWinner(null);
$('#btnSetStart').onclick=async()=>{ if(!selectedPointId) return; await api(`/api/points/${selectedPointId}/start`,{time:video.currentTime}); await refresh(); setStatus('시작 시각을 고쳤어요.'); };
$('#btnSetEnd').onclick=async()=>{ if(!selectedPointId) return; await api(`/api/points/${selectedPointId}/end`,{time:video.currentTime}); await refresh(); setStatus('끝 시각을 고쳤어요.'); };
$('#btnDelete').onclick=async()=>{ if(!selectedPointId) return; if(!confirm('이 포인트를 지울까요?')) return; await api(`/api/points/${selectedPointId}/delete`,{}); selectedPointId=null; await refresh(); setStatus('포인트를 지웠어요.'); };

/* ── project menu ───────────────────────────────── */
$('#btnSave').onclick=async()=>{ $('.menu').open=false; const n=prompt('저장할 파일 이름','tennis_project.json'); if(!n) return; try{ const j=await api('/api/save',{name:n}); setStatus(`저장했어요: ${j.path}`); }catch(e){ setStatus(e.message,true); } };
$('#btnLoad').onclick=async()=>{ $('.menu').open=false; try{ const j=await api('/api/load',{}); if(j.loaded){ await refresh(); setStatus('프로젝트를 열었어요.'); } }catch(e){ setStatus(e.message,true); } };
document.addEventListener('click',e=>{ const m=$('.menu'); if(m.open && !m.contains(e.target)) m.open=false; });

/* ── keyboard ───────────────────────────────────── */
const LABEL_KEYS={x:'serve',f:'forehand',d:'backhand',v:'volley',m:'smash',l:'lob'};
document.addEventListener('keydown',e=>{
  if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)||e.ctrlKey||e.metaKey||e.altKey) return;
  const k=e.key.toLowerCase();
  if(e.code==='Space'){ e.preventDefault(); video.paused?video.play():video.pause(); }
  else if(LABEL_KEYS[k]){ e.preventDefault(); addShotLabel(LABEL_KEYS[k],e.shiftKey?'far':'near'); }
  else if(e.key===','||e.key==='<'){ setRate(Math.max(0.25,video.playbackRate/2)); setStatus(`재생 속도 ${video.playbackRate}×`); }
  else if(e.key==='.'||e.key==='>'){ setRate(Math.min(2,video.playbackRate*2)); setStatus(`재생 속도 ${video.playbackRate}×`); }
  else if(k==='s') $('#btnStart').click();
  else if(k==='e') $('#btnEnd').click();
  else if(k==='a') $('#btnWinnerA').click();
  else if(k==='b') $('#btnWinnerB').click();
});

/* ── grading (ground-truth labels) ──────────────── */
async function addShotLabel(shot, side){
  if(!currentFileId||!fileById(currentFileId)?.proxy) return setStatus('재생용 사본이 있는 영상에서만 정답을 기록할 수 있어요.',true);
  try{
    const r=await api('/api/shot_labels',{file_id:currentFileId,t:video.currentTime,shot,side});
    project=r.project; renderShotLabels(); setStatus(`정답 기록: ${fmtShort(r.label.t)} ${SHOT_KO[shot]} (${side==='far'?'먼 쪽':'카메라 쪽'})`); refreshLabelEval();
  }catch(err){ setStatus(err.message,true); }
}
function renderShotLabels(){
  const box=$('#labelList');
  const labs=(project?.shot_labels||[]).filter(l=>l.file_id===currentFileId);
  if(!labs.length){ box.innerHTML='<div class="empty">아직 기록한 정답이 없어요. 위의 키로 기록하세요.</div>'; return; }
  box.innerHTML=`<div class="table-wrap"><table class="t"><tr><th>시각</th><th>타구</th><th>선수</th><th>메모</th><th></th></tr>`
    +labs.map(l=>`<tr class="seek" data-t="${l.t}"><td>${fmtShort(l.t)}</td><td>${SHOT_KO[l.shot]}${l.spin?` <span class="sub">${SPIN_KO[l.spin]||l.spin}</span>`:''}</td><td>${l.side==='far'?'먼 쪽':l.side==='near'?'카메라 쪽':'-'}</td><td class="sub">${esc(l.note||'')}</td><td class="num"><button class="icon-btn" data-del="${l.id}" aria-label="정답 삭제" title="삭제">×</button></td></tr>`).join('')+'</table></div>';
  box.querySelectorAll('tr.seek').forEach(tr=>tr.addEventListener('click',e=>{ if(e.target.dataset.del) return; seekTo(currentFileId,Number(tr.dataset.t)-0.6); }));
  box.querySelectorAll('button[data-del]').forEach(b=>b.addEventListener('click',async()=>{ const r=await api(`/api/shot_labels/${b.dataset.del}/delete`,{}); project=r.project; renderShotLabels(); refreshLabelEval(); }));
}
async function refreshLabelEval(){
  const box=$('#labelEval'); if(!currentFileId){ box.innerHTML=''; return; }
  try{
    const r=await api(`/api/shot_label_eval/${currentFileId}`); const ev=r.eval;
    if(!ev){ box.innerHTML=r.reason?`<p class="hint">${esc(r.reason)}</p>`:''; return; }
    const acc=ev.labels?Math.round(ev.correct/ev.labels*100):0;
    const rows=Object.entries(ev.per_type).map(([k,v])=>`<tr><td>${SHOT_KO[k]}</td><td class="num">${v.correct} / ${v.labelled}</td><td class="num">${v.ai}</td><td class="num">${v.recall==null?'-':v.recall+'%'}</td><td class="num">${v.precision==null?'-':v.precision+'%'}</td></tr>`).join('');
    const conf=Object.entries(ev.confusion).map(([k,v])=>{ const [a,b]=k.split('->'); return `<li>${SHOT_KO[a]}을(를) ${SHOT_KO[b]}(으)로 ${v}번</li>`; }).join('');
    const side=ev.side?.checked?`${ev.side.correct}/${ev.side.checked}`:'-';
    const spin=ev.serve_spin?.checked?`${ev.serve_spin.correct}/${ev.serve_spin.checked}`:'-';
    box.innerHTML=`<div class="grade-sum">
        <div><b>${acc}%</b><span>타구 종류 맞음 (${ev.correct}/${ev.labels})</span></div>
        <div><b>${side}</b><span>친 쪽 맞음</span></div>
        <div><b>${spin}</b><span>서브 구질 맞음</span></div>
        <div><b>${ev.missed.length}</b><span>놓친 타구</span></div></div>
      <div class="table-wrap"><table class="t"><tr><th>타구</th><th class="num">맞음 / 정답</th><th class="num">AI가 낸 수</th><th class="num">찾은 비율</th><th class="num">맞힌 비율</th></tr>${rows}</table></div>
      ${conf?`<details class="notes"><summary>헷갈린 것 보기</summary><ul>${conf}</ul></details>`:''}
      ${ev.missed.length?`<details class="notes"><summary>놓친 타구 ${ev.missed.length}개</summary><ul>${ev.missed.map(m=>`<li>${fmtShort(m.t)} ${SHOT_KO[m.shot]}</li>`).join('')}</ul></details>`:''}
      <p class="footnote">정답을 기록한 포인트(${ev.points_scored}개)만 채점해요.</p>`;
  }catch(err){ box.innerHTML=`<p class="hint bad">${esc(err.message)}</p>`; }
}

/* ── analysis results ───────────────────────────── */
function who(e){
  if(e.player){ const s=project?.settings||{}; const team=e.player[0]==='A'?s.team_a:s.team_b; const name=(team||[])[Number(e.player.slice(1))-1]; return esc(name||e.player); }
  if(e.team) return `Team ${e.team}`;
  return {near:'카메라 쪽',far:'먼 쪽'}[e.side]||'-';
}
function fixMark(e){ const f=(e.fixes||[]).filter(x=>x!=='tracked_hit'); return f.length?`<span class="mark" title="${esc(f.map(x=>FIX_KO[x]||x).join(', '))}">보정</span>`:''; }
function stat(k,v,s){ return `<div class="stat"><span class="k">${k}</span><span class="v">${v}</span><span class="s">${s||'&nbsp;'}</span></div>`; }
function speedText(sp){ return sp&&sp.n?`평균 ${sp.avg} · 최고 ${sp.max} km/h`:''; }

async function loadShotResult(fid){
  if(!fid){ $('#shotResult').innerHTML=''; return; }
  shotLoadedFor=fid;
  if(!(project?.points||[]).some(p=>p.file_id===fid)) $('#shotScope').value='full';
  try{
    const r=await api(`/api/shot_result/${fid}`); if(shotLoadedFor!==fid) return;
    renderShotResult(r.result);
    const st=await api(`/api/shot_status/${fid}`); if(['queued','running'].includes(st.job?.status)) waitForShotJob(fid).catch(e=>setStatus(e.message,true));
    const dj=await api(`/api/export_dashboard_status/${fid}`); if(['queued','running'].includes(dj.job?.status)) waitForDashJob(fid).catch(e=>setStatus(e.message,true));
  }catch(e){ $('#shotResult').innerHTML=`<p class="hint bad">결과를 불러오지 못했어요: ${esc(e.message)}</p>`; }
}
function renderShotResult(res){
  lastShotResult=res||null; updateDashButton(); renderTrack();
  const box=$('#shotResult');
  if(!res){
    box.innerHTML=`<div class="empty">아직 분석 결과가 없어요.<br>${filePoints().length?'위의 <b>분석하기</b>를 누르세요.':'포인트 탭에서 포인트를 먼저 기록하면 서버·팀까지 구분돼요.'}</div>`;
    return;
  }
  const sm=res.summary, sv=sm.serve, c=sm.counts;
  const first=sv.by_number['1'], second=sv.by_number['2'];
  const ms=res.measured_serves;
  let h=`<p class="res-meta">${esc(res.analyzed_at.replace('T',' ').slice(0,16))} · ${res.scope==='points'?`포인트 ${res.segment_count}개`:'영상 전체'} · ${esc(res.model)}${res.prompt_variant==='hits'?' · 타구마다 판정':''}</p>`;
  h+=`<div class="stats">`
    +stat('포핸드',c.forehand,speedText(sm.forehand.speed))
    +stat('백핸드',c.backhand,speedText(sm.backhand.speed))
    +stat('서브',sv.count,`1st ${first.count} · 2nd ${second.count}`)
    +stat('발리',c.volley,speedText(sm.volley.speed))
    +stat('스매시 · 로브',`${c.smash} · ${c.lob}`,c.dropshot?`드롭샷 ${c.dropshot}`:'')
    +stat('서브 속도',ms?.count?`${ms.avg}`:'-',ms?.count?`km/h 평균 · 최고 ${ms.max}`:'공 추적 필요')
    +`</div>`;

  const types=['flat','slice','kick','unknown'];
  const maxType=Math.max(1,...types.map(t=>sv.by_type[t].count));
  h+=`<div class="split"><section class="block"><div class="block-head"><h3>서브 구질</h3></div><div class="table-wrap"><table class="t"><tr><th>구질</th><th class="num">횟수</th><th class="num">평균</th><th class="num">최고</th></tr>`
    +types.filter(t=>sv.by_type[t].count).map(t=>{ const b=sv.by_type[t]; return `<tr><td>${SPIN_KO[t]}<div class="bar"><i style="width:${b.count/maxType*100}%"></i></div></td><td class="num">${b.count}</td><td class="num">${kmh(b.speed.avg)}</td><td class="num">${kmh(b.speed.max)}</td></tr>`; }).join('')
    +`</table></div></section>`;
  h+=`<section class="block"><div class="block-head"><h3>1st · 2nd 서브</h3></div><div class="table-wrap"><table class="t"><tr><th></th><th class="num">횟수</th><th class="num">들어감</th><th class="num">평균</th></tr>`
    +[['1st',first],['2nd',second]].map(([k,b])=>`<tr><td>${k}</td><td class="num">${b.count}</td><td class="num">${b.in_rate==null?'-':b.in_rate+'%'}</td><td class="num">${kmh(b.speed.avg)}</td></tr>`).join('')
    +`</table></div><p class="footnote">방향 ${['T','body','wide'].map(d=>`${DIR_KO[d]} ${sv.directions[d]}`).join(' · ')}</p></section></div>`;

  const groups=Object.values(sm.groups||{}), servers=Object.values(sm.servers||{});
  if(groups.length){
    const cols=['serve','forehand','backhand','volley','smash','lob'];
    h+=`<section class="block"><div class="block-head"><h3>선수·위치별</h3></div><div class="table-wrap"><table class="t"><tr><th></th>${cols.map(k=>`<th class="num">${SHOT_KO[k]}</th>`).join('')}</tr>`
      +groups.map(g=>`<tr><td>${esc(g.label)}</td>${cols.map(k=>`<td class="num">${g.counts[k]||0}</td>`).join('')}</tr>`).join('')+`</table></div></section>`;
  }
  if(servers.length){
    h+=`<section class="block"><div class="block-head"><h3>서버별</h3></div><div class="table-wrap"><table class="t"><tr><th>서버</th><th class="num">서브</th><th class="num">1st/2nd</th><th>구질</th><th class="num">평균/최고</th></tr>`
      +servers.map(s=>`<tr><td>${esc(s.label)}</td><td class="num">${s.count}</td><td class="num">${s.first}/${s.second}</td><td>${['flat','slice','kick'].filter(t=>s.types[t]).map(t=>`${SPIN_KO[t]} ${s.types[t]}`).join(' · ')||'-'}</td><td class="num">${s.speed.n?`${s.speed.avg}/${s.speed.max}`:'-'}</td></tr>`).join('')+`</table></div></section>`;
  }
  h+=`<section class="block"><div class="block-head"><h3>타구 ${res.events.length}개</h3><span class="hint">줄을 누르면 그 장면으로</span></div><div class="table-wrap"><table class="t"><tr><th>시각</th><th>포인트</th><th>타구</th><th>누가</th><th>결과</th><th class="num">속도</th></tr>`
    +res.events.map(e=>`<tr class="seek" data-t="${e.t}" data-f="${e.file_id}"><td>${fmtShort(e.t)}</td><td>${e.point_no?'#'+e.point_no:'-'}</td><td>${SHOT_KO[e.shot]}${e.shot==='serve'&&e.serve_type&&e.serve_type!=='unknown'?` <span class="sub">${SPIN_KO[e.serve_type]}</span>`:''}${fixMark(e)}</td><td>${who(e)}</td><td>${RESULT_KO[e.result]||'-'}</td><td class="num">${kmh(e.speed_kmh)}</td></tr>`).join('')
    +`</table></div></section>`;

  const warn=[...(res.measure_warnings||[]),...(res.warnings||[])];
  if(warn.length) h+=`<details class="notes warn"><summary>확인할 점 ${warn.length}개</summary><ul>${warn.map(w=>`<li>${esc(w)}</li>`).join('')}</ul></details>`;
  if(res.observations?.length) h+=`<details class="notes"><summary>코칭 메모 ${res.observations.length}개</summary><ul>${res.observations.map(o=>`<li>${esc(o)}</li>`).join('')}</ul></details>`;
  box.innerHTML=h;
  box.querySelectorAll('tr.seek').forEach(tr=>tr.addEventListener('click',()=>seekTo(tr.dataset.f,Number(tr.dataset.t)-1.0)));
}
function renderServeResult(res){
  lastServeResult=res||null; updateDashButton(); renderTrack();
  const box=$('#serveResult');
  if(!res){ box.innerHTML=''; return; }
  const sv=res.serves||[];
  let h=`<section class="block"><div class="block-head"><h3>서브 실측 ${sv.length}개</h3><span class="hint">공 추적 · ${esc(res.measured_at.replace('T',' ').slice(0,16))}</span></div>`;
  if(res.warnings?.length) h+=`<details class="notes warn"><summary>확인할 점 ${res.warnings.length}개</summary><ul>${res.warnings.map(w=>`<li>${esc(w)}</li>`).join('')}</ul></details>`;
  h+=`<div class="table-wrap"><table class="t"><tr><th>시각</th><th>서버</th><th></th><th class="num">속도</th><th>판정</th><th>방향</th><th>구질</th></tr>`
    +(sv.length?sv.map(s=>{
      const detail=`비행 ${s.flight_s.toFixed(2)}초 · 거리 ${s.distance_m.toFixed(1)}m · 바운스 (${s.bounce_court[0].toFixed(1)}, ${s.bounce_court[1].toFixed(1)})m`;
      const spin=s.serve_type_measured?`${SPIN_KO[s.serve_type_measured]||s.serve_type_measured}${s.serve_type_how==='motion'?`<span class="mark" title="서버 동작으로 판정: 친 뒤 두 발 높이 차 ${s.leg_split} (키 대비) · 이 서버의 정답 기록에서 배운 기준 ${s.serve_type_motion_cut}">동작</span>`:`<span class="mark" title="공이 휜 정도로 판정: 옆 ${s.spin_side_ms2} m/s² · 위 ${s.spin_up_ms2} m/s²">궤적</span>`}`:'<span class="sub">모름</span>';
      return `<tr class="seek" data-t="${s.contact_t}" data-f="${res.file_id}"><td>${fmtShort(s.contact_t)}</td><td>${s.server_end==='near'?'카메라 쪽':'먼 쪽'} <span class="sub">${s.side==='deuce'?'듀스':'애드'}</span></td><td class="sub">${s.serve_number?s.serve_number+(s.serve_number===1?'st':'nd'):''}</td><td class="num" title="${esc(detail)}"><b>${s.speed_kmh}</b> <span class="sub">±${s.speed_err_kmh}</span>${s.contact_src==='audio'?'<span class="mark" title="먼 쪽 서브는 타구음으로 임팩트 시각을 보정했어요">소리</span>':''}</td><td><span class="tag ${s.call}">${CALL_KO[s.call]||s.call}</span>${s.method==='3d'?'<span class="mark" title="바운스 장면을 놓쳐 3D 궤적으로 추정 (오차 큼)">추정</span>':''}</td><td>${DIR_KO[s.direction]||'-'}</td><td>${spin}</td></tr>`;
    }).join(''):`<tr><td colspan="7" class="sub">측정된 서브가 없어요. 코트 위치가 정확한지, 공이 화면에 보이는지 확인하세요.</td></tr>`)
    +`</table></div><p class="footnote">속도는 임팩트부터 첫 바운스까지 비행시간과 코트 거리로 계산한 초속(km/h)이에요. 라인에서 25cm 안에 떨어지면 “라인 근처”로 판정을 미뤄요.</p></section>`;
  box.innerHTML=h;
  box.querySelectorAll('tr.seek').forEach(tr=>tr.addEventListener('click',()=>seekTo(tr.dataset.f,Number(tr.dataset.t)-1.0)));
}
function seekTo(fid,t){
  if(fid && fid!==currentFileId){ currentFileId=fid; $('#fileSelect').value=fid; renderCurrentFile(); }
  if(!fileById(currentFileId)?.proxy){ setStatus('재생용 사본을 먼저 만들어 주세요.',true); return; }
  video.currentTime=Math.max(0,t); video.play();
}
async function loadServeResult(fid){
  try{
    const r=await api(`/api/serves_result/${fid}`); if(fid===currentFileId) renderServeResult(r.result);
    const st=await api(`/api/serve_measure_status/${fid}`); if(['queued','running'].includes(st.job?.status)) waitForServeJob(fid).catch(e=>setStatus(e.message,true));
  }catch(e){ $('#serveResult').innerHTML=''; }
}

/* ── analysis jobs ──────────────────────────────── */
let servePromise=null, shotPromise=null, dashPromise=null;
function updateRunButtons(){
  const busy=!!(servePromise||shotPromise);
  $('#btnAnalyzeAll').disabled=busy; $('#btnStopAll').classList.toggle('hidden',!busy);
  $('#btnServeMeasure').disabled=busy; $('#btnShotAnalyze').disabled=busy;
  $('#btnServeCancel').classList.toggle('hidden',!servePromise); $('#btnShotCancel').classList.toggle('hidden',!shotPromise);
}
function waitForServeJob(fid){
  if(servePromise) return servePromise;
  servePromise=(async()=>{
    try{
      while(true){
        const job=(await api(`/api/serve_measure_status/${fid}`)).job||{};
        if(job.status==='done'){ setBusyProgress(false); setStatus(job.message||'공 추적 완료'); await loadServeResult(fid); shotLoadedFor=null; await loadShotResult(fid); return 'done'; }
        if(job.status==='error'){ setBusyProgress(false); throw new Error(job.error||job.message); }
        if(job.status==='cancelled'||job.status==='idle'){ setBusyProgress(false); setStatus(job.message||'멈췄어요.'); return job.status; }
        jobProgress(job,'공 추적 중');
        await sleep(800);
      }
    }finally{ servePromise=null; updateRunButtons(); }
  })();
  updateRunButtons();
  return servePromise;
}
function waitForShotJob(fid){
  if(shotPromise) return shotPromise;
  shotPromise=(async()=>{
    try{
      while(true){
        const job=(await api(`/api/shot_status/${fid}`)).job||{};
        if(job.status==='done'){ setBusyProgress(false); setStatus(job.message||'타구 판정 완료'); shotLoadedFor=null; if(fid===currentFileId) await loadShotResult(fid); refreshLabelEval(); return 'done'; }
        if(job.status==='error'){ setBusyProgress(false); throw new Error(job.error||job.message||'타구 판정 실패'); }
        if(job.status==='cancelled'){ setBusyProgress(false); setStatus('타구 판정을 멈췄어요.'); return 'cancelled'; }
        if(job.status==='idle'){ setBusyProgress(false); return 'idle'; }
        jobProgress(job,'타구 판정 중');
        await sleep(800);
      }
    }finally{ shotPromise=null; updateRunButtons(); }
  })();
  updateRunButtons();
  return shotPromise;
}
async function startServeMeasure(fid){ showProgress(0,'공 추적 시작 중… (처음 한 번은 모델을 내려받아요)'); await api(`/api/serve_measure/${fid}`,{}); return waitForServeJob(fid); }
function shotNeedsKey(){ return $('#shotMethod').value!=='hits' || $('#shotGemini').checked; }
async function startShotAnalysis(fid){
  if(shotNeedsKey() && !shotConfig?.has_key){ showTab('setup'); $('#shotKeyForm').classList.remove('hidden'); throw new Error('Gemini 키가 필요해요. 준비 탭에서 설정하세요.'); }
  showProgress(0,'타구 판정 시작 중…');
  await api(`/api/shot_analysis/${fid}`,{scope:$('#shotScope').value,quality:$('#shotQuality').value,high_res:$('#shotHighRes').checked,prompt:$('#shotMethod').value,gemini:$('#shotGemini').checked});
  return waitForShotJob(fid);
}
$('#btnAnalyzeAll').onclick=async()=>{
  const fid=currentFileId; if(!fid) return setStatus('먼저 영상을 추가하세요.',true);
  if((fileById(fid)?.calibration?.points?.length||0)<4){ showTab('setup'); return setStatus('코트 위치를 먼저 찍어 주세요 (준비 탭).',true); }
  try{
    const r=await startServeMeasure(fid);
    if(r!=='done') return;
    if(shotNeedsKey() && !shotConfig?.has_key){ setStatus('공 추적은 끝났어요. Gemini 키가 없어 타구 종류 판정은 건너뛰었어요 (준비 탭에서 키 설정).'); return; }
    await startShotAnalysis(fid);
  }catch(e){ clearBusyProgressOnError(e.message); }
};
$('#btnStopAll').onclick=async()=>{
  if(!currentFileId) return;
  try{
    if(servePromise) await api(`/api/serve_measure_cancel/${currentFileId}`,{});
    if(shotPromise) await api(`/api/shot_cancel/${currentFileId}`,{});
  }catch(e){ setStatus(e.message,true); }
};
$('#btnServeMeasure').onclick=async()=>{ if(!currentFileId) return; try{ await startServeMeasure(currentFileId); }catch(e){ clearBusyProgressOnError(e.message); } };
$('#btnShotAnalyze').onclick=async()=>{ if(!currentFileId) return; try{ await startShotAnalysis(currentFileId); }catch(e){ clearBusyProgressOnError(e.message); } };
$('#btnServeCancel').onclick=async()=>{ if(currentFileId){ try{ await api(`/api/serve_measure_cancel/${currentFileId}`,{}); }catch(e){ setStatus(e.message,true); } } };
$('#btnShotCancel').onclick=async()=>{ if(currentFileId){ try{ await api(`/api/shot_cancel/${currentFileId}`,{}); }catch(e){ setStatus(e.message,true); } } };

/* ── exports ────────────────────────────────────── */
function updateDashButton(){ $('#btnDashExport').disabled=!!dashPromise||!(lastShotResult||lastServeResult?.serves); }
function setDashRunning(running){ $('#btnDashCancel').classList.toggle('hidden',!running); updateDashButton(); }
function waitForDashJob(fid){
  if(dashPromise) return dashPromise;
  dashPromise=(async()=>{
    setDashRunning(true);
    try{
      while(true){
        const job=(await api(`/api/export_dashboard_status/${fid}`)).job||{};
        if(job.status==='done'){ setBusyProgress(false); setStatus(job.message||'대시보드 영상 완료'); statusLink(job.url,'영상 열기'); return; }
        if(job.status==='error'){ setBusyProgress(false); throw new Error(job.error||job.message); }
        if(job.status==='cancelled'||job.status==='idle'){ setBusyProgress(false); setStatus(job.message||''); return; }
        jobProgress(job,'대시보드 영상 만드는 중');
        await sleep(800);
      }
    }finally{ dashPromise=null; setDashRunning(false); }
  })();
  return dashPromise;
}
$('#btnDashExport').onclick=async()=>{
  if(!currentFileId) return; const fid=currentFileId;
  try{ showProgress(0,'대시보드 영상 시작 중…'); await api(`/api/export_dashboard/${fid}`,{quality:$('#shotQuality').value,ball:$('#dashBall').checked,pose:$('#dashPose').checked}); await waitForDashJob(fid); }
  catch(e){ clearBusyProgressOnError(e.message); }
};
$('#btnDashCancel').onclick=async()=>{ if(currentFileId){ try{ await api(`/api/export_dashboard_cancel/${currentFileId}`,{}); }catch(e){ setStatus(e.message,true); } } };
async function doExport(path){
  try{ setBusyProgress(true,'영상 만드는 중… 길이에 따라 시간이 걸려요.'); const j=await api(path,{}); await finishBusyProgress('영상을 만들었어요.'); statusLink(j.url,'영상 열기'); }
  catch(e){ clearBusyProgressOnError(e.message); }
}
$('#btnExport').onclick=()=>doExport('/api/export_gameplay');
$('#btnBroadcast').onclick=()=>doExport('/api/export_broadcast');

Promise.all([refresh(),refreshAIStatus(),refreshShotConfig()]).then(()=>{
  if(!project?.files?.length) showTab('setup');
}).catch(e=>setStatus(e.message,true));

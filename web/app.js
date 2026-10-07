// 돌려 보는 라이다 — 화면. 계산은 lidar.wasm(C++), 색·카메라·문장은 core.js.
import createLidar from './lidar.mjs';
import { STAGES, STAGE_NAME, tourStage, colorize, median, perspective, lookAt, mul, orbitEye, describe, budgetText } from './core.js';

const $ = (id) => document.getElementById(id);
const RECORD = { cpu1: 5.55, cpu8: 1.55, gpu: 0.70 };        // README 표(개발 PC, 원본 108장 중앙값 ms)
const FRAME_MS = 1700, TARGET = [6, 0, -1];
const VIEWS = { side: [0.55, 0.42, 58], top: [0, 1.5, 96], near: [0.35, 0.26, 24] };   // yaw, pitch, 거리(m)
const CAR = [-1.6, -0.9, 2.6, -0.9, 2.6, 0.9, -1.6, 0.9].flatMap((v, i) => (i % 2 ? [v, -1.7] : [v]));

const S = {
  M: null, expected: null, frames: [], frame: 0, stage: 'raw', pinned: null, playing: true, spin: true,
  seed: 0, iters: 256, n: 0, stats: null, times: null, recent: [], base: [], dirty: true,
  cam: { yaw: VIEWS.side[0], pitch: VIEWS.side[1], dist: VIEWS.side[2] }, goal: null, nextAt: 0, lastT: 0,
};

// ── 계산 ───────────────────────────────────────────────────────────────────
function runScan(bytes, seed) {
  const M = S.M, n = bytes.byteLength / 6;
  const ptr = M._malloc(bytes.byteLength);
  M.HEAPU16.set(new Uint16Array(bytes.buffer, bytes.byteOffset, n * 3), ptr >> 1);
  M._lidar_run(ptr, n, seed);
  M._free(ptr);
  const s = (k) => M._lidar_stat(k), h = (k) => M._lidar_hash(k) >>> 0;
  return {
    stats: { points: s(0), voxels: s(1), best_hyp: s(2), best_score: s(3), ground: s(4), clusters: s(5), small: s(6),
      h_keys: h(0), h_labels: h(1), h_centroids: h(2) },
    times: { voxel: M._lidar_ms(0), ransac: M._lidar_ms(1), cluster: M._lidar_ms(2), total: M._lidar_ms(3) },
  };
}

async function loadFrame(k) {
  if (!S.frames[k]) {
    S.frames[k] = fetch(`data/${S.expected.frames[k].file}`).then((r) => {
      if (!r.ok) throw new Error(`data ${k}: ${r.status}`);
      return r.arrayBuffer();
    }).then((b) => new Uint8Array(b));
  }
  return S.frames[k];
}

// ── 그리기(WebGL) ──────────────────────────────────────────────────────────
const canvas = $('view');
const gl = canvas.getContext('webgl', { antialias: true, alpha: false });
let prog, bufPos, bufCol, bufCar, loc = {}, colors = new Uint8Array(0);

function initGL() {
  const sh = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, `attribute vec3 aPos; attribute vec3 aCol; uniform mat4 uM; uniform float uSize; uniform vec3 uFlat;
    varying vec3 vCol; void main(){ gl_Position = uM * vec4(aPos, 1.0); gl_PointSize = clamp(uSize / gl_Position.w, 1.5, uSize * 0.12);
    vCol = uFlat.r >= 0.0 ? uFlat : aCol; }`));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, 'precision mediump float; varying vec3 vCol; void main(){ gl_FragColor = vec4(vCol, 1.0); }'));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  for (const k of ['aPos', 'aCol']) loc[k] = gl.getAttribLocation(prog, k);
  for (const k of ['uM', 'uSize', 'uFlat']) loc[k] = gl.getUniformLocation(prog, k);
  bufPos = gl.createBuffer(); bufCol = gl.createBuffer(); bufCar = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, bufCar);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(CAR), gl.STATIC_DRAW);
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(5 / 255, 8 / 255, 11 / 255, 1);
}

/** 단계에 맞는 점(원본·바닥·물체는 점마다, 복셀은 칸마다)을 GPU 버퍼로 올린다. */
function upload() {
  const M = S.M, vox = S.stage === 'voxel';
  const n = vox ? S.stats.voxels : S.stats.points;
  const xyz = new Float32Array(M.HEAPF32.buffer, vox ? M._lidar_vxyz() : M._lidar_xyz(), n * 3);
  const lab = new Int32Array(M.HEAP32.buffer, vox ? M._lidar_vlabel() : M._lidar_plabel(), n);
  if (colors.length < n * 3) colors = new Uint8Array(n * 3);
  colorize(S.stage, xyz, lab, n, colors);
  gl.bindBuffer(gl.ARRAY_BUFFER, bufPos);
  gl.bufferData(gl.ARRAY_BUFFER, xyz, gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, bufCol);
  gl.bufferData(gl.ARRAY_BUFFER, colors.subarray(0, n * 3), gl.DYNAMIC_DRAW);
  S.n = n;
  S.dirty = true;
}

function draw() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  gl.viewport(0, 0, w, h);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  if (!S.n) return;
  const c = S.cam, eye = orbitEye(c.yaw, c.pitch, c.dist, TARGET);
  const up = c.pitch > 1.45 ? [Math.cos(c.yaw), Math.sin(c.yaw), 0] : [0, 0, 1];   // 바로 위에서는 z를 위로 둘 수 없다
  const m = mul(perspective(0.8, w / h, 0.5, 400), lookAt(eye, TARGET, up));
  gl.uniformMatrix4fv(loc.uM, false, new Float32Array(m));
  gl.uniform1f(loc.uSize, (S.stage === 'voxel' ? 110 : 62) * dpr * (h / 900 + 0.45));
  gl.uniform3f(loc.uFlat, -1, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, bufPos);
  gl.enableVertexAttribArray(loc.aPos);
  gl.vertexAttribPointer(loc.aPos, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, bufCol);
  gl.enableVertexAttribArray(loc.aCol);
  gl.vertexAttribPointer(loc.aCol, 3, gl.UNSIGNED_BYTE, true, 0, 0);
  gl.drawArrays(gl.POINTS, 0, S.n);
  gl.disableVertexAttribArray(loc.aCol);                       // 차 윤곽(센서가 실린 자리)
  gl.uniform3f(loc.uFlat, 0.91, 0.93, 0.96);
  gl.bindBuffer(gl.ARRAY_BUFFER, bufCar);
  gl.vertexAttribPointer(loc.aPos, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.LINE_LOOP, 0, 4);
}

// ── 화면 글자 ──────────────────────────────────────────────────────────────
const ms = (v) => (v < 10 ? v.toFixed(1) : v.toFixed(0)) + 'ms';

function paint() {
  const f = S.expected.frames[S.frame], st = { ...S.stats, iters: S.iters };
  const [title, text] = describe(S.stage, st, S.times);
  $('title').textContent = title;
  $('text').textContent = text;
  $('tag').textContent = `장면 ${S.frame + 1}/12 · ${STAGES.indexOf(S.stage) + 1}단계 ${STAGE_NAME[S.stage]} · KITTI ${f.scan}번`;
  $('frame').value = S.frame;
  for (const b of $('chips').children) b.setAttribute('aria-pressed', String(b.dataset.stage === S.stage));
  for (const b of $('iters').children) b.setAttribute('aria-pressed', String(+b.dataset.iters === S.iters));
  $('btnPlay').textContent = S.playing ? '멈춤' : '저절로 돌기';

  const med = (k) => median(S.recent.map((t) => t[k]));
  const total = med('total');
  $('total').textContent = ms(total);
  $('budget').textContent = budgetText(total);
  for (const [k, id] of [['voxel', 'Voxel'], ['ransac', 'Ransac'], ['cluster', 'Cluster']]) {
    $('t' + id).textContent = ms(med(k));
    $('b' + id).style.width = Math.min(100, med(k) / total * 100) + '%';
  }
  const mine = median(S.base), top = Math.max(mine, RECORD.cpu1);   // 기록이 256번 기준이라 이 줄도 256번으로 잰 값만 쓴다
  $('vWasm').textContent = ms(mine);
  $('cWasm').style.width = mine / top * 100 + '%';
  $('cCpu1').style.width = RECORD.cpu1 / top * 100 + '%';
  $('cCpu8').style.width = RECORD.cpu8 / top * 100 + '%';
  $('cGpu').style.width = RECORD.gpu / top * 100 + '%';
}

/** 한 장을 돌리고 그린다. seed를 안 주면 대조 정답과 같은 시드를 쓴다. */
async function show(frame, stage, seed) {
  const k = ((frame % 12) + 12) % 12;
  const bytes = await loadFrame(k);
  S.frame = k;
  S.stage = stage || S.pinned || tourStage(k);
  S.seed = seed ?? S.expected.frames[k].seed;
  S.M._lidar_set_iters(S.iters);
  const r = runScan(bytes, S.seed);
  S.stats = r.stats; S.times = r.times;
  S.recent.push(r.times);
  if (S.recent.length > 12) S.recent.shift();
  if (S.iters === 256) { S.base.push(r.times.total); if (S.base.length > 12) S.base.shift(); }
  upload();
  paint();
  if (S.playing) loadFrame((k + 1) % 12).catch(() => {});      // 다음 장을 미리 받는다
  return r;
}

function pause() { S.playing = false; S.spin = false; paint(); }

// ── 조작 ───────────────────────────────────────────────────────────────────
function bind() {
  $('chips').addEventListener('click', (e) => {
    const st = e.target.dataset?.stage;
    if (!st) return;
    S.pinned = st;
    $('note').textContent = '';
    show(S.frame, st, S.seed);
  });
  $('btnPlay').addEventListener('click', () => {
    S.playing = !S.playing;
    if (S.playing) { S.pinned = null; S.spin = true; S.nextAt = performance.now() + 300; $('note').textContent = ''; }
    paint();
  });
  $('frame').addEventListener('input', (e) => { pause(); $('note').textContent = ''; show(+e.target.value, S.pinned || S.stage); });
  for (const [id, v] of [['btnTop', 'top'], ['btnSide', 'side'], ['btnNear', 'near']])
    $(id).addEventListener('click', () => { S.spin = false; S.goal = VIEWS[v]; });

  const rerun = async (seed, label) => {
    const before = S.stats;
    pause();
    S.pinned = 'object';
    const r = await show(S.frame, 'object', seed);
    const d = (a, b) => `${a.toLocaleString('ko-KR')} → ${b.toLocaleString('ko-KR')}`;
    $('note').innerHTML = `${label} 물체 <b>${d(before.clusters, r.stats.clusters)}덩이</b> · 바닥 ${d(before.ground, r.stats.ground)}칸. 점은 그대로입니다.`;
  };
  $('btnReseed').addEventListener('click', () => rerun((Math.random() * 0xffffffff) >>> 0, '무작위만 바꿨습니다:'));
  $('iters').addEventListener('click', (e) => {
    const n = +e.target.dataset?.iters;
    if (!n || n === S.iters) return;
    S.iters = n;
    S.recent = [];
    rerun(S.seed, `평면을 ${n.toLocaleString('ko-KR')}번 세웠습니다:`);
  });

  // 끌기: 마우스는 좌우·위아래, 손가락은 좌우(위아래는 페이지 스크롤). 두 손가락·휠은 거리.
  const ptr = new Map();
  let pinch = 0;
  const touched = () => { S.spin = false; S.goal = null; $('hint').style.opacity = 0; };
  canvas.addEventListener('pointerdown', (e) => { ptr.set(e.pointerId, [e.clientX, e.clientY]); canvas.setPointerCapture(e.pointerId); pinch = 0; });
  canvas.addEventListener('pointermove', (e) => {
    const p = ptr.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p[0], dy = e.clientY - p[1];
    ptr.set(e.pointerId, [e.clientX, e.clientY]);
    touched();
    if (ptr.size === 2) {
      const [a, b] = [...ptr.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (pinch) S.cam.dist = Math.min(140, Math.max(10, S.cam.dist * pinch / d));
      pinch = d;
    } else {
      S.cam.yaw -= dx * 0.008;
      if (e.pointerType === 'mouse') S.cam.pitch = Math.min(1.5, Math.max(0.08, S.cam.pitch + dy * 0.006));
    }
    S.dirty = true;
  });
  for (const ev of ['pointerup', 'pointercancel']) canvas.addEventListener(ev, (e) => { ptr.delete(e.pointerId); pinch = 0; });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    touched();
    S.cam.dist = Math.min(140, Math.max(10, S.cam.dist * Math.exp(e.deltaY * 0.0012)));
    S.dirty = true;
  }, { passive: false });
  new ResizeObserver(() => { S.dirty = true; }).observe(canvas);
}

function tick(now) {
  const dt = Math.min(0.1, (now - S.lastT) / 1000 || 0);
  S.lastT = now;
  if (S.spin && !reduced) { S.cam.yaw += dt * 0.11; S.dirty = true; }
  if (S.goal) {
    const c = S.cam, [y, p, d] = S.goal, k = 1 - Math.exp(-dt * 6);
    const dy = Math.atan2(Math.sin(y - c.yaw), Math.cos(y - c.yaw));
    c.yaw += dy * k; c.pitch += (p - c.pitch) * k; c.dist += (d - c.dist) * k;
    if (Math.abs(dy) < 0.002 && Math.abs(p - c.pitch) < 0.002 && Math.abs(d - c.dist) < 0.05) S.goal = null;
    S.dirty = true;
  }
  if (S.playing && !busy && now >= S.nextAt) step(now);
  if (S.dirty) { S.dirty = false; draw(); }
  requestAnimationFrame(tick);
}

let busy = false;
async function step(now = performance.now()) {
  busy = true;
  S.nextAt = now + FRAME_MS;
  try { await show(S.frame + 1); } catch (e) { fail(e); } finally { busy = false; }
}

// ── 자체 대조(?selftest=1): 12장을 정답과 같은 시드로 돌려 네이티브 결과와 비교 ──
async function selftest() {
  S.playing = false;
  const keys = ['points', 'voxels', 'best_hyp', 'best_score', 'ground', 'clusters', 'small', 'h_keys', 'h_labels', 'h_centroids'];
  const bad = [], total = [];
  S.M._lidar_set_iters(256);
  for (let k = 0; k < 12; k++) {
    const f = S.expected.frames[k], bytes = await loadFrame(k);
    for (let r = 0; r < 5; r++) {
      const got = runScan(bytes, f.seed);
      if (r === 0) for (const key of keys) if (got.stats[key] !== f[key]) bad.push(`${f.file}.${key}`);
      total.push(got.times.total);
    }
    $('title').textContent = `대조 중 ${k + 1}/12`;
  }
  const res = { ok: bad.length === 0, frames: 12, checks: 12 * keys.length, bad, wasm_ms: +median(total).toFixed(2),
    native_ms: +median(S.expected.frames.map((f) => f.native_ms.total)).toFixed(2) };
  window.__selftest = res;
  S.iters = 256; S.recent = [];
  await show(0, 'object');
  $('note').innerHTML = res.ok
    ? `자체 대조: 12장 × ${keys.length}항목 = ${res.checks}개가 PC에서 돌린 결과와 <b style="color:var(--ok)">전부 같습니다</b>. 이 기기 ${res.wasm_ms}ms / 같은 입력 PC ${res.native_ms}ms.`
    : `자체 대조 <b>불일치 ${bad.length}개</b>: ${bad.slice(0, 6).join(', ')}`;
  return res;
}

function fail(e) {
  console.error(e);
  S.playing = false;
  $('title').textContent = '이 브라우저에서는 열 수 없습니다';
  $('title').classList.add('error');
  $('text').textContent = String(e.message || e);
  $('tag').textContent = '오류';
}

const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

(async () => {
  try {
    if (!gl) throw new Error('WebGL을 쓸 수 없습니다.');
    initGL();
    bind();
    [S.M, S.expected] = await Promise.all([createLidar(), fetch('data/expected.json').then((r) => r.json())]);
    window.__lidar = { S, show, step, selftest, stop: () => { S.playing = false; S.spin = false; }, draw };
    if (new URLSearchParams(location.search).has('selftest')) await selftest();
    else { await show(0); S.nextAt = performance.now() + FRAME_MS; }
    requestAnimationFrame(tick);
  } catch (e) { fail(e); }
})();

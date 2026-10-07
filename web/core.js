// 돌려 보는 라이다 — 순수 계산(색·카메라·문장). node 테스트 대상.

export const STAGES = ['raw', 'voxel', 'ground', 'object'];
export const STAGE_NAME = { raw: '원본 점', voxel: '복셀', ground: '바닥', object: '물체' };

/** 저절로 돌 때 프레임 번호 → 보여 줄 단계. 12장 한 바퀴에 네 단계를 다 지난다. */
export function tourStage(frame) {
  const k = ((frame % 12) + 12) % 12;
  return k < 2 ? 'raw' : k < 4 ? 'voxel' : k < 6 ? 'ground' : 'object';
}

const RAMP = [[38, 70, 140], [40, 150, 190], [120, 205, 150], [245, 215, 100], [255, 250, 235]];
/** 높이(센서 기준 m, 바닥은 약 -1.7) → 색. */
export function heightColor(z) {
  const t = Math.min(0.9999, Math.max(0, (z + 2.2) / 3.4)) * (RAMP.length - 1);
  const i = Math.floor(t), f = t - i, a = RAMP[i], b = RAMP[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

const PALETTE = [[255, 107, 107], [255, 180, 84], [250, 232, 96], [126, 231, 135], [91, 192, 255], [176, 132, 255],
  [255, 138, 208], [64, 224, 208], [255, 153, 102], [173, 255, 47], [135, 176, 255], [255, 99, 172]];
/** 물체 라벨(성분 대표 셀 번호) → 색. 같은 라벨은 늘 같은 색. */
export function labelColor(label) {
  let h = (label * 2654435761) >>> 0;
  h = (h ^ (h >>> 15)) >>> 0;                    // ^ 는 부호 있는 값을 낸다 — 음수면 팔레트 밖을 읽는다
  return PALETTE[h % PALETTE.length];
}

export const GROUND_COLOR = [52, 96, 150], GROUND_DIM = [30, 44, 62], SMALL_COLOR = [92, 104, 120], OFF_GROUND = [255, 180, 84];

/** 단계에 맞춰 점마다 색을 채운다. labels: -2 바닥, -1 작은 조각, 0 이상 물체. */
export function colorize(stage, xyz, labels, n, out) {
  for (let i = 0; i < n; i++) {
    const l = labels[i];
    let c;
    if (stage === 'raw' || stage === 'voxel') c = heightColor(xyz[3 * i + 2]);
    else if (stage === 'ground') c = l === -2 ? GROUND_COLOR : OFF_GROUND;
    else c = l === -2 ? GROUND_DIM : l === -1 ? SMALL_COLOR : labelColor(l);
    out[3 * i] = c[0]; out[3 * i + 1] = c[1]; out[3 * i + 2] = c[2];
  }
  return out;
}

export function median(values) {
  if (!values.length) return NaN;
  const v = [...values].sort((a, b) => a - b);
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

// ── 카메라(열 우선 4×4) ─────────────────────────────────────────────────────
export function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
}

export function lookAt(eye, target, up) {
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const norm = (a) => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  const z = norm(sub(eye, target)), x = norm(cross(up, z)), y = cross(z, x);
  return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
}

export function mul(a, b) {
  const o = new Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++)
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  return o;
}

/** 차 주위를 도는 눈 위치. yaw 0은 차 뒤에서 앞을 본다(KITTI: x 앞, y 왼쪽, z 위). */
export function orbitEye(yaw, pitch, dist, target) {
  const cp = Math.cos(pitch);
  return [target[0] - dist * cp * Math.cos(yaw), target[1] - dist * cp * Math.sin(yaw), target[2] + dist * Math.sin(pitch)];
}

const num = (v) => Math.round(v).toLocaleString('ko-KR');
const ms = (v) => (v < 10 ? v.toFixed(1) : v.toFixed(0)) + 'ms';

/** 단계별 화면 문장. st = {points, voxels, ground, clusters, small, iters}, t = {voxel, ransac, cluster, total}. */
export function describe(stage, st, t) {
  if (stage === 'raw') return [`점 ${num(st.points)}개`, '레이저 스캐너가 한 바퀴(0.1초) 돌며 찍은 점입니다. 색은 높이입니다.'];
  if (stage === 'voxel') return [`${num(st.points)} → ${num(st.voxels)}개`, `20cm 칸마다 한 점으로 줄였습니다. 뒤 계산이 ${(st.points / st.voxels).toFixed(1)}배 가벼워집니다 · ${ms(t.voxel)}`];
  if (stage === 'ground') return [`바닥 ${num(st.ground)}칸`, `점 3개로 평면을 ${num(st.iters)}번 세워 보고 가장 많이 맞는 것을 바닥으로 잡았습니다 · ${ms(t.ransac)}`];
  return [`물체 ${num(st.clusters)}덩이`, `바닥을 빼고 붙어 있는 점끼리 묶었습니다. 회색은 너무 작아 버린 조각 ${num(st.small)}개 · ${ms(t.cluster)}`];
}

/** 한 스캔 예산(10Hz면 100ms)에서 차지하는 비율(%) 문장. */
export function budgetText(totalMs, hz = 10) {
  const pct = totalMs / (1000 / hz) * 100;
  return `스캔 한 장 예산 ${Math.round(1000 / hz)}ms의 ${pct < 10 ? pct.toFixed(1) : pct.toFixed(0)}%`;
}

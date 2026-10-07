import test from 'node:test';
import assert from 'node:assert/strict';
import { tourStage, heightColor, labelColor, colorize, median, perspective, lookAt, mul, orbitEye, describe, budgetText,
  GROUND_COLOR, GROUND_DIM, SMALL_COLOR, OFF_GROUND } from '../core.js';

test('저절로 돌 때 12장 한 바퀴에 네 단계를 차례로 지난다', () => {
  const seq = Array.from({ length: 12 }, (_, k) => tourStage(k));
  assert.deepEqual([...new Set(seq)], ['raw', 'voxel', 'ground', 'object']);
  assert.equal(tourStage(12), 'raw');
  assert.equal(tourStage(-1), 'object');
});

test('높이 색은 범위 밖에서도 0~255 안이고 위로 갈수록 밝아진다', () => {
  for (const z of [-50, -2.2, -1.7, 0, 1.2, 50]) for (const c of heightColor(z)) assert.ok(c >= 0 && c <= 255);
  const sum = (c) => c[0] + c[1] + c[2];
  assert.ok(sum(heightColor(1.0)) > sum(heightColor(-1.7)));
});

test('물체 색은 라벨이 같으면 같다', () => {
  assert.deepEqual(labelColor(12345), labelColor(12345));
  assert.equal(labelColor(0).length, 3);
  for (let l = 0; l < 40000; l++) assert.equal(labelColor(l).length, 3, `label ${l}`);   // 격자 200×200의 모든 셀 번호
});

test('단계별 색: 바닥·작은 조각·물체가 구분된다', () => {
  const xyz = new Float32Array(9), labels = new Int32Array([-2, -1, 777]), out = new Uint8Array(9);
  colorize('ground', xyz, labels, 3, out);
  assert.deepEqual([...out.slice(0, 3)], GROUND_COLOR);
  assert.deepEqual([...out.slice(3, 6)], OFF_GROUND);
  colorize('object', xyz, labels, 3, out);
  assert.deepEqual([...out.slice(0, 3)], GROUND_DIM);
  assert.deepEqual([...out.slice(3, 6)], SMALL_COLOR);
  assert.deepEqual([...out.slice(6, 9)], labelColor(777));
});

test('중앙값', () => {
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 2, 3]), 2.5);
  assert.ok(Number.isNaN(median([])));
});

test('카메라: 바라보는 점이 화면 가운데로 온다', () => {
  const target = [0, 0, -1];
  const eye = orbitEye(0.7, 0.5, 40, target);
  assert.ok(Math.abs(Math.hypot(eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]) - 40) < 1e-9);
  const m = mul(perspective(0.9, 1.5, 0.5, 300), lookAt(eye, target, [0, 0, 1]));
  const p = [0, 1, 2, 3].map((r) => m[r] * target[0] + m[4 + r] * target[1] + m[8 + r] * target[2] + m[12 + r]);
  assert.ok(Math.abs(p[0] / p[3]) < 1e-9 && Math.abs(p[1] / p[3]) < 1e-9);
  assert.ok(p[3] > 0);
});

test('카메라: yaw 0이면 차 뒤(-x)에서 본다', () => {
  const eye = orbitEye(0, 0.3, 10, [0, 0, 0]);
  assert.ok(eye[0] < 0 && Math.abs(eye[1]) < 1e-9 && eye[2] > 0);
});

test('문장은 받은 숫자만 쓴다', () => {
  const st = { points: 116751, voxels: 31359, ground: 16510, clusters: 73, small: 40, iters: 256 };
  const t = { voxel: 6.24, ransac: 2.1, cluster: 0.3, total: 8.64 };
  assert.match(describe('raw', st, t)[0], /116,751/);
  assert.match(describe('voxel', st, t)[1], /3\.7배/);
  assert.match(describe('ground', st, t)[1], /256번/);
  assert.match(describe('object', st, t)[0], /73덩이/);
  assert.equal(budgetText(8.64), '스캔 한 장 예산 100ms의 8.6%');
  assert.equal(budgetText(25), '스캔 한 장 예산 100ms의 25%');
});

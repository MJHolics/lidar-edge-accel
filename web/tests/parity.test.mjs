// WASM으로 낸 같은 C++ 코드가 네이티브(g++ -ffp-contract=off)와 같은 답을 내는지 12장 전부 본다.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import createLidar from '../lidar.mjs';

const dir = fileURLToPath(new URL('../data/', import.meta.url));
const expected = JSON.parse(fs.readFileSync(dir + 'expected.json', 'utf8'));
const M = await createLidar();

export function run(M, bytes, seed) {
  const n = bytes.byteLength / 6;
  const ptr = M._malloc(bytes.byteLength);
  M.HEAPU16.set(new Uint16Array(bytes.buffer, bytes.byteOffset, n * 3), ptr >> 1);
  M._lidar_run(ptr, n, seed);
  M._free(ptr);
  const s = (k) => M._lidar_stat(k), h = (k) => M._lidar_hash(k) >>> 0;
  return { points: s(0), voxels: s(1), best_hyp: s(2), best_score: s(3), ground: s(4), clusters: s(5), small: s(6),
    h_keys: h(0), h_labels: h(1), h_centroids: h(2) };
}

for (const f of expected.frames) {
  test(`${f.file}(scan ${f.scan}): 복셀 키·중심, 최적 가설, 지면 수, 클러스터 라벨 배열이 네이티브와 같다`, () => {
    const got = run(M, fs.readFileSync(dir + f.file), f.seed);
    for (const k of Object.keys(got)) assert.equal(got[k], f[k], k);
  });
}

test('점마다 붙인 라벨 수가 점 수와 같고, 지면 복셀 수가 통계와 맞는다', () => {
  const f = expected.frames[0];
  run(M, fs.readFileSync(dir + f.file), f.seed);
  const vl = new Int32Array(M.HEAP32.buffer, M._lidar_vlabel(), f.voxels);
  assert.equal(vl.filter((v) => v === -2).length, f.ground);
  const pl = new Int32Array(M.HEAP32.buffer, M._lidar_plabel(), f.points);
  assert.ok(pl.every((v) => v >= -2));
  assert.equal(new Set(vl.filter((v) => v >= 0)).size, f.clusters);
});

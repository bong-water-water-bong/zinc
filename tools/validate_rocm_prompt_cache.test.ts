import { expect, test } from 'bun:test';
import { parseArgs, validateResults } from './validate_rocm_prompt_cache.mjs';

function results(hits: number[]) {
  return hits.map((cached, i) => ({
    content_sha256: `response-${i}`,
    usage: { prompt_tokens: 9000, completion_tokens: 8, total_tokens: 9008 },
    cached_prompt_tokens: cached,
  }));
}

test('cache validator rejects changed responses, usage, and stale prefix reuse', () => {
  const control = results([0, 0, 0, 0, 0, 0]);
  const cached = results([0, 8192, 0, 8192, 0, 0]);
  expect(() => validateResults(control, cached)).not.toThrow();
  const changed = structuredClone(cached);
  changed[1]!.content_sha256 = 'different';
  expect(() => validateResults(control, changed)).toThrow('output or usage');
  const undercounted = structuredClone(cached);
  undercounted[1]!.usage.prompt_tokens -= 8192;
  expect(() => validateResults(control, undercounted)).toThrow('output or usage');
  const stale = results([0, 8192, 8192, 8192, 0, 0]);
  expect(() => validateResults(control, stale)).toThrow('invalidation');
  expect(() => validateResults(control.slice(1), cached)).toThrow('Missing');
});

test('model-loading validator requires explicit inputs and a usable context', () => {
  const args = ['--model', 'model.gguf', '--gpu', 'selected-device', '--output', 'fresh-results'];
  expect(parseArgs(args).context).toBe(16384);
  expect(parseArgs(args).kvCache).toBe('f32');
  expect(parseArgs([...args, '--kv-cache', 'q8']).kvCache).toBe('q8');
  expect(() => parseArgs([...args, '--kv-cache', 'q4'])).toThrow();
  expect(() => parseArgs([...args, '--context', '8192'])).toThrow();
  expect(() => parseArgs([...args, '--mtp', 'invalid'])).toThrow();
  expect(() => parseArgs([...args, '--model'])).toThrow();
  expect(() => parseArgs(['--gpu', 'selected-device', '--output', 'fresh-results'])).toThrow();
});

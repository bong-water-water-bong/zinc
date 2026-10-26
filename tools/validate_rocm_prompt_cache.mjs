#!/usr/bin/env bun
// Run on the Linux GPU test host. Owns one model server at a time.
import { spawn, execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import net from 'node:net';
import path from 'node:path';
import { once } from 'node:events';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const hash = (value) => createHash('sha256').update(value).digest('hex');

export function parseArgs(argv) {
  const opts = { zinc: './zig-out/bin/zinc', context: 16384, mtp: '1', kvCache: 'f32' };
  for (let i = 0; i < argv.length; i++) {
    const key = {
      '--zinc': 'zinc',
      '--model': 'model',
      '--gpu': 'gpu',
      '--output': 'output',
      '--context': 'context',
      '--mtp': 'mtp',
      '--kv-cache': 'kvCache',
    }[argv[i]];
    if (!key || !argv[i + 1] || argv[i + 1].startsWith('--'))
      throw new Error(
        'Use --model FILE --gpu DEVICE --output DIR [--zinc BIN --context N --mtp 0|1 --kv-cache f32|q8]',
      );
    opts[key] = argv[++i];
  }
  opts.gpu ??= process.env.ZINC_GPU;
  opts.context = Number(opts.context);
  if (
    !opts.model ||
    !opts.gpu ||
    !opts.output ||
    !Number.isSafeInteger(opts.context) ||
    opts.context < 16384 ||
    !['0', '1'].includes(opts.mtp) ||
    !['f32', 'q8'].includes(opts.kvCache)
  )
    throw new Error(
      'A model, GPU, fresh output directory, context >= 16384, MTP 0|1, and KV f32|q8 are required',
    );
  return opts;
}

export function validateResults(control, cached) {
  if (control.length !== 6 || cached.length !== 6) throw new Error('Missing validation cases');
  const expectedHits = [0, 8192, 0, 8192, 0, 0];
  for (let i = 0; i < control.length; i++) {
    if (control[i].cached_prompt_tokens !== 0 || cached[i].cached_prompt_tokens !== expectedHits[i])
      throw new Error('Cache hit/invalidation accounting failed');
    if (
      control[i].content_sha256 !== cached[i].content_sha256 ||
      JSON.stringify(control[i].usage) !== JSON.stringify(cached[i].usage)
    )
      throw new Error('Greedy output or usage changed with caching');
  }
}

function body(padding, key) {
  const content = `Remember the exact answer key ${key}. The padding is irrelevant.\n${' alpha'.repeat(padding)}\nReturn the answer key exactly as written, preserving every character, and nothing else.`;
  return {
    messages: [{ role: 'user', content }],
    max_tokens: 20,
    stream: false,
    temperature: 0,
    enable_thinking: false,
  };
}

export async function validate(opts) {
  if (process.platform !== 'linux')
    throw new Error('Run this model-loading validator on the Linux GPU test host');
  const output = path.resolve(opts.output);
  await mkdir(output, { mode: 0o700 }); // fresh directory; never overwrite an earlier run
  const binary = path.resolve(opts.zinc);
  const model = path.resolve(opts.model);
  const probe = net.createServer();
  probe.listen(0, '127.0.0.1');
  await once(probe, 'listening');
  const port = probe.address().port;
  await new Promise((resolve, reject) =>
    probe.close((error) => (error ? reject(error) : resolve())),
  );
  const base = `http://127.0.0.1:${port}`;
  const report = {
    model: path.basename(model),
    context: opts.context,
    mtp: opts.mtp === '1',
    kv_cache: opts.kvCache,
    q8_attention_blas: process.env.ZINC_Q8_ATTN_BLAS ?? '1',
    q8_attention_f16: process.env.ZINC_Q8_ATTN_F16 ?? '0',
    binary_sha256: hash(await readFile(binary)),
    variants: [],
  };
  let server;
  let serverClosed;
  let interrupted = false;
  async function stop() {
    if (!server) return;
    const child = server;
    if (child.exitCode === null && child.signalCode === null) child.kill('SIGTERM');
    let timer;
    await Promise.race([
      serverClosed,
      new Promise((resolve) => {
        timer = setTimeout(resolve, 15000);
      }),
    ]);
    clearTimeout(timer);
    if (child.exitCode === null && child.signalCode === null) {
      child.kill('SIGKILL');
      await serverClosed;
    }
  }
  const onSignal = () => {
    interrupted = true;
    void stop();
  };
  process.on('SIGINT', onSignal);
  process.on('SIGTERM', onSignal);
  async function call(route, payload, timeout = 600000) {
    if (interrupted) throw new Error('Interrupted');
    const response = await fetch(base + route, {
      method: payload ? 'POST' : 'GET',
      headers: { 'content-type': 'application/json' },
      body: payload ? JSON.stringify(payload) : undefined,
      signal: AbortSignal.timeout(timeout),
    });
    if (!response.ok) throw new Error(`API check failed with HTTP ${response.status}`);
    return response.json();
  }
  const cases = [
    ['cold', body(8500, 'MANGO_7391'), 'MANGO_7391'],
    ['shared prefix', body(8700, 'MANGO_7391'), 'MANGO_7391'],
    ['changed prefix', body(8700, 'BERRY_5826'), 'BERRY_5826'],
    ['second shared prefix', body(8800, 'BERRY_5826'), 'BERRY_5826'],
    [
      'short reset',
      {
        messages: [{ role: 'user', content: 'Reply with OK.' }],
        max_tokens: 8,
        stream: false,
        temperature: 0,
        enable_thinking: false,
      },
      'OK',
    ],
    ['long after reset', body(8800, 'BERRY_5826'), 'BERRY_5826'],
  ];
  try {
    for (const enabled of [false, true]) {
      if (interrupted) throw new Error('Interrupted');
      for (const name of ['zinc', 'llama-server', 'llama-cli', 'llama-bench']) {
        try {
          execFileSync('pgrep', ['-x', name], { stdio: 'ignore' });
          throw new Error('A GPU process is already running');
        } catch (error) {
          if (error.status !== 1)
            throw new Error('GPU preflight failed; inspect existing workloads');
        }
      }
      const logFile = Bun.file(path.join(output, `server-${Number(enabled)}.log`));
      const log = logFile.writer();
      server = spawn(
        binary,
        ['-m', model, '-d', '0', '-c', String(opts.context), '-p', String(port), '--parallel', '1'],
        {
          env: {
            ...process.env,
            ROCR_VISIBLE_DEVICES: opts.gpu,
            ZINC_MTP: opts.mtp,
            ZINC_KV_CACHE: opts.kvCache,
            ZINC_PROMPT_CACHE: enabled ? '1' : '0',
            ZINC_PREFILL_CHUNK: '4096',
            ZINC_BATCHED_PREFILL: '1',
          },
          stdio: ['ignore', 'pipe', 'pipe'],
        },
      );
      const child = server;
      serverClosed = once(child, 'close').catch(() => {});
      const writeLog = (chunk) => {
        try {
          log.write(chunk);
        } catch {
          interrupted = true;
          void stop();
        }
      };
      child.stdout.on('data', writeLog);
      child.stderr.on('data', writeLog);
      const entry = { cache_enabled: enabled, cases: [] };
      report.variants.push(entry);
      try {
        let healthy = false;
        for (let i = 0; i < 240; i++) {
          if (interrupted || !child.pid || child.exitCode !== null || child.signalCode !== null)
            throw new Error('Server exited during startup');
          try {
            healthy = (await call('/health', undefined, 2000)).context_length === opts.context;
          } catch {
            /* startup */
          }
          if (healthy) break;
          await sleep(1000);
        }
        if (!healthy) throw new Error('Server startup check failed');
        for (const [name, payload, expected] of cases) {
          const before = await call('/stats');
          const started = performance.now();
          const response = await call('/v1/chat/completions', payload);
          const seconds = (performance.now() - started) / 1000;
          const after = await call('/stats');
          const content = response.choices?.[0]?.message?.content;
          if (content?.trim() !== expected) throw new Error('Long-context response check failed');
          const computed = after.prefill_tokens - before.prefill_tokens;
          entry.cases.push({
            name,
            wall_seconds: seconds,
            usage: response.usage,
            computed_prompt_tokens: computed,
            cached_prompt_tokens: response.usage.prompt_tokens - computed,
            content_sha256: hash(content),
          });
          console.log(`${enabled ? 'cache' : 'control'}: ${name} passed`);
        }
      } finally {
        await stop();
        await log.end();
        entry.owned_server_stopped =
          !child.pid || child.exitCode !== null || child.signalCode !== null;
      }
      const diagnostics = await logFile.text();
      if (
        /\[rocm_shim\].*(?:error|failed|invalid|exceed)|illegal memory|Tensile host|panic/i.test(
          diagnostics,
        )
      )
        throw new Error('GPU diagnostics failed');
    }
    validateResults(report.variants[0].cases, report.variants[1].cases);
    report.status = 'passed';
    return report;
  } catch (error) {
    report.status = 'failed';
    throw error;
  } finally {
    await stop();
    process.off('SIGINT', onSignal);
    process.off('SIGTERM', onSignal);
    await writeFile(path.join(output, 'result.json'), JSON.stringify(report, null, 2) + '\n', {
      mode: 0o600,
    });
  }
}

if (import.meta.main) {
  try {
    if (!globalThis.Bun || !Bun.semver.satisfies(Bun.version, '>=1.4.4'))
      throw new Error('Bun 1.4.4 or newer is required');
    const report = await validate(parseArgs(process.argv.slice(2)));
    console.log(
      JSON.stringify({ status: report.status, cases: 12, exact_greedy_and_usage_parity: true }),
    );
  } catch {
    console.error(
      'Prompt-cache validation failed; inspect the private server logs and result.json. Run with --model FILE --gpu DEVICE --output FRESH_DIR.',
    );
    process.exitCode = 1;
  }
}

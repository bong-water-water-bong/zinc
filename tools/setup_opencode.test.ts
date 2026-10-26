import { afterEach, describe, expect, test } from 'bun:test';
import { mkdtemp, readFile, readdir, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { EventEmitter } from 'node:events';
import {
  buildConfig,
  checkServer,
  CONFIG_NAME,
  parseArgs,
  runOpenCode,
  runVerification,
  runVerifiedOpenCode,
  runtimeConfig,
  writeConfig,
} from './setup_opencode.mjs';

const dirs: string[] = [];
afterEach(async () => {
  for (const dir of dirs.splice(0)) await rm(dir, { recursive: true, force: true });
});
async function temp() {
  const dir = await mkdtemp(path.join(tmpdir(), 'zinc-opencode-setup-'));
  dirs.push(dir);
  return dir;
}

describe('portable OpenCode setup', () => {
  test('has only the local provider, bounded context, and no captured environment', () => {
    const config = buildConfig();
    expect(config.enabled_providers).toEqual(['zinc']);
    expect(config.small_model).toBe(config.model);
    expect(config.provider.zinc.options).toEqual({
      baseURL: 'http://127.0.0.1:9090/v1',
      apiKey: '{env:ZINC_API_KEY}',
    });
    expect(config.agent.zinc.steps).toBe(24);
    expect(config.provider.zinc.models['qwen38-27b-q4k-m'].limit.output).toBe(4096);
    expect(
      buildConfig({ context: 8192 }).provider.zinc.models['qwen38-27b-q4k-m'].limit.output,
    ).toBe(2048);
    expect(config.compaction.reserved).toBe(4096);
    expect(config.compaction.reserved).toBeLessThan(
      config.provider.zinc.models['qwen38-27b-q4k-m'].limit.context,
    );
    expect(JSON.stringify(config)).not.toContain(process.cwd());
  });
  test('parses task text without interpreting it as options', () => {
    const opts = parseArgs([
      'run',
      '--project',
      'project with spaces',
      '--auto',
      '--',
      'explain --help',
    ]);
    expect(opts.prompt).toEqual(['explain --help']);
    expect(opts.auto).toBe(true);
    expect(() => parseArgs(['--misspelled'])).toThrow('Unknown option');
    expect(() => parseArgs(['--model'])).toThrow('Missing value');
  });
  test('rejects invalid model IDs and memory budgets', () => {
    for (const model of [
      '../model',
      '/model.gguf',
      'https://example.com/model',
      '{env:SECRET}',
      'a\nb',
    ])
      expect(() => buildConfig({ model })).toThrow();
    for (const context of [NaN, -1, 4096, 8192.5]) expect(() => buildConfig({ context })).toThrow();
    for (const output of [0, NaN, -1, 16384]) expect(() => buildConfig({ output })).toThrow();
  });
  test('writes a standalone config, preserves existing files, and refuses symlinks', async () => {
    const dir = await temp();
    await writeFile(path.join(dir, 'opencode.json'), '{"model":"existing/model"}\n');
    await writeConfig(dir, buildConfig());
    expect(await readFile(path.join(dir, 'opencode.json'), 'utf8')).toBe(
      '{"model":"existing/model"}\n',
    );
    await expect(writeConfig(dir, buildConfig())).rejects.toThrow();
    await rm(path.join(dir, CONFIG_NAME));
    const target = path.join(dir, 'do-not-overwrite');
    await writeFile(target, 'keep');
    await symlink(target, path.join(dir, CONFIG_NAME));
    await expect(writeConfig(dir, buildConfig(), { force: true })).rejects.toThrow();
    expect(await readFile(target, 'utf8')).toBe('keep');
  });
  test('runtime endpoints and keys never enter the portable config', async () => {
    const config = buildConfig();
    const env = {
      ZINC_API_BASE_URL: 'https://example.invalid/api/v1/',
      ZINC_API_KEY: 'test-only-key',
    };
    const live = runtimeConfig(config, env);
    expect(live.provider.zinc.options.baseURL).toBe('https://example.invalid/api/v1');
    expect(JSON.stringify(live)).not.toContain(env.ZINC_API_KEY);
    const dir = await temp();
    await writeConfig(dir, config);
    const saved = await readFile(path.join(dir, CONFIG_NAME), 'utf8');
    expect(saved).not.toContain('example.invalid');
    expect(saved).not.toContain(env.ZINC_API_KEY);
    expect(saved).not.toContain(dir);
  });
  test('rejects credential-bearing URLs and malformed endpoints', () => {
    for (const url of [
      'file:///v1',
      'https://user:password@example.invalid/v1',
      'http://localhost/v1?token=x',
      'http://localhost/v1#x',
      'http://localhost/chat',
      'http://localhost/not-v1',
      'not a URL',
    ]) {
      expect(() => runtimeConfig(buildConfig(), { ZINC_API_BASE_URL: url })).toThrow(
        'ZINC_API_BASE_URL',
      );
    }
  });
  test('headless execution forwards literal task text and the child exit status', async () => {
    const dir = await temp();
    let captured: any;
    const status = await runOpenCode(
      buildConfig(),
      { project: dir, auto: true, prompt: ['Fix $(literal) `text`'] },
      {
        env: { OPENCODE_BIN: 'custom-opencode', ZINC_API_KEY: 'test-only-key' },
        spawnImpl: ((bin: string, args: string[], options: any) => {
          captured = { bin, args, options };
          const child = new EventEmitter();
          queueMicrotask(() => child.emit('close', 7, null));
          return child;
        }) as any,
      },
    );
    expect(status).toBe(7);
    expect(captured.bin).toBe('custom-opencode');
    expect(captured.args.at(-1)).toBe('Fix $(literal) `text`');
    expect(captured.args).toContain('--auto');
    expect(captured.args).toContain('--pure');
    expect(captured.options.shell).toBeUndefined();
    expect(captured.options.env.OPENCODE_CONFIG_CONTENT).not.toContain('test-only-key');
  });
  test('a timed-out agent returns 124 and receives a termination signal', async () => {
    const dir = await temp();
    const signals: string[] = [];
    const code = await runOpenCode(
      buildConfig(),
      { project: dir, prompt: ['task'], timeoutMs: 20 },
      {
        env: {},
        spawnImpl: (() => {
          const child: any = new EventEmitter();
          child.kill = (signal: string) => {
            signals.push(signal);
            queueMicrotask(() => child.emit('close', null, signal));
          };
          return child;
        }) as any,
      },
    );
    expect(code).toBe(124);
    expect(signals).toEqual(['SIGTERM']);
  });
  test('JSON error events fail a task even when OpenCode exits zero', async () => {
    const dir = await temp();
    let output = '';
    const code = await runOpenCode(
      buildConfig(),
      { project: dir, prompt: ['task'], session: 'ses_example' },
      {
        env: {},
        stdout: {
          write: (chunk) => {
            output += chunk.toString();
          },
        } as any,
        spawnImpl: ((_bin, args) => {
          expect(args).toContain('ses_example');
          const child: any = new EventEmitter();
          child.stdout = new EventEmitter();
          queueMicrotask(() => {
            child.stdout.emit(
              'data',
              Buffer.from('{"type":"error","error":{"name":"APIError"}}\n'),
            );
            child.emit('close', 0, null);
          });
          return child;
        }) as any,
      },
    );
    expect(code).toBe(1);
    expect(output).toContain('APIError');
  });
});

describe('independent verification', () => {
  const quiet = { write: () => {} } as any;

  test('feeds a failing check into the same session and accepts only a passing check', async () => {
    const config = buildConfig({ steps: 64 });
    const before = JSON.stringify(config);
    const rounds: any[] = [];
    let checks = 0;
    const code = await runVerifiedOpenCode(
      config,
      {
        project: '.',
        prompt: ['Implement the feature'],
        verify: 'project-tests',
        attempts: 3,
      },
      {
        stderr: quiet,
        runModel: async (live, opts, deps) => {
          rounds.push({ live, opts });
          deps.onEvent({ sessionID: 'ses_verified' });
          return 0;
        },
        runCheck: async (command) => {
          expect(command).toBe('project-tests');
          checks++;
          return checks === 1
            ? { code: 1, output: 'compiler: missing route implementation' }
            : { code: 0, output: 'tests passed' };
        },
      },
    );
    expect(code).toBe(0);
    expect(checks).toBe(2);
    expect(rounds).toHaveLength(2);
    expect(rounds[0].live.agent.zinc.steps).toBe(8);
    expect(rounds[1].opts.session).toBe('ses_verified');
    expect(rounds[1].opts.prompt[0]).toContain('compiler: missing route implementation');
    expect(rounds[1].opts.prompt[0]).toContain('Implement the feature');
    expect(JSON.stringify(config)).toBe(before);
  });

  test('never turns failed checks or model errors into success', async () => {
    const opts = { project: '.', prompt: ['task'], verify: 'check', attempts: 2 };
    let checks = 0;
    const failing = await runVerifiedOpenCode(buildConfig(), opts, {
      stderr: quiet,
      runModel: async (_config, _opts, deps) => {
        deps.onEvent({ sessionID: 'ses_failed' });
        return 0;
      },
      runCheck: async () => {
        checks++;
        return { code: 7, output: 'failed assertion' };
      },
    });
    expect(failing).toBe(7);
    expect(checks).toBe(2);
    const modelError = await runVerifiedOpenCode(buildConfig(), opts, {
      stderr: quiet,
      runModel: async () => 1,
      runCheck: async () => {
        throw new Error('verification must not mask an API error');
      },
    });
    expect(modelError).toBe(1);
  });

  test('reserves verification time and checks partial work after an editing timeout', async () => {
    const rounds: any[] = [];
    let checks = 0;
    let progress = '';
    const code = await runVerifiedOpenCode(
      buildConfig(),
      {
        project: '.',
        prompt: ['Implement the feature'],
        verify: 'acceptance-check',
        timeoutMs: 10000,
        roundTimeoutMs: 10000,
      },
      {
        stderr: {
          write: (text) => {
            progress += text;
          },
        } as any,
        runModel: async (_config, opts, deps) => {
          rounds.push(opts);
          deps.onEvent({ sessionID: 'ses_timed_round' });
          return rounds.length === 1 ? 124 : 0;
        },
        runCheck: async (_command, opts) => {
          expect(opts.timeoutMs).toBeGreaterThan(0);
          return ++checks === 1
            ? { code: 7, output: 'error: missing implementation' }
            : { code: 0, output: 'accepted' };
        },
      },
    );
    expect(code).toBe(0);
    expect(checks).toBe(2);
    expect(rounds[0].timeoutMs).toBeLessThanOrEqual(7500);
    expect(rounds[0].timeoutMs).toBeGreaterThan(0);
    expect(rounds[1].session).toBe('ses_timed_round');
    expect(rounds[1].prompt[0]).toContain('reached its time limit');
    expect(rounds[1].prompt[0]).toContain('missing implementation');
    expect(progress).toContain('timed out; checking partial work');
    expect(progress).toContain('Verification round 2/3: passed');
  });

  test('bounds individual rounds and cannot accept failed or cancelled verification', async () => {
    for (const [checkCode, expected] of [
      [7, 124],
      [124, 124],
      [130, 130],
      [143, 143],
    ]) {
      let checks = 0;
      const code = await runVerifiedOpenCode(
        buildConfig(),
        {
          project: '.',
          prompt: ['task'],
          verify: 'check',
          attempts: 1,
          timeoutMs: 10000,
          roundTimeoutMs: 300,
        },
        {
          stderr: quiet,
          runModel: async (_config, opts, deps) => {
            expect(opts.timeoutMs).toBe(300);
            deps.onEvent({ sessionID: 'ses_timed_round' });
            return 124;
          },
          runCheck: async () => {
            checks++;
            return { code: checkCode, output: 'failed' };
          },
        },
      );
      expect(code).toBe(expected);
      expect(checks).toBe(1);
    }
    for (const code of [1, 130, 143]) {
      expect(
        await runVerifiedOpenCode(
          buildConfig(),
          { project: '.', prompt: ['task'], verify: 'check' },
          {
            stderr: quiet,
            runModel: async () => code,
            runCheck: async () => {
              throw new Error('must not verify an API error or cancelled run');
            },
          },
        ),
      ).toBe(code);
    }
  });

  test('uses a five-minute default and rejects invalid round deadlines', async () => {
    expect(
      parseArgs(['run', '--round-timeout-ms', '120000', '--verify', 'check']).roundTimeoutMs,
    ).toBe(120000);
    const code = await runVerifiedOpenCode(
      buildConfig(),
      { project: '.', prompt: ['task'], verify: 'check', timeoutMs: 900000 },
      {
        stderr: quiet,
        runModel: async (_config, opts) => {
          expect(opts.timeoutMs).toBe(300000);
          return 1;
        },
      },
    );
    expect(code).toBe(1);
    for (const roundTimeoutMs of [0, -1, NaN, 1.5, Infinity])
      await expect(
        runVerifiedOpenCode(buildConfig(), {
          project: '.',
          prompt: ['task'],
          verify: 'check',
          roundTimeoutMs,
        }),
      ).rejects.toThrow('Round timeout');
  });

  test('CLI rejects round deadlines without verification before writing any files', async () => {
    const dir = await temp();
    for (const args of [['setup'], ['check'], ['run']]) {
      const child = Bun.spawnSync([
        process.execPath,
        path.join(import.meta.dir, 'setup_opencode.mjs'),
        ...args,
        '--project',
        path.join(dir, 'uncreated'),
        '--round-timeout-ms',
        '100',
      ]);
      expect(child.exitCode).toBe(1);
      expect(child.stderr.toString()).toMatch(
        /Verification options|Round timeout requires --verify/,
      );
      expect(await readdir(dir)).toEqual([]);
    }
  });

  test.skipIf(process.platform === 'win32')(
    'checks a real partial edit after timeout and stops background tools',
    async () => {
      const dir = await temp();
      const fixture = path.join(dir, 'opencode-fixture');
      await writeFile(
        fixture,
        `#!${process.execPath}
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
const child = spawn(process.execPath, ['-e', 'process.on("SIGTERM", () => {}); setInterval(() => {}, 1000);'], { stdio: 'ignore' });
writeFileSync('tool.pid', String(child.pid));
writeFileSync('feature.mjs', 'export const answer = 42;\\n');
console.log(JSON.stringify({ type: 'step_start', sessionID: 'ses_process_fixture' }));
process.on('SIGTERM', () => process.exit(0));
setInterval(() => {}, 1000);
`,
        { mode: 0o700 },
      );
      let toolPid: number | undefined;
      try {
        const code = await runVerifiedOpenCode(
          buildConfig(),
          {
            project: dir,
            prompt: ['task'],
            timeoutMs: 10000,
            roundTimeoutMs: 1000,
            verify: `'${process.execPath.replaceAll("'", "'\\''")}' -e 'import { answer } from "./feature.mjs"; if (answer !== 42) process.exit(7);'`,
          },
          { env: { ...process.env, OPENCODE_BIN: fixture }, stdout: quiet, stderr: quiet },
        );
        expect(code).toBe(0);
        toolPid = Number(await readFile(path.join(dir, 'tool.pid'), 'utf8'));
        expect(toolPid).toBeGreaterThan(0);
        // A reparented Linux zombie is already stopped and may await init's reap.
        const status = Bun.spawnSync(['ps', '-p', String(toolPid), '-o', 'stat=']);
        const state = status.stdout.toString().trim();
        expect(state === '' || state.startsWith('Z')).toBe(true);
      } finally {
        if (toolPid) {
          try {
            process.kill(toolPid, 'SIGKILL');
          } catch {}
        }
      }
    },
  );

  test('cancellation and API errors take priority over an editing timeout', async () => {
    for (const signal of ['SIGINT', 'SIGTERM'] as const) {
      const spawnImpl = (() => {
        const child: any = new EventEmitter();
        let interrupted = false;
        child.kill = () => {
          if (!interrupted) {
            interrupted = true;
            process.emit(signal);
          }
          queueMicrotask(() => child.emit('close', 0, null));
        };
        return child;
      }) as any;
      expect(
        await runOpenCode(
          buildConfig(),
          { project: '.', prompt: ['task'], timeoutMs: 10 },
          { spawnImpl, env: {}, stdout: quiet },
        ),
      ).toBe(signal === 'SIGINT' ? 130 : 143);
      expect(
        (
          await runVerification(
            'check',
            { project: '.', timeoutMs: 10 },
            { spawnImpl, stderr: quiet },
          )
        ).code,
      ).toBe(signal === 'SIGINT' ? 130 : 143);
    }
    const code = await runOpenCode(
      buildConfig(),
      { project: '.', prompt: ['task'], timeoutMs: 10 },
      {
        env: {},
        stdout: quiet,
        spawnImpl: (() => {
          const child: any = new EventEmitter();
          child.stdout = new EventEmitter();
          child.kill = () => {
            child.stdout.emit('data', Buffer.from('{"type":"error"}\n'));
            queueMicrotask(() => child.emit('close', 0, null));
          };
          return child;
        }) as any,
      },
    );
    expect(code).toBe(1);
  });

  test.skipIf(process.platform === 'win32')(
    'a successful verifier cannot leave background processes running',
    async () => {
      const dir = await temp();
      await writeFile(
        path.join(dir, 'check.mjs'),
        `
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
const child = spawn(process.execPath, ['-e', 'process.on("SIGTERM", () => {}); setInterval(() => {}, 1000);'], { stdio: 'ignore' });
writeFileSync('tool.pid', String(child.pid));
process.exit(0);
`,
      );
      let toolPid: number | undefined;
      try {
        const checked = await runVerification(
          `'${process.execPath.replaceAll("'", "'\\''")}' check.mjs`,
          { project: dir, timeoutMs: 3000 },
          { stderr: quiet },
        );
        toolPid = Number(await readFile(path.join(dir, 'tool.pid'), 'utf8'));
        expect(checked.code).toBe(0);
        const status = Bun.spawnSync(['ps', '-p', String(toolPid), '-o', 'stat=']);
        const state = status.stdout.toString().trim();
        expect(state === '' || state.startsWith('Z')).toBe(true);
      } finally {
        if (toolPid) {
          try {
            process.kill(toolPid, 'SIGKILL');
          } catch {}
        }
      }
    },
  );

  test('captures real command failure output without running OpenCode', async () => {
    const dir = await temp();
    const checked = await runVerification(
      'printf "compiler diagnostic"; exit 7',
      { project: dir, timeoutMs: 5000 },
      { stderr: quiet },
    );
    expect(checked.code).toBe(7);
    expect(checked.output).toBe('compiler diagnostic');
  });

  test('bounds feedback while retaining the first diagnostic and final summary', async () => {
    const checked = await runVerification(
      'literal command',
      { project: '.', timeoutMs: 1000 },
      {
        stderr: quiet,
        spawnImpl: (() => {
          const child: any = new EventEmitter();
          child.stdout = new EventEmitter();
          child.stderr = new EventEmitter();
          queueMicrotask(() => {
            child.stderr.emit(
              'data',
              Buffer.from('FIRST ERROR\n' + 'x'.repeat(20000) + '\nFINAL SUMMARY'),
            );
            child.emit('close', 1, null);
          });
          return child;
        }) as any,
      },
    );
    expect(checked.output).toStartWith('FIRST ERROR');
    expect(checked.output).toEndWith('FINAL SUMMARY');
    expect(checked.output.length).toBeLessThan(8100);
  });

  test('terminates a timed-out verifier and preserves timeout exit status', async () => {
    const signals: string[] = [];
    const checked = await runVerification(
      'checks',
      { project: '.', timeoutMs: 10 },
      {
        stderr: quiet,
        spawnImpl: (() => {
          const child: any = new EventEmitter();
          child.kill = (signal: string) => {
            signals.push(signal);
            queueMicrotask(() => child.emit('close', null, signal));
          };
          return child;
        }) as any,
      },
    );
    expect(checked.code).toBe(124);
    expect(signals).toEqual(['SIGTERM']);
    expect(
      parseArgs(['run', '--verify', 'bun test', '--attempts', '2', '--', 'task']).attempts,
    ).toBe(2);
  });

  test('retains a failure in the middle of a long passing-test log', async () => {
    const checked = await runVerification(
      'project-tests',
      { project: '.', timeoutMs: 1000 },
      {
        stderr: quiet,
        spawnImpl: (() => {
          const child: any = new EventEmitter();
          child.stderr = new EventEmitter();
          queueMicrotask(() => {
            child.stderr.emit('data', Buffer.from('(pass) unrelated test\n'.repeat(1000)));
            child.stderr.emit(
              'data',
              Buffer.from('module.zig:9: error: missing module documentation\n'),
            );
            child.stderr.emit('data', Buffer.from('expected: a module description\n'));
            child.stderr.emit('data', Buffer.from('(pass) later unrelated test\n'.repeat(1000)));
            child.emit('close', 1, null);
          });
          return child;
        }) as any,
      },
    );
    expect(checked.code).toBe(1);
    expect(checked.output).toContain('module.zig:9: error: missing module documentation');
    expect(checked.output).toContain('expected: a module description');
    expect(checked.output.length).toBeLessThan(8100);
  });

  test('an interrupted child exiting zero cannot produce success', async () => {
    for (const [signal, expected] of [
      ['SIGINT', 130],
      ['SIGTERM', 143],
    ] as const) {
      const spawnImpl = (() => {
        const child: any = new EventEmitter();
        child.kill = () => queueMicrotask(() => child.emit('close', 0, null));
        queueMicrotask(() => process.emit(signal));
        return child;
      }) as any;
      const checked = await runVerification(
        'checks',
        { project: '.', timeoutMs: 1000 },
        {
          spawnImpl,
          stderr: quiet,
        },
      );
      expect(checked.code).toBe(expected);
      const code = await runOpenCode(
        buildConfig(),
        {
          project: '.',
          prompt: ['task'],
          timeoutMs: 1000,
        },
        { spawnImpl, stdout: quiet, env: {} },
      );
      expect(code).toBe(expected);
    }
  });
});

describe('connection checks', () => {
  test('checks JSON and streaming tool completions rather than health alone', async () => {
    const seen: any[] = [];
    const tool = {
      id: 'call_probe',
      type: 'function',
      function: { name: 'zinc_probe', arguments: '{"message":"OK"}' },
    };
    const result = await checkServer(buildConfig(), {
      env: {},
      fetchImpl: (async (_url: string, opts: any) => {
        const body = opts.body ? JSON.parse(opts.body) : null;
        seen.push(body);
        if (!body) return Response.json({ status: 'ok' });
        if (!body.tools)
          return Response.json({
            choices: [{ message: { content: 'OK' }, finish_reason: 'stop' }],
          });
        if (!body.stream)
          return Response.json({
            choices: [{ message: { tool_calls: [tool] }, finish_reason: 'tool_calls' }],
          });
        return new Response(
          `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, ...tool }] } }] })}\n\ndata: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'tool_calls' }] })}\n\ndata: [DONE]\n\n`,
        );
      }) as any,
    });
    expect(result.streaming_tools).toBe('pass');
    expect(seen.length).toBe(4);
    expect(seen.at(-1).stream).toBe(true);
  });
  test('fails clearly when the server answers chat but cannot return tool calls', async () => {
    await expect(
      checkServer(buildConfig(), {
        env: {},
        fetchImpl: (async (_url: string, opts: any) =>
          opts.body
            ? Response.json({
                choices: [{ message: { content: 'OK' }, finish_reason: 'stop' }],
              })
            : Response.json({ status: 'ok' })) as any,
      }),
    ).rejects.toThrow('invalid tool completion');
  });
  test('reports HTTP status without reflecting an endpoint or response secret', async () => {
    await expect(
      checkServer(buildConfig(), {
        env: {},
        fetchImpl: (async () => new Response('private response', { status: 503 })) as any,
      }),
    ).rejects.toThrow('ZINC health check returned HTTP 503');
  });
});

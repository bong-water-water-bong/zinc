// Frozen from tools/opencode_eval.mjs at Zinc commit 70c840bd.
// Relative workspace and public defaults make this snapshot portable.
const DEFAULT_MODEL = 'zinc/qwen36-35b-a3b-q4k-xl';
const DEFAULT_PROXY_MODEL = 'qwen36-35b-a3b-q4k-xl';
const DEFAULT_PROXY_PORT = 9091;
const DEFAULT_ZINC_UPSTREAM = 'http://127.0.0.1:9090/v1';
const DEFAULT_LLAMA_UPSTREAM = 'http://127.0.0.1:9088/v1';
const DEFAULT_TIMEOUT_MS = 15 * 60 * 1000;
const REPO_ROOT = '.';

export function parseArgs(argv = process.argv.slice(2)) {
  const opts = {
    provider: 'zinc',
    fixtureIds: ['all'],
    outputDir: null,
    opencodeBin: 'opencode',
    model: DEFAULT_MODEL,
    proxyModel: DEFAULT_PROXY_MODEL,
    proxyPort: DEFAULT_PROXY_PORT,
    zincUpstream: DEFAULT_ZINC_UPSTREAM,
    llamaUpstream: DEFAULT_LLAMA_UPSTREAM,
    traceDir: null,
    manageProxy: false,
    replaceProxy: false,
    keepProxy: false,
    dryRun: false,
    runBaseline: true,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    maxTokensCap: 512,
    forceEnableThinking: false,
    temperature: 0,
    topP: 1,
    repoRoot: REPO_ROOT,
  };

  const args = [...argv];
  const readValue = (i) => {
    if (i + 1 >= args.length || args[i + 1].startsWith('--')) return [true, i];
    return [args[i + 1], i + 1];
  };

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === '--provider') {
      const [v, ni] = readValue(i);
      opts.provider = String(v);
      i = ni;
    } else if (arg === '--fixtures' || arg === '--fixture') {
      const [v, ni] = readValue(i);
      opts.fixtureIds = String(v)
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);
      i = ni;
    } else if (arg === '--output' || arg === '--output-dir') {
      const [v, ni] = readValue(i);
      opts.outputDir = String(v);
      i = ni;
    } else if (arg === '--opencode-bin') {
      const [v, ni] = readValue(i);
      opts.opencodeBin = String(v);
      i = ni;
    } else if (arg === '--model') {
      const [v, ni] = readValue(i);
      opts.model = String(v);
      i = ni;
    } else if (arg === '--proxy-model') {
      const [v, ni] = readValue(i);
      opts.proxyModel = String(v);
      i = ni;
    } else if (arg === '--proxy-port') {
      const [v, ni] = readValue(i);
      opts.proxyPort = Number(v);
      i = ni;
    } else if (arg === '--zinc-upstream') {
      const [v, ni] = readValue(i);
      opts.zincUpstream = String(v);
      i = ni;
    } else if (arg === '--llama-upstream') {
      const [v, ni] = readValue(i);
      opts.llamaUpstream = String(v);
      i = ni;
    } else if (arg === '--trace-dir') {
      const [v, ni] = readValue(i);
      opts.traceDir = String(v);
      i = ni;
    } else if (arg === '--manage-proxy') {
      opts.manageProxy = true;
    } else if (arg === '--replace-proxy') {
      opts.replaceProxy = true;
    } else if (arg === '--keep-proxy') {
      opts.keepProxy = true;
    } else if (arg === '--dry-run') {
      opts.dryRun = true;
    } else if (arg === '--no-baseline') {
      opts.runBaseline = false;
    } else if (arg === '--timeout-ms') {
      const [v, ni] = readValue(i);
      opts.timeoutMs = Number(v);
      i = ni;
    } else if (arg === '--max-tokens-cap') {
      const [v, ni] = readValue(i);
      opts.maxTokensCap = Number(v);
      i = ni;
    } else if (arg === '--force-enable-thinking') {
      const [v, ni] = readValue(i);
      opts.forceEnableThinking = parseBool(v);
      i = ni;
    } else if (arg === '--temperature') {
      const [v, ni] = readValue(i);
      opts.temperature = Number(v);
      i = ni;
    } else if (arg === '--top-p') {
      const [v, ni] = readValue(i);
      opts.topP = Number(v);
      i = ni;
    } else if (arg === '--list') {
      opts.list = true;
    } else if (arg === '--help' || arg === '-h') {
      opts.help = true;
    }
  }

  return opts;
}

function parseBool(value) {
  if (typeof value === 'boolean') return value;
  return /^(1|true|yes|on)$/i.test(String(value));
}

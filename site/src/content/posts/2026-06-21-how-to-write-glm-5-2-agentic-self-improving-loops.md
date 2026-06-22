---
title: "How to Write GLM 5.2 Loops: Agentic, Self-Improving Coding Loops"
seoTitle: "GLM 5.2 Loops: Build Self-Improving Agentic Loops"
date: "2026-06-21"
tags:
  - glm-5.2
  - glm
  - agentic-loops
  - self-improving-loop
  - ai-loop
  - agentic-coding
  - autonomous-coding-loop
  - loops
  - llm-inference
keywords:
  - GLM 5.2
  - GLM 5.2 loops
  - agentic loops
  - self-improving loops
  - self-improving loop
  - agentic coding loop
  - autonomous coding loop
  - GLM 5.2 agentic coding
  - how to write GLM loops
  - self-improving AI loop
  - GLM 5.2 vs Claude
  - open-weight coding model
  - overnight optimization loop
  - GPU kernel optimization loop
excerpt: "A loop that runs 40 cycles overnight is only as good as the model you can afford to run 40 times. Here is how to write agentic, self-improving GLM 5.2 loops — the controller pattern, the prompts, the drop-in, and why a cheap, capable, open-weight agent changes the economics of autonomous coding."
seoDescription: "How to write GLM 5.2 loops: a practical guide to building agentic, self-improving coding loops that edit, verify on real hardware, and keep only the wins."
faqs:
  - question: "What is a GLM 5.2 loop?"
    answer: "A GLM 5.2 loop is an agentic coding loop where GLM 5.2 proposes one focused code change, a controller builds and runs the result on a fixed benchmark, and the change is kept only if a trusted evaluator says it improved. Memory of past cycles feeds the next prompt, which makes the loop self-improving over a run."
  - question: "How do you swap GLM 5.2 into an existing agentic coding loop?"
    answer: "The loop controller shells out to a coding-agent CLI and the model is just a string. Point the CLI at GLM 5.2 through its Anthropic-compatible or OpenAI-compatible endpoint with a base URL and API key, set the model id, and the same controller now drives GLM 5.2 with no rewrite."
  - question: "Why is GLM 5.2 a good model for self-improving loops?"
    answer: "Loops are token-hungry and run unattended, so the model that drives them has to be cheap per cycle, reliable at tool use over long horizons, and able to carry large memory context. GLM 5.2's coding-agent focus, long context, aggressive pricing, and open-weight lineage make running many cycles continuously affordable instead of a luxury."
  - question: "GLM 5.2 vs Claude for agentic coding loops?"
    answer: "Both work as loop drivers because both expose an agent CLI the controller can call. The practical difference is economics and control: GLM 5.2's lower cost per cycle lets you run loops continuously and fan out parallel loops, and its open-weight lineage lets you self-host the loop's brain, which is awkward with a closed frontier API."
  - question: "Can you run GLM 5.2 loops locally?"
    answer: "Yes. Because the GLM line ships open weights, you can serve GLM 5.2 through a local runtime such as vLLM, llama.cpp, or Ollama and point the loop's agent CLI at that local endpoint, so the model driving the loop runs on the same hardware the loop is optimizing."
---

A GLM 5.2 loop is an agentic coding loop where the model edits real code, a controller builds and runs the result, and only changes that measurably improve survive. The interesting part is not that GLM 5.2 can write code. Every capable model can write code. The interesting part is that you can now afford to let it write code forty times in a row, every night, across every subsystem, without flinching at the bill. That single economic fact is why GLM 5.2 is a different input to **self-improving loops** than the frontier APIs most of us started on.

This post is about how to write those loops. It is deliberately a little engine-agnostic: the pattern works whether you are optimizing a database, a compiler, a web app, or — as in our case — GPU inference kernels. We will use our own loops as the worked example, because we have run them for thousands of cycles and learned where they break, but the shape transfers directly.

The thesis is simple. The loop is the product. The model is the engine. And for the first time, the engine is cheap and capable enough that the loop can actually run at the scale where it pays off.

## What a GLM 5.2 loop actually is

There are two phrases that get used interchangeably and should not be. An **agentic loop** is a loop where a model takes actions — reads files, edits code, runs commands — instead of just emitting text. A **self-improving loop** is an agentic loop with memory and a keep-or-revert gate, so the search gets better at searching over time. A GLM 5.2 loop, done right, is both.

We wrote a whole post on the underlying idea — [the Karpathy loop, autoresearch, and the self-improving AI loop](/blog/2026-03-28-karpathy-loop-autoresearch-and-the-self-improving-ai-loop-behind-zinc) — and the durable conclusion there holds no matter which model you plug in. A self-improving loop is not mystical recursion. It is a controlled search process with four hard properties:

1. **A bounded mutation surface.** The agent changes one thing per cycle, not the whole repo.
2. **An evaluator you trust.** Something objective decides better from worse — a benchmark number, a test suite, a coherence check.
3. **Memory across attempts.** Each cycle carries forward what already worked and what already failed.
4. **A cheap way to revert.** Bad ideas are thrown away with one reset, not a manual untangle.

When those four properties are present, the development cycle gets dramatically shorter without pretending the underlying engineering got easy. The model is the thing inside the loop that proposes mutations. Everything else is the controller, and the controller is where the quality lives.

## The anatomy of an agentic loop

Strip away the domain and every good **agentic coding loop** runs the same cycle:

```text
ship → build → run → measure → propose ONE change → verify → keep or revert → remember
```

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-06-21-glm-loop-anatomy.svg" alt="A horizontal pipeline of seven loop stages — ship, build, run, measure, propose one change with GLM 5.2, verify, and keep or revert — flowing into a controller and memory panel that feeds the next cycle. A footnote notes the brain is swappable with one environment variable." loading="lazy" />
  <figcaption>Every stage except <em>propose</em> is controller plumbing. The model is a subprocess the loop calls — which is exactly why swapping in GLM 5.2 is a config change, not a rewrite.</figcaption>
</figure>

In our case the controller is a TypeScript program of a few hundred lines. It does not contain any model intelligence. It is plumbing: it syncs code to a machine, builds it, runs a fixed benchmark, parses the result into a struct, hands that struct to the agent as a prompt, captures the agent's edit, re-runs everything, and then makes a cold-blooded decision about whether reality actually improved.

Here is the important structural insight: the model is not the loop. The model is a subprocess the loop calls. Our controller shells out to a coding-agent CLI exactly the way you would type it by hand:

```ts
// The model is just a string the controller passes to an agent CLI.
const CLAUDE_MODEL = process.env.ZINC_CLAUDE_MODEL ?? "claude-opus-4-8[1m]";

function buildAgentArgs(prompt: string): string[] {
  return [
    "-p", prompt,
    "--output-format", "stream-json",
    "--verbose",
    "--permission-mode", "bypassPermissions",
    "--model", CLAUDE_MODEL,
  ];
}
```

That `--model` being an environment variable is the whole trick. The controller does not care who is on the other end of the CLI. It cares about the contract: take a prompt, edit the working tree, print a structured summary. Any model that can honor that contract is a candidate to drive the loop. Which is exactly why dropping in GLM 5.2 is a configuration change, not a rewrite.

## How to drop GLM 5.2 into the loop

The reason GLM 5.2 is so easy to adopt for loops is that the GLM line ships an **Anthropic-compatible** endpoint specifically so it works as a drop-in behind the same agent CLIs people already script against. (There is an OpenAI-compatible path too, if your loop drives a Codex-style CLI instead.) So "writing a GLM 5.2 loop" usually means pointing your existing loop at a GLM endpoint:

```bash
# Drive the SAME loop controller with GLM 5.2 instead of a frontier API.
export ANTHROPIC_BASE_URL="https://api.z.ai/api/anthropic"   # GLM's Anthropic-compatible gateway
export ANTHROPIC_AUTH_TOKEN="$GLM_API_KEY"
export ZINC_CLAUDE_MODEL="glm-5.2"                            # the only line that changed

bun loops/optimize_gpu.ts --agent claude --cycles 40
```

Nothing in the controller changes. The `claude` CLI is now talking to GLM 5.2, the loop runs its forty cycles, and the keep-or-revert gate judges GLM 5.2's edits exactly the way it judged the previous model's. If you want to run GLM 5.2 *locally* — which, as we will get to, is the most interesting option for our domain — you serve the open weights through vLLM, llama.cpp, or Ollama and point `ANTHROPIC_BASE_URL` at `http://localhost:port` instead. The loop neither knows nor cares.

This is the part people underestimate: the hard work of building a loop is one-time and model-agnostic. Once the controller exists, swapping the brain is a single env var. So the right mental model is not "I am writing a GLM 5.2 loop." It is "I am writing a loop, and GLM 5.2 happens to be the most cost-effective brain to run in it."

## Writing prompts a loop model will actually follow

A loop runs with no human in the chair, which changes how you prompt. In a chat you can correct a model that drifts. In a loop, drift is silently committed or silently reverted, and you find out tomorrow. So the prompt has to enforce a contract the controller can parse, and the model has to follow that contract on cycle 1 and on cycle 400.

Three things make GLM 5.2 prompts loop-safe:

**One focused mutation.** The prompt asks for a single change, not a refactor. This keeps diffs reviewable and keeps the evaluator's signal clean — if the change made things worse, you know exactly what to blame.

**A machine-readable summary.** We require the model to end its turn with tagged blocks the controller greps out:

```text
@@@DESCRIPTION: one-line summary of the change
@@@SELF_ANALYSIS: what I believe this did and why
@@@NEXT_IDEAS: ranked hypotheses for the next cycle
```

`@@@SELF_ANALYSIS` and `@@@NEXT_IDEAS` are not decoration. They are how the loop becomes self-improving: the controller feeds last cycle's analysis and ideas into next cycle's prompt, so the search compounds instead of restarting from amnesia every time.

**Carried memory.** Each prompt includes a compact history — the last N cycles, whether each was kept or reverted, the output snippet, and the previous self-analysis. This is the single biggest lever on loop quality. In our own runs, the same agent went from keeping **0 of 43** cycles with a thin prompt to keeping **40 of 44** with full architecture context and carried memory. The model did not get smarter between those runs. The prompt stopped making it rediscover the codebase every cycle.

GLM 5.2 matters here for a concrete reason: a loop model has to be a reliable *instruction follower and tool caller* over long horizons, because nobody is watching. A model that nails the structured contract 95% of the time still corrupts one cycle in twenty, and a corrupted cycle either wastes a build or sneaks a fake win past the gate. The GLM coding lineage is tuned for exactly this agentic, tool-driven, format-stable behavior, which is what makes it usable unattended rather than just impressive in a demo.

## Why GLM 5.2 is a game changer for loops

Plenty of models can edit code. The reason GLM 5.2 changes what loops are *for* comes down to four properties that matter far more in a loop than in a chat.

**Cost per cycle is the whole ballgame.** Loops are absurdly token-hungry. A single cycle reads the repo, the carried history, architecture notes, and large source files, then emits a diff and an analysis. Multiply by 40 cycles a night, times every subsystem, times the number of parallel loops you want to run, and the per-token price stops being a footnote and becomes the gating constraint. When the brain is cheap enough, loops graduate from a treat you run occasionally to infrastructure you leave on. You stop rationing cycles. You fan out ten loops instead of one. The aggressive pricing of GLM's coding plans is not a discount — at loop scale it is a capability unlock.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-06-21-glm-loop-economics.svg" alt="Loop cost equals tokens per cycle times cycles times price per token, with price per token highlighted as the lever. The left panel shows a single cycle is token-heavy and mostly fixed — repo context, carried memory, model reasoning, diff and analysis. The right panel shows a rising staircase of regimes unlocked as price per token drops: run occasionally, leave it running every night, then fan out parallel loops." loading="lazy" />
  <figcaption>Tokens-per-cycle and cycle count are fixed by the work. The model price is the term you choose — and dropping it is what moves a loop from "run occasionally" to "fan out across every subsystem."</figcaption>
</figure>

**Agentic reliability over long horizons.** A loop is the worst possible environment for a model that is brilliant once and flaky on the tenth turn. GLM 5.2 continues a line built for agentic coding — tool use, multi-step edits, and stable output formatting — which is precisely the behavior a keep-or-revert gate depends on.

**Long context for memory.** The self-improving property lives in the carried history. A model that can hold the architecture, the failed approaches, and the last fifteen cycles in context without summarizing them away searches better than one that has to forget. Long context is not a vanity metric in a loop; it is the memory substrate.

**Open weights, and what they unlock.** The GLM family has been an open-weight family, and that is the property that most changes the calculus for builders. An open-weight agentic model means the loop's brain is not a remote dependency you rent. You can self-host it, pin a version so your loop is reproducible, run it air-gapped on private code, and avoid the situation where a silent model update changes your loop's behavior overnight. Compared with a closed frontier API — Claude or Codex driving the same loop — the controller is identical; the difference is that with GLM 5.2 you can own the engine, not just call it. The nearest open-weight peers, like DeepSeek's coding models, share this advantage, but GLM 5.2's drop-in Anthropic-compatible surface makes it the least-friction swap into an existing Claude-Code-style loop.

Put those together and the headline is not "GLM 5.2 is smart." It is "GLM 5.2 is cheap, reliable, long-memoried, and ownable enough to run inside a loop continuously" — which is the exact profile a self-improving loop has always wanted and rarely been able to afford.

## Why this is the perfect fit for our domain

Our loops optimize a local inference engine: they edit Zig dispatch code and GLSL or CUDA kernels, sync to a remote GPU box, build on the target, run real inference against a fixed prompt, and check two things at once — did the output stay coherent, and did tokens-per-second go up. The evaluator is attached to hardware, so a "win" is not a model's opinion; it is a number the GPU produced. (We wrote about why that strictness matters in [ZINC_RT and the honest 35 tok/s baseline](/blog/2026-06-14-zinc-rt-runtime-local-inference-needs).)

That is a brutal environment for a loop, which is what makes a cheap, reliable agent so valuable in it. The wins our loops surface are not typos. They are the bugs and optimizations that live at the boundary between tensor layout, shader assumptions, dispatch code, and model architecture — subgroup reduction losses, wrong quant sub-block pairing, fused-kernel launch reductions that clear a hardware boost-clock floor. A human gets bored after the fifth failed attempt at a numerical bug. A loop does not get bored. It keeps doing the disciplined thing long after a person would start improvising — but only if you can afford to keep it running, which loops back to cost per cycle.

And here is the part that makes GLM 5.2 specifically poetic for anyone working on local inference. The GLM line is open-weight. We build an engine that runs open-weight models fast on consumer and workstation GPUs. So the endgame is a closed circle: serve GLM 5.2 *on the very engine the loop is optimizing*, on the very hardware the loop is tuning, and let it improve the runtime it runs on. A self-improving inference engine whose optimizer is a model hosted by that engine is not a thought experiment once the brain is open-weight and the runtime is yours. That is the kind of recursion the [whole project was built toward](/zinc).

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-06-21-glm-self-hosted-circle.svg" alt="A four-stage clockwise cycle: GLM 5.2 open weights are served by the local inference engine, the loop tunes the engine's Zig, GLSL and CUDA kernels, which yields a faster engine, which in turn serves the model that drives the loop. The center reads self-hosted self-improvement on your hardware." loading="lazy" />
  <figcaption>The endgame of an open-weight loop brain: the model that optimizes the engine is served <em>by</em> that engine, on the hardware the loop is tuning. The faster it makes the runtime, the faster its own next cycle runs.</figcaption>
</figure>

## The parts that quietly break loops

Swapping in a great model does not save a sloppy controller. The failures below are model-agnostic, and they are where most home-grown loops die. Get these right and GLM 5.2 will run honestly for a thousand cycles.

**Serialize the hardware.** If two cycles touch the same GPU at once, you are not measuring anything. We wrap the run in a `flock` so only one experiment owns the device at a time. A loop without a hardware lock is "a race condition with branding."

**Checkpoint before every edit.** The controller makes a throwaway commit before the agent touches the tree, so a bad cycle is one `git reset` away instead of a manual untangle. Cheap revert is property four; this is how you implement it.

**Reject fake wins.** This is the hardest and most important code in the whole loop. The gate must refuse a throughput gain that broke output coherence, refuse "different output" masquerading as "better output," and refuse unchanged results dressed up as progress. Without this layer, a self-improving loop degenerates into a commit generator — and a cheap model that runs more cycles will generate fake wins *faster*, so the gate matters more as the model gets cheaper, not less.

**Detect stalls and change behavior.** When the same output repeats for several cycles, the loop should not "try harder." It should switch modes — narrow the mutation surface, add a reference comparison, run a microbenchmark instead of full inference. A stall is a signal to change the kind of experiment, not the intensity.

**Log everything as if tomorrow is a post-mortem.** Every cycle should write its build log, run log, the exact prompt that produced the patch, the agent's raw output, and a structured result file, plus a resumable run state. This paper trail is what lets you improve the *controller* — which, over weeks, is where most of the gains actually come from.

## The durable principles

Models will keep changing. Today GLM 5.2 is the most cost-effective capable brain to run in a loop; in six months it might be something else, and because the model is one env var, you will swap it in an afternoon. What survives every model generation is the loop design:

1. Keep the mutation surface small enough to search.
2. Use an evaluator you actually trust — ideally one attached to reality, like real hardware.
3. Make bad changes cheap to discard.
4. Carry memory from one cycle into the next.

The viral framing of all this is "the agent improves your code while you sleep." That is catchy and it blurs the real boundary. What improves is not a mystical agent essence. What improves is a bounded search process over a measurable environment, run by a controller that gets stricter as it accumulates evidence.

GLM 5.2 does not change those principles. It changes the budget line under them — and at loop scale, the budget line is the difference between a clever demo you run once and an **autonomous coding loop** you leave running on every subsystem you own. That is the game-changer: not a smarter answer, but a loop you can finally afford to keep running.

If you want the rest of the story behind these loops, start with [why we are building this engine](/blog/2026-03-25-why-we-are-building-zinc), then the [self-improving loop deep dive](/blog/2026-03-28-karpathy-loop-autoresearch-and-the-self-improving-ai-loop-behind-zinc). The same point shows up from every angle: once the loop around the model becomes part of the product, the model you can afford to run inside it becomes a strategic choice — and GLM 5.2 is the one that finally makes the math work.

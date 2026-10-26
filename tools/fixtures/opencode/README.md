# OpenCode coding tasks

`zinc-cli-args.mjs` freezes the argument parser from Zinc commit
[`70c840bd`](https://github.com/zolotukhin/zinc/blob/70c840bd/tools/opencode_eval.mjs).
Its repository root is relative and its defaults use loopback URLs. Keeping a
snapshot lets the maintenance task remain reproducible after the real parser
is fixed. This is task input, not a replacement implementation or an answer.

The evaluation suite copies it into a disposable project. OpenCode must fix
validation in source; the harness checks the tests independently and verifies
that their contents did not change. The remaining tasks and all test inputs
are defined in `tools/opencode_eval_fixtures.mjs`.

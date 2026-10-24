#!/usr/bin/env node
// The executable GPT4O regex from llama.cpp/src/llama-vocab.cpp.
// GPT-OSS (gpt-4o) and Muse Glimmer (llama4) use this same splitter.
import { mkdir, writeFile } from 'node:fs/promises';

const pattern = String.raw`[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`;
const inputs = [
  '', 'camelCase XMLHttpRequest HTTPServer ABCdef ABCDef JSON_RESULT',
  "Can't I'M he'll You've she'd We're it'S O'Reilly",
  'token-type snake_case __word _of_turn -response',
  '0123456789 12 1234 １２３４ ١٢٣٤ Ⅻ²³⁴',
  '<p align="center">\n  <img width="360">',
  '</s> <>& <?> <()> <x> & <>literal.',
  'hello.\nworld /word///\r\nnext',
  'a  b\t\tc   ', '\t\tword\r\n \t\n  next',
  '\u00a0word\u2003\u2003next\u3000',
  'ǅABCĳdef ǅABC ABCǅDEFǅabc Ångström ÄÖÜäöü',
  'ΕλληνικάРусский中文 日本語한글',
  'e\u0301x A\u0301B \u0301word a\u200db',
  '🧠word 🧑‍💻code 👩🏽‍🚀\nnext',
  '._Word !?word  —word ©word',
  'a\r\n\t \u00a0\n \u2003b',
  'αABβCDγδZ zǅABCǅDEF x中文ABC中文def',
];

// Fixed-seed combinations exercise regex priority, UTF-8 boundaries and the
// overlap between the two Unicode letter classes independently of Zig code.
const alphabet = ['a', 'z', 'A', 'Z', 'é', 'Ω', '中', 'ǅ', 'ĳ', '\u0301', '1', '２', 'Ⅻ', '²', '_', '-', '/', '.', "'", '🧠', '\u200d', ' ', '\t', '\n', '\r', '\u00a0', '\u2003'];
let state = 0x4017;
const random = () => {
  state ^= state << 13;
  state ^= state >>> 17;
  state ^= state << 5;
  return state >>> 0;
};
for (let index = 0; index < 128; index += 1) {
  const length = 1 + random() % 40;
  inputs.push(Array.from({ length }, () => alphabet[random() % alphabet.length]).join(''));
}

const regex = new RegExp(pattern, 'uy');
const cases = inputs.map(input => {
  const chunks = [];
  let offset = 0;
  while (offset < input.length) {
    regex.lastIndex = offset;
    const match = regex.exec(input);
    if (!match || match[0].length === 0) throw new Error(`No progress at ${offset}: ${JSON.stringify(input)}`);
    chunks.push(match[0]);
    offset = regex.lastIndex;
  }
  return { input, chunks };
});
const output = new URL('../src/model/fixtures/gpt4o-chunks.json', import.meta.url);
await mkdir(new URL('.', output), { recursive: true });
await writeFile(output, JSON.stringify({ pattern, cases }, null, 2) + '\n');
const corpus = new URL('../docs/validation/zig-0.17-bpe/reference-corpus/gpt4o-unicode.txt', import.meta.url);
await mkdir(new URL('.', corpus), { recursive: true });
await writeFile(corpus, inputs.join('\n--\n') + '\n');
console.log(`Wrote ${cases.length} reference chunk cases.`);

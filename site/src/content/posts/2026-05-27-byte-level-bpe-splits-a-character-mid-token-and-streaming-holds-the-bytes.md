---
title: "Byte-level BPE splits a character mid-token, and streaming has to hold the bytes"
date: "2026-05-27"
tags:
  - zinc
  - tokenizer
  - byte-level-bpe
  - utf-8
  - streaming
  - detokenization
  - sse
  - qwen3
  - byte-fallback
  - llm-inference
keywords:
  - byte-level BPE UTF-8 token boundary
  - streaming detokenization replacement character
  - U+FFFD emoji streamed token by token
  - incremental detokenization local LLM
  - byte fallback token 0xNN SentencePiece
  - hold incomplete UTF-8 bytes streaming chunk
  - lastCompleteUtf8End codepoint boundary
  - Qwen3 emoji CJK broken streaming
  - TextDecoder replacement character SSE
  - local LLM server UTF-8 multi-byte token split
excerpt: "A local model can spell an emoji or a full Chinese sentence correctly and still have it land in the browser as a row of black diamonds. The cause is not the model and not the network. Byte-level BPE, the tokenizer scheme behind Qwen3, GPT-2, and most modern LLMs, emits tokens that are spans of raw bytes rather than whole characters, so a single four-byte character like the slightly smiling face 🙂 (F0 9F 99 82) is routinely split across two or more tokens. Ship each token's bytes to the client the instant it decodes and the receiver's UTF-8 decoder sees an unfinished sequence, which it turns into the U+FFFD replacement character. zinc fixes this the way every serious serving stack eventually does: it detokenizes against the codepoint boundary instead of the token boundary, holding the incomplete trailing bytes, up to the three continuation bytes a four-byte sequence can leave dangling, until the next token closes the character. The boundary, not the token, is the unit you are allowed to put on the wire."
---

A local model can generate a perfectly correct emoji and still show the user a black diamond with a question mark in it. Ask Qwen3 for a sentence of Japanese and watch it stream as a run of those diamonds, then snap into clean characters the moment generation stops. The model did nothing wrong. The bytes were all correct. The engine broke them on the way out by flushing them one token at a time.

This is the streaming detokenization problem, and it is one of those bugs that is invisible until the day it is not. ASCII never triggers it, so an English-only test suite stays green forever. Then someone pastes in a Chinese prompt, or the model reaches for an emoji, and half the reply is replacement characters. The fix is small and the mechanism is worth understanding, because it sits at the seam between two systems that disagree about what a unit of text is: the tokenizer thinks in tokens, and UTF-8 thinks in codepoints, and a streaming server has to translate between them token by token without ever shipping half a character.

We have spent a lot of this month on the parts of a local engine that move weights and keys around the [Radeon AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html). This is the opposite kind of post. The decode kernel can be flawless and the answer can still arrive corrupted, because the last hop, turning token IDs back into text on a live stream, has a correctness trap that has nothing to do with the GPU.

## Why a streaming engine hits this and a batch one does not

The trap is specific to streaming. If you generate the whole response, detokenize it once, and return it, none of this matters, because by the time you decode you are holding every byte the model produced. UTF-8 reassembles trivially when all the pieces are present.

A chat server does not have that luxury. It streams, usually as server-sent events, emitting text as each token is sampled so the user sees words appear in real time. zinc serves a single user decoding at roughly a hundred tokens per second, and the streaming loop in the server sends the newly generated text after every step. That is the right behavior for latency and the wrong moment to assume you have a whole character in hand. The token you just sampled might be the first half of one.

This is also why it took the field a while to standardize a fix. Greedy, decode-everything-then-return pipelines never saw it. The moment serving stacks went to token-by-token streaming, every one of them grew the same wart, which is the subject of an open [Hugging Face tokenizers issue on incremental detokenization](https://github.com/huggingface/tokenizers/issues/1666) where the vLLM and TGI maintainers compare notes on their respective workarounds.

## A token is a span of bytes, not a character

To see why a token can end mid-character, you have to look at what modern tokenizers actually store. Qwen3, GPT-2, Llama, and most current models use byte-level byte-pair encoding. The scheme was introduced with [GPT-2's byte-level BPE](https://raw.githubusercontent.com/openai/gpt-2/master/src/encoder.py), and its defining trick is that it operates on the 256 raw byte values, not on characters. Every piece of text is first encoded to UTF-8 bytes, and BPE merges are learned over those bytes. The vocabulary is built so that any string at all can be represented, because in the worst case it falls back to single bytes.

That worst case is common for anything outside the training-heavy scripts. A SentencePiece-style tokenizer with byte fallback represents an unknown byte as a literal token like `<0x9F>`, and llama.cpp tracks the same need under the heading of [supporting partial unicode codepoints in tokens](https://github.com/ggml-org/llama.cpp/issues/6462). The consequence is the same in both designs: the atomic unit the model emits is a byte or a run of bytes, and a multi-byte character is assembled from several of them.

The slightly smiling face 🙂 is a clean example. Its codepoint is U+1F642, which UTF-8 encodes as four bytes, F0 9F 99 82. A byte-level tokenizer that has not merged that exact emoji into a single vocab entry will emit it as up to four separate tokens, one per byte. Each token decodes to a single byte that is, on its own, not valid UTF-8. F0 is the start of a four-byte sequence with nothing after it. 9F, 99, and 82 are bare continuation bytes with no leader. Decode any one of them in isolation and a strict UTF-8 decoder has no choice but to emit U+FFFD, the replacement character, which is exactly the black diamond the user sees. The [WHATWG Encoding standard](https://encoding.spec.whatwg.org/) that browsers implement is explicit that a malformed sequence is replaced rather than passed through, so the receiver is doing the correct thing with the broken input the server handed it.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-27-utf8-token-boundary-stream-hold-buffer.svg" alt="A two-band schematic on a warm parchment background titled one emoji, four byte tokens, one streaming decision, noting that the face emoji U+1F642 is the four bytes F0 9F 99 82, often split across tokens. The top band, path one, flush each token's bytes the moment it decodes, shows four token chips reading 0xF0, 0x9F, 0x99, 0x82, an arrow labeled decode each token alone, and an output strip labeled browser TextDecoder byte by byte that displays four red U+FFFD replacement characters with the note four lone bytes, each invalid UTF-8, each replaced, plus a red cross and the label the emoji never renders. The bottom band, path two, hold the incomplete tail until the codepoint closes, notes that zinc walks back over up to three trailing continuation bytes and flushes only to the last complete boundary in lastCompleteUtf8End. It shows four steps in time order: after appending 0xF0 the hold buffer is one byte, incomplete, emit zero; after 0x9F it is two bytes, incomplete, emit zero; after 0x99 it is three bytes, incomplete, emit zero; after 0x82 it is a complete four-byte run shown in green, flush the emoji, emit four. A dashed time arrow runs left to right ending at one clean emoji on the wire. A legend marks held bytes, completed codepoints, and replacement characters, and credits RFC 3629 for the UTF-8 byte structure." loading="lazy" />
  <figcaption>Top: flushing each token's bytes as it lands sends four lone bytes to the client, and the receiver's UTF-8 decoder replaces every one with U+FFFD, so a valid emoji renders as garbage. Bottom: holding the incomplete trailing bytes until a complete codepoint forms, the strategy in zinc's lastCompleteUtf8End, emits nothing for the first three steps and one clean character on the fourth. The byte values are real; the layout is schematic.</figcaption>
</figure>

The top band is the bug and the bottom band is the fix. The only difference between them is when the server is willing to flush. Path one flushes on every token boundary and corrupts anything wider than one byte. Path two flushes only on a codepoint boundary, which sometimes means a token produces no output at all and the next one produces several characters at once.

## The magic number is three

The fix needs to know, given a buffer of bytes, where the last complete character ends. UTF-8 makes that decidable from the bytes alone, which is one of the format's best properties and the reason this is a small fix rather than a stateful nightmare. The [UTF-8 definition in RFC 3629](https://datatracker.ietf.org/doc/html/rfc3629) lays out the entire structure in one table.

| Code point range | Bytes | Lead byte | Continuation bytes | Example characters |
| --- | ---: | --- | ---: | --- |
| U+0000 – U+007F | 1 | `0xxxxxxx` | 0 | ASCII, `A` `9` `\n` |
| U+0080 – U+07FF | 2 | `110xxxxx` | 1 | `é`, Greek, Cyrillic, Arabic |
| U+0800 – U+FFFF | 3 | `1110xxxx` | 2 | most CJK, `語` `好` |
| U+10000 – U+10FFFF | 4 | `11110xxx` | 3 | emoji 🙂, supplementary planes |

Every continuation byte has the high bits `10`, and no lead byte does, so the two are never confused. A sequence is at most four bytes, which means the longest incomplete tail you can ever be holding is three bytes: the three continuation bytes of a four-byte character whose lead has not arrived in this chunk, or any shorter prefix of a multi-byte sequence. That is the whole reason the number three shows up in the code. To find the last safe place to cut, you walk backward over at most three trailing continuation bytes, look at the byte before them, and ask whether you are holding as many bytes as that lead byte promised.

Read the table next to the diagram and the risk profile by language falls out immediately. ASCII is one byte and can never split, which is why English never triggers the bug. Accented Latin, Greek, Cyrillic, Hebrew, and Arabic are two bytes. CJK is three. Emoji and the supplementary planes are four, and so are the most likely to be torn across tokens, because they are the most likely to fall back to per-byte tokens in the first place. The scripts most exposed to this bug are exactly the ones an English-first test never exercises.

## What zinc does at the boundary

zinc's streaming path detokenizes to the codepoint boundary before each flush. The function that finds it is small, and it is small precisely because RFC 3629 made the structure self-describing.

```zig
// Longest prefix of `bytes` that ends on a complete UTF-8 codepoint.
// Trailing bytes of an unfinished sequence are held for the next chunk.
fn lastCompleteUtf8End(bytes: []const u8) usize {
    var i = bytes.len;
    var continuations: usize = 0;
    while (i > 0 and continuations < 3) {        // at most 3 trailing 10xxxxxx
        if ((bytes[i - 1] & 0xC0) != 0x80) break;
        i -= 1;
        continuations += 1;
    }
    const lead = bytes[i - 1];
    const expected: usize = if (lead < 0x80) 1
        else if ((lead & 0xE0) == 0xC0) 2
        else if ((lead & 0xF0) == 0xE0) 3
        else if ((lead & 0xF8) == 0xF0) 4
        else 1;                                  // malformed lead, let it through
    const have = bytes.len - (i - 1);
    return if (have >= expected) bytes.len else i - 1;  // hold the unfinished tail
}
```

The streaming loop calls this on the bytes it has accumulated since the last send, flushes the prefix up to the returned offset, and leaves the rest in the buffer. If a token contributes the F0 of an emoji, the function reports that nothing past the previous boundary is complete, so the server sends zero new bytes and carries F0 forward. Three tokens later, when 82 arrives and the buffer holds F0 9F 99 82, the function reports the full length and all four bytes go out together as one clean 🙂. The user sees the character appear slightly later than a pure ASCII character would, by a few tokens, but never sees a replacement glyph.

Decoding a single token into its bytes is its own small chore, because byte-level BPE does not store raw bytes in the vocabulary. It stores them through GPT-2's reversible byte-to-unicode mapping, which lifts the 256 byte values into printable codepoints so the merge tables never contain control characters. zinc's `decodeToken` reverses that mapping to recover the original bytes, and passes non-ASCII codepoints through as raw UTF-8 rather than substituting a question mark, so a token that already carries a whole multi-byte character survives intact. The boundary check then runs over the reconstructed byte stream, not over the tokens, which is the entire point.

## The costs of holding a byte

This is not free, and the honest version of the story names the costs. The first is latency, though it is tiny. Holding the tail of an unfinished character delays it by however many tokens complete it, at most three decode steps, which at a hundred tokens per second is on the order of tens of milliseconds and invisible next to the wait for the next word.

The second cost is that the held bytes interact with everything else the streaming layer does. Stop-sequence matching, tool-call tag detection, and the trimming of chat-template artifacts all run on the same text buffer, and they have to agree on the same boundary rule or they reintroduce the bug from a different direction. zinc treats a run of pure replacement characters as an artifact to suppress rather than stream, and it refuses to flush a dangling partial chunk while it is still deciding whether a tool-call tag is forming. This is the same family of seam-level correctness work we hit with [JSON-constrained decoding](/blog/2026-05-03-why-json-constrained-decode-no-longer-scans-a-151k-token-bitmask/), where the text leaving the model has structure the server must respect mid-stream.

The third cost is conceptual, and it is the one worth keeping. A whole class of local-engine bugs lives in this gap between what the model produced and what the user receives, and they are nasty because they are not reproducible from the logits. We made the same point about [why temperature zero is not deterministic on RDNA4 yet](/blog/2026-05-14-temperature-zero-is-not-deterministic-for-local-qwen3-on-rdna4-yet/): the model can be exactly right and the surrounding machinery still hands the user something wrong. Owning that machinery, which is part of [why we wrote our own runtime](/blog/2026-05-18-inside-the-decision-to-write-our-own-gpu-runtime-for-local-llm-inference/) and saw [plenty break early](/blog/2026-03-27-what-broke-first-when-we-built-zinc-on-amd-rdna4/), means these are our bugs to fix rather than a black box's to hide.

## It is a serving problem too, not just a local one

The reason this is worth a post rather than a footnote is that it does not go away at scale. The large serving stacks fight the same battle, and they fight a harder version of it. Their incremental detokenization has to handle not only the byte-boundary problem but also tokenizer cleanup rules, where the decoder adds or drops a space depending on the surrounding token IDs, which means the correct text for a token can depend on tokens that have not arrived yet. The vLLM contributors describe exactly this in the [tokenizers incremental detokenization thread](https://github.com/huggingface/tokenizers/issues/1666), calling their own code slow and nasty and asking whether the upstream library could own the logic, and pointing at IBM's text-generation-inference decoder as a reference implementation. At high batch sizes this runs per request, per token, and the cost is real enough to be worth optimizing.

A single-user local engine gets off easier on cost and not at all on correctness. There is one stream, so the per-token work is negligible, which is one more way the batch-of-one regime simplifies a problem that is brutal under concurrency. But the user staring at one chat window is the least forgiving audience for a corrupted emoji, because there is no aggregate throughput number to hide behind. One visible black diamond in a reply reads as a broken product.

## What comes next

The rule that falls out of this is short. On a streaming engine, the token boundary is not a safe place to cut text, because byte-level BPE will hand you the front half of a character and call it a token. The codepoint boundary is the safe place, UTF-8 makes it cheap to find, and the longest you ever have to wait for one is three more bytes. zinc holds the unfinished tail and flushes on the boundary, and the cost is a few tens of milliseconds of delay on the rare token that splits a character.

The thing to carry forward is the framing. A local engine is judged on the text that reaches the user, not the token IDs that leave the sampler, and the translation between them is a correctness surface in its own right. We have spent the month making decode fast on RDNA4. This is a reminder that fast is necessary and not sufficient, and that the last few inches of the pipeline, the ones with no GPU in them, are where a perfectly good answer most quietly goes wrong.

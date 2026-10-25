import { expect, test } from 'bun:test';
import { parseCompletion } from './validate_opencode_api.mjs';

test('assembles split tool arguments with Unicode and terminal usage', () => {
  const frames = [
    {
      choices: [
        {
          delta: {
            tool_calls: [
              { index: 0, id: 'call_1', function: { name: 'probe', arguments: '{"message":' } },
            ],
          },
        },
      ],
    },
    { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '"東京🚀"}' } }] } }] },
    {
      choices: [{ delta: {}, finish_reason: 'tool_calls' }],
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    },
  ];
  const text =
    frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join('') + 'data: [DONE]\n\n';
  const parsed = parseCompletion(text, true);
  expect(parsed.finish).toBe('tool_calls');
  expect(parsed.usage.total_tokens).toBe(15);
  expect(parsed.message.tool_calls[0].id).toBe('call_1');
  expect(JSON.parse(parsed.message.tool_calls[0].function.arguments)).toEqual({
    message: '東京🚀',
  });
});

test('rejects truncated streams and streamed server errors', () => {
  expect(() =>
    parseCompletion('data: {"choices":[{"delta":{"content":"OK"}}]}\n\n', true),
  ).toThrow();
  expect(() =>
    parseCompletion('data: {"error":{"message":"failed"}}\n\ndata: [DONE]\n\n', true),
  ).toThrow();
});

test('keeps non-streaming tool calls and usage intact', () => {
  const json = {
    choices: [
      {
        message: { role: 'assistant', content: null, tool_calls: [{ id: 'call_1' }] },
        finish_reason: 'tool_calls',
      },
    ],
    usage: { total_tokens: 20 },
  };
  const parsed = parseCompletion(JSON.stringify(json), false);
  expect(parsed.message).toEqual(json.choices[0].message);
  expect(parsed.usage).toEqual(json.usage);
});

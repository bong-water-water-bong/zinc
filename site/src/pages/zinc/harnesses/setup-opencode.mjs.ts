import source from '../../../../../tools/setup_opencode.mjs?raw';

export const prerender = true;

export function GET() {
  return new Response(source, {
    headers: {
      'Content-Type': 'text/javascript; charset=utf-8',
      'Content-Disposition': 'attachment; filename="setup-opencode.mjs"',
    },
  });
}

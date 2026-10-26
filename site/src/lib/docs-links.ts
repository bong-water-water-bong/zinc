type DocLinkTarget = { slug: string; sourcePath: string };
type DocLinkOptions = { sourcePath: string; docs: readonly DocLinkTarget[] };

export function createDocLinkResolver({ sourcePath, docs }: DocLinkOptions) {
  const routes = new Map(docs.map((doc) => [doc.sourcePath, `/zinc/docs/${doc.slug}/`]));
  return (href: string): string => {
    if (!href || /^(?:[a-z][a-z\d+.-]*:|\/|#)/i.test(href)) return href;
    try {
      const target = new URL(href, `https://repo.invalid/${sourcePath}`);
      if (!/\.md$/i.test(target.pathname)) return href;
      const path = decodeURIComponent(target.pathname.slice(1));
      const route = routes.get(path);
      return route
        ? `${route}${target.search}${target.hash}`
        : `https://github.com/zolotukhin/zinc/blob/main${target.pathname}${target.search}${target.hash}`;
    } catch {
      return href;
    }
  };
}

export function rehypeDocLinks(options: DocLinkOptions) {
  const resolve = createDocLinkResolver(options);
  return (tree: any) => {
    const walk = (node: any) => {
      if (!node || typeof node !== 'object') return;
      if (
        node.type === 'element' &&
        node.tagName === 'a' &&
        typeof node.properties?.href === 'string'
      ) {
        node.properties.href = resolve(node.properties.href);
      }
      if (Array.isArray(node.children)) node.children.forEach(walk);
    };
    walk(tree);
  };
}

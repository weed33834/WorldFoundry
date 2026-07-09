function normalizeBasePath(value: string | undefined) {
  if (!value || value === '/') return '';
  const withLeadingSlash = value.startsWith('/') ? value : `/${value}`;
  return withLeadingSlash.endsWith('/') ? withLeadingSlash.slice(0, -1) : withLeadingSlash;
}

export const basePath = normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH);
const demoAssetBaseUrl = process.env.NEXT_PUBLIC_DEMO_ASSET_BASE_URL?.replace(/\/+$/g, '');

export function withBasePath(path: string | undefined) {
  if (!path || !basePath) return path;
  if (/^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(path)) return path;

  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  if (normalizedPath === basePath || normalizedPath.startsWith(`${basePath}/`)) {
    return normalizedPath;
  }

  return `${basePath}${normalizedPath}`;
}

export function withMediaPath(path: string | undefined) {
  if (!path) return path;
  if (/^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(path)) return path;

  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  if (demoAssetBaseUrl && normalizedPath.startsWith('/demos/')) {
    return `${demoAssetBaseUrl}${normalizedPath}`;
  }

  return withBasePath(normalizedPath);
}

export function stripBasePath(pathname: string) {
  if (!basePath) return pathname;
  if (pathname === basePath) return '/';
  if (pathname.startsWith(`${basePath}/`)) return pathname.slice(basePath.length);
  return pathname;
}

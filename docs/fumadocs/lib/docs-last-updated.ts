import fs from 'node:fs';
import path from 'node:path';

import type { Locale } from '@/lib/i18n';

function resolveContentPath(pagePath: string) {
  const baseDir = path.join(process.cwd(), 'content/docs');
  const candidates = [
    path.join(baseDir, pagePath),
    path.join(baseDir, `${pagePath}.mdx`),
    path.join(baseDir, `${pagePath}.md`),
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return null;
}

export function getDocsLastUpdated(pagePath: string, locale: Locale) {
  const localizedSuffix = locale === 'en' ? '' : `.${locale}`;
  const localizedCandidates = [
    pagePath.replace(/\.mdx$/, `${localizedSuffix}.mdx`),
    pagePath.replace(/\.md$/, `${localizedSuffix}.md`),
    `${pagePath}${localizedSuffix}`,
  ];

  const filePath =
    localizedCandidates.map(resolveContentPath).find(Boolean) ?? resolveContentPath(pagePath);

  if (!filePath) return null;

  try {
    const { mtimeMs } = fs.statSync(filePath);
    const date = new Date(mtimeMs);

    return {
      date,
      iso: date.toISOString(),
      formatted: new Intl.DateTimeFormat(locale === 'zh' ? 'zh-CN' : 'en-US', {
        dateStyle: 'medium',
      }).format(date),
    };
  } catch {
    return null;
  }
}

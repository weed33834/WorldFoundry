import { getBenchmarkCatalogEntry } from '@/lib/benchmark-catalog';
import { getNavPageLabel } from '@/lib/docs-navigation';
import type { Locale } from '@/lib/i18n';
import { source } from '@/lib/source';
import type { DocsBreadcrumbItem } from '@/components/docs-breadcrumb';

export function getDocsBreadcrumbs(
  slugs: readonly string[],
  locale: Locale,
  title: string,
): DocsBreadcrumbItem[] {
  if (slugs.length === 0) return [];

  const items: DocsBreadcrumbItem[] = [];
  const segments: string[] = [];

  for (const segment of slugs) {
    segments.push(segment);
    const page = source.getPage(segments, locale);
    if (!page) continue;

    const isLast = segments.length === slugs.length;
    let label = getNavPageLabel(page.slugs, locale, page.data.title);

    if (isLast && slugs[0] === 'evaluation' && slugs[1] === 'benchmark-hub' && slugs.length === 3) {
      label = getBenchmarkCatalogEntry(slugs[2])?.name ?? title;
    }

    items.push({
      href: page.url,
      label,
    });
  }

  return items;
}

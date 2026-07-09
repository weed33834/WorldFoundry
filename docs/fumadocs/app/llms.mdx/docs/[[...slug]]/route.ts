import { getLLMText, getPageMarkdownUrl, source } from '@/lib/source';
import { defaultLocale, isLocale } from '@/lib/i18n';
import { notFound } from 'next/navigation';

export const revalidate = false;

export async function GET(_req: Request, { params }: RouteContext<'/llms.mdx/docs/[[...slug]]'>) {
  const { slug } = await params;
  // remove the appended "content.md"
  const slugs = slug?.slice(0, -1) ?? [];
  const maybeLocale = slugs[0];
  const locale = isLocale(maybeLocale) ? maybeLocale : defaultLocale;
  const pageSlugs = isLocale(maybeLocale) ? slugs.slice(1) : slugs;
  const page = source.getPage(pageSlugs, locale);
  if (!page) notFound();

  return new Response(await getLLMText(page), {
    headers: {
      'Content-Type': 'text/markdown',
    },
  });
}

export function generateStaticParams() {
  return source.getPages().map((page) => ({
    slug: getPageMarkdownUrl(page).segments,
  }));
}

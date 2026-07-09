import { DocsPage } from '@/components/docs-page';
import { generateDocsMetadata, generateLocalizedDocsStaticParams } from '@/lib/docs-page-server';
import { defaultLocale } from '@/lib/i18n';
import type { Metadata } from 'next';

type Params = {
  lang: string;
  slug?: string[];
};

export default async function Page({ params }: { params: Promise<Params> }) {
  const { lang, slug } = await params;

  return <DocsPage slug={slug} locale={lang} />;
}

export function generateStaticParams() {
  return generateLocalizedDocsStaticParams().filter((item) => item.lang !== defaultLocale);
}

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const { lang, slug } = await params;

  return generateDocsMetadata(slug, lang);
}

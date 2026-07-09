import { getPageImage, source } from '@/lib/source';
import { defaultLocale, isLocale } from '@/lib/i18n';
import { notFound } from 'next/navigation';
import { ImageResponse } from 'next/og';
import { appName } from '@/lib/shared';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

export const revalidate = false;

const fontDir = join(process.cwd(), 'node_modules/@fontsource/noto-sans-sc/files');
let ogFontsPromise:
  | Promise<
      {
        data: ArrayBuffer;
        name: string;
        style: 'normal';
        weight: 400 | 700;
      }[]
    >
  | undefined;

function toArrayBuffer(buffer: Buffer): ArrayBuffer {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) as ArrayBuffer;
}

function getOgFonts() {
  ogFontsPromise ??= Promise.all([
    readFile(join(fontDir, 'noto-sans-sc-chinese-simplified-400-normal.woff')),
    readFile(join(fontDir, 'noto-sans-sc-chinese-simplified-700-normal.woff')),
  ]).then(([regular, bold]) => [
    {
      data: toArrayBuffer(regular),
      name: 'Noto Sans SC',
      style: 'normal' as const,
      weight: 400 as const,
    },
    {
      data: toArrayBuffer(bold),
      name: 'Noto Sans SC',
      style: 'normal' as const,
      weight: 700 as const,
    },
  ]);

  return ogFontsPromise;
}

export async function GET(_req: Request, { params }: RouteContext<'/og/docs/[...slug]'>) {
  const { slug } = await params;
  const slugs = slug.slice(0, -1);
  const maybeLocale = slugs[0];
  const locale = isLocale(maybeLocale) ? maybeLocale : defaultLocale;
  const pageSlugs = isLocale(maybeLocale) ? slugs.slice(1) : slugs;
  const page = source.getPage(pageSlugs, locale);
  if (!page) notFound();

  return new ImageResponse(
    (
      <div
        style={{
          background: '#f7f5ef',
          color: '#111',
          display: 'flex',
          flexDirection: 'column',
          fontFamily: 'Noto Sans SC',
          height: '100%',
          padding: '64px',
          width: '100%',
        }}
      >
        <div style={{ display: 'flex', fontSize: 32, fontWeight: 700, justifyContent: 'space-between' }}>
          <span>{appName}</span>
          <span style={{ color: '#65645f', fontSize: 24 }}>docs</span>
        </div>
        <div
          style={{
            borderTop: '2px solid #111',
            display: 'flex',
            flexDirection: 'column',
            gap: 24,
            marginTop: 'auto',
            paddingTop: 40,
          }}
        >
          <div style={{ display: 'flex', fontSize: 76, fontWeight: 700, letterSpacing: 0, lineHeight: 1.04 }}>
            {page.data.title}
          </div>
          <div style={{ color: '#65645f', display: 'flex', fontSize: 28, lineHeight: 1.35, maxWidth: 920 }}>
            {page.data.description}
          </div>
        </div>
      </div>
    ),
    {
      fonts: await getOgFonts(),
      width: 1200,
      height: 630,
    },
  );
}

export function generateStaticParams() {
  return source.getPages().map((page) => ({
    slug: getPageImage(page).segments,
  }));
}

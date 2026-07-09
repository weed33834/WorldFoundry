'use client';

import { AnchorProvider, ScrollProvider, type TOCItemType } from 'fumadocs-core/toc';
import { useRef } from 'react';

import { DocsTocLinks } from '@/components/docs-toc-links';
import { useMediaQuery } from '@/lib/use-media-query';

type DocsTocRailProps = {
  title: string;
  items: TOCItemType[];
};

export function DocsTocRail({ title, items }: DocsTocRailProps) {
  const wide = useMediaQuery('(min-width: 1280px)');
  const linksRef = useRef<HTMLDivElement>(null);

  if (!wide || items.length === 0) return null;

  return (
    <AnchorProvider toc={items} single>
      <ScrollProvider containerRef={linksRef}>
        <aside className="pi-doc-right-rail" aria-label={title}>
          <nav className="pi-doc-toc">
            <span>{title}</span>
            <DocsTocLinks items={items} linksRef={linksRef} />
          </nav>
        </aside>
      </ScrollProvider>
    </AnchorProvider>
  );
}

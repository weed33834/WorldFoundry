'use client';

import { TOCItem, type TOCItemType } from 'fumadocs-core/toc';
import type { CSSProperties, RefObject } from 'react';

type DocsTocLinksProps = {
  items: TOCItemType[];
  linksRef: RefObject<HTMLDivElement | null>;
  className?: string;
};

export function DocsTocLinks({ items, linksRef, className = 'pi-doc-toc-links' }: DocsTocLinksProps) {
  return (
    <div className={className} ref={linksRef}>
      {items.map((item, index) => (
        <TOCItem
          href={item.url}
          key={`${item.url}-${index}`}
          style={
            {
              '--toc-indent': `${Math.max(0, item.depth - 2) * 12}px`,
            } as CSSProperties
          }
        >
          {item.title}
        </TOCItem>
      ))}
    </div>
  );
}

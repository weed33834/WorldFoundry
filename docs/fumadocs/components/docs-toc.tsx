'use client';

import { AnchorProvider, ScrollProvider, TOCItem, type TOCItemType } from 'fumadocs-core/toc';
import { useRef, type CSSProperties } from 'react';

type DocsTocProps = {
  title: string;
  items: TOCItemType[];
};

export function DocsToc({ title, items }: DocsTocProps) {
  const linksRef = useRef<HTMLDivElement>(null);

  return (
    <aside className="pi-doc-right-rail" aria-label={title}>
      <AnchorProvider toc={items} single>
        <ScrollProvider containerRef={linksRef}>
          <nav className="pi-doc-toc">
            <span>{title}</span>
            <div className="pi-doc-toc-links" ref={linksRef}>
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
          </nav>
        </ScrollProvider>
      </AnchorProvider>
    </aside>
  );
}

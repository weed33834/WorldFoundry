import Link from 'next/link';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import type { DocsPaginationLink } from '@/lib/docs-pagination';

type DocsPaginationProps = {
  prev?: DocsPaginationLink;
  next?: DocsPaginationLink;
  previousLabel: string;
  nextLabel: string;
};

export function DocsPagination({ prev, next, previousLabel, nextLabel }: DocsPaginationProps) {
  if (!prev && !next) return null;

  return (
    <nav className="pi-doc-pagination" aria-label="Document pagination">
      {prev ? (
        <Link href={prev.url} className="pi-doc-pagination-link pi-doc-pagination-link-prev" rel="prev">
          <ChevronLeft aria-hidden="true" />
          <span>
            <span className="pi-doc-pagination-kicker">{previousLabel}</span>
            <span className="pi-doc-pagination-title">{prev.label}</span>
          </span>
        </Link>
      ) : (
        <span className="pi-doc-pagination-spacer" aria-hidden="true" />
      )}
      {next ? (
        <Link href={next.url} className="pi-doc-pagination-link pi-doc-pagination-link-next" rel="next">
          <span>
            <span className="pi-doc-pagination-kicker">{nextLabel}</span>
            <span className="pi-doc-pagination-title">{next.label}</span>
          </span>
          <ChevronRight aria-hidden="true" />
        </Link>
      ) : (
        <span className="pi-doc-pagination-spacer" aria-hidden="true" />
      )}
    </nav>
  );
}

import Link from 'next/link';
import type { DocsRelatedLink } from '@/lib/docs-related-links';

type DocsRelatedLinksProps = {
  title: string;
  links: DocsRelatedLink[];
};

export function DocsRelatedLinks({ title, links }: DocsRelatedLinksProps) {
  if (links.length === 0) return null;

  return (
    <section className="pi-doc-related" aria-label={title}>
      <h2 className="pi-doc-related-title">{title}</h2>
      <div className="pi-doc-related-links">
        {links.map((link) => (
          <Link href={link.url} className="pi-doc-related-link" key={link.url}>
            {link.label}
          </Link>
        ))}
      </div>
    </section>
  );
}

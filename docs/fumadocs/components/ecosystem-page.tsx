import Link from 'next/link';

import { SiteNav, type SiteNavItemId } from '@/components/site-nav';
import { SiteSearchTrigger } from '@/components/site-search-trigger';
import { WorldFoundryWordmarkLink } from '@/components/worldfoundry-wordmark';
import { WORLDFOUNDRY_GITHUB_REPO } from '@/lib/site-links';

type EcosystemLink = {
  href: string;
  label: string;
  text: string;
};

type EcosystemSection = {
  title: string;
  links: EcosystemLink[];
};

type EcosystemPageProps = {
  active: SiteNavItemId;
  comingSoon?: string;
  description?: string;
  footerLabel: string;
  label: string;
  sections?: EcosystemSection[];
  title: string;
};

export function EcosystemPage({
  active,
  comingSoon,
  description,
  footerLabel,
  label,
  sections = [],
  title,
}: EcosystemPageProps) {
  return (
    <main className="pi-home-shell">
      <div className="mx-auto w-full max-w-7xl px-4 py-8 md:px-8 md:py-12">
        <header className="pi-header">
          <div className="flex flex-wrap items-center justify-between w-full">
            <WorldFoundryWordmarkLink variant="header" />
            <div className="pi-site-header-tools ml-auto">
              <SiteNav active={active} />
              <SiteSearchTrigger />
            </div>
          </div>
        </header>

        <section className="pi-ecosystem-hero">
          <p className="pi-label">{label}</p>
          <h1>{title}</h1>
          {description ? <p>{description}</p> : null}
        </section>

        {comingSoon ? (
          <section className="pi-open-section pi-coming-soon" aria-labelledby="coming-soon-title">
            <h2 id="coming-soon-title">Coming Soon</h2>
            <p>{comingSoon}</p>
          </section>
        ) : null}

        {sections.map((section) => (
          <section className="pi-open-section" key={section.title}>
            <h2>{section.title}</h2>
            <div className="pi-ecosystem-list">
              {section.links.map((item) => {
                const external = item.href.startsWith('http');
                return (
                  <a
                    className="pi-ecosystem-link"
                    href={item.href}
                    key={item.href}
                    {...(external ? { target: '_blank', rel: 'noreferrer' } : {})}
                  >
                    <span>{item.label}</span>
                    <p>{item.text}</p>
                  </a>
                );
              })}
            </div>
          </section>
        ))}

        <footer className="pi-footer">
          <p>{footerLabel}</p>
          <div>
            <Link href="/docs">Docs</Link>
            <a href={WORLDFOUNDRY_GITHUB_REPO} rel="noreferrer" target="_blank">
              Community
            </a>
          </div>
        </footer>
      </div>
    </main>
  );
}

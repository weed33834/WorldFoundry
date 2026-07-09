import {
  metricDocsPath,
  metricQuickNavGroups,
  metricQuickNavGroupsForPage,
  metricQuickNavLabels,
  metricQuickNavPageMeta,
  type MetricQuickNavPageId,
} from '@/lib/metric-quick-nav';
import type { Locale } from '@/lib/i18n';

type MetricQuickNavProps = {
  locale: Locale;
  page?: MetricQuickNavPageId;
  variant?: 'page' | 'hub';
};

export function MetricQuickNav({ locale, page, variant = page ? 'page' : 'hub' }: MetricQuickNavProps) {
  const labels = metricQuickNavLabels[locale];

  if (variant === 'hub') {
    const metricCount = metricQuickNavGroups.reduce((count, group) => count + group.items.length, 0);
    const referencePath = metricDocsPath(locale, '/evaluation/metrics/reference');

    return (
      <details className="pi-metric-quick-nav" id="metric-quick-nav">
        <summary className="pi-metric-quick-nav-summary">
          <span className="pi-metric-quick-nav-summary-label">{labels.hubTitle}</span>
          <span className="pi-metric-quick-nav-summary-meta">
            {labels.summary.replace('{count}', String(metricCount))}
          </span>
        </summary>

        <nav className="pi-metric-quick-nav-body" aria-label={labels.hubTitle}>
          <div className="pi-metric-quick-nav-sections">
            <span className="pi-metric-quick-nav-section-item">
              <a href={referencePath}>
                {locale === 'zh' ? 'Registry 与参考' : 'Registry & reference'}
              </a>
            </span>
          </div>

          <div className="pi-metric-quick-nav-groups">
            {(Object.keys(metricQuickNavPageMeta) as MetricQuickNavPageId[]).map((pageId) => {
              const meta = metricQuickNavPageMeta[pageId];
              const groups = metricQuickNavGroupsForPage(pageId);
              const count = groups.reduce((sum, group) => sum + group.items.length, 0);

              return (
                <section className="pi-metric-quick-nav-group" key={pageId}>
                  <h3 className="pi-metric-quick-nav-group-title">
                    <a href={metricDocsPath(locale, meta.slug)}>{meta.label[locale]}</a>
                  </h3>
                  <p className="pi-metric-quick-nav-group-desc">{meta.description[locale]}</p>
                  <ul className="pi-metric-quick-nav-list">
                    {groups.flatMap((group) =>
                      group.items.map((item) => (
                        <li key={`${pageId}-${item.id[locale]}`}>
                          <a href={`${metricDocsPath(locale, meta.slug)}#${item.id[locale]}`}>
                            {item.label[locale]}
                          </a>
                        </li>
                      )),
                    )}
                  </ul>
                  <p className="pi-metric-quick-nav-group-meta">
                    {count} {locale === 'zh' ? '个指标' : 'metrics'}
                  </p>
                </section>
              );
            })}
          </div>
        </nav>
      </details>
    );
  }

  const groups = page ? metricQuickNavGroupsForPage(page) : metricQuickNavGroups;
  const metricCount = groups.reduce((count, group) => count + group.items.length, 0);
  const basePath = page ? metricDocsPath(locale, metricQuickNavPageMeta[page].slug) : '';

  return (
    <details className="pi-metric-quick-nav" id="metric-quick-nav">
      <summary className="pi-metric-quick-nav-summary">
        <span className="pi-metric-quick-nav-summary-label">{labels.title}</span>
        <span className="pi-metric-quick-nav-summary-meta">
          {labels.summary.replace('{count}', String(metricCount))}
        </span>
      </summary>

      <nav className="pi-metric-quick-nav-body" aria-label={labels.title}>
        <div className="pi-metric-quick-nav-sections">
          <span className="pi-metric-quick-nav-section-item">
            <a href={metricDocsPath(locale, '/evaluation/metrics')}>
              {locale === 'zh' ? '指标概览' : 'Metrics overview'}
            </a>
          </span>
        </div>

        <div className="pi-metric-quick-nav-groups">
          {groups.map((group) => (
            <section className="pi-metric-quick-nav-group" key={group.id}>
              <h3 className="pi-metric-quick-nav-group-title">{group.label[locale]}</h3>
              <ul className="pi-metric-quick-nav-list">
                {group.items.map((item) => (
                  <li key={item.id[locale]}>
                    <a href={`${basePath}#${item.id[locale]}`}>{item.label[locale]}</a>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      </nav>
    </details>
  );
}

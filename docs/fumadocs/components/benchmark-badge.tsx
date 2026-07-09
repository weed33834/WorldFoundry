import { getBenchmarkBadgeLabel, type BenchmarkBadgeKind } from '@/lib/benchmark-catalog';
import type { Locale } from '@/lib/i18n';

type BenchmarkBadgeProps = {
  kind: BenchmarkBadgeKind;
  locale: Locale;
};

export function BenchmarkBadge({ kind, locale }: BenchmarkBadgeProps) {
  return (
    <span className={`pi-doc-badge pi-doc-badge-${kind}`}>{getBenchmarkBadgeLabel(kind, locale)}</span>
  );
}

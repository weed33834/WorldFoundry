import Link from 'next/link';
import type { ComponentPropsWithoutRef } from 'react';

type WordmarkVariant = 'compact' | 'header' | 'hero' | 'doc-hero';

type WorldFoundryWordmarkProps = {
  as?: 'span' | 'h1' | 'h2';
  variant?: WordmarkVariant;
  className?: string;
};

function joinClasses(...values: Array<string | false | undefined>) {
  return values.filter(Boolean).join(' ');
}

export function WorldFoundryWordmark({
  as: Tag = 'span',
  variant = 'header',
  className,
}: WorldFoundryWordmarkProps) {
  return (
    <Tag
      className={joinClasses('pi-brand-display', `pi-brand-display-${variant}`, className)}
      aria-label="WorldFoundry"
    >
      WorldFoundry
    </Tag>
  );
}

type WorldFoundryWordmarkLinkProps = {
  variant?: WordmarkVariant;
  href?: ComponentPropsWithoutRef<typeof Link>['href'];
  className?: string;
};

export function WorldFoundryWordmarkLink({
  variant = 'header',
  href = '/',
  className,
}: WorldFoundryWordmarkLinkProps) {
  return (
    <Link href={href} className={joinClasses('pi-wordmark', 'pi-brand-link', className)}>
      <WorldFoundryWordmark variant={variant} />
    </Link>
  );
}

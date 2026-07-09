import { withBasePath } from '@/lib/site-path';

const homeTeaserSrc = withBasePath('/teaser.png');
const docsTeaserSrc = withBasePath('/docs-teaser.png');

type TeaserImageProps = {
  alt: string;
  variant?: 'home' | 'docs';
};

export function TeaserImage({ alt, variant = 'docs' }: TeaserImageProps) {
  const src = variant === 'home' ? homeTeaserSrc : docsTeaserSrc;
  const image = (
    <img
      src={src}
      alt={alt}
      className={variant === 'home' ? 'w-full h-auto object-contain' : undefined}
    />
  );

  if (variant === 'home') {
    return (
      <div className="w-full overflow-hidden border-2 border-[var(--pi-ink)] shadow-[8px_8px_0_var(--pi-ink)] bg-white mb-20">
        {image}
      </div>
    );
  }

  return image;
}

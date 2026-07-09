'use client';

import { useEffect, useState } from 'react';

export function DocsReadingProgress() {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const main = document.querySelector<HTMLElement>('.pi-doc-main');
    if (!main) return;

    const update = () => {
      const scrollable = main.scrollHeight - main.clientHeight;
      setProgress(scrollable <= 0 ? 0 : Math.min(1, main.scrollTop / scrollable));
    };

    update();
    main.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);

    return () => {
      main.removeEventListener('scroll', update);
      window.removeEventListener('resize', update);
    };
  }, []);

  return (
    <div className="pi-doc-reading-progress" aria-hidden="true">
      <span style={{ width: `${Math.round(progress * 100)}%` }} />
    </div>
  );
}

'use client';

import { useEffect } from 'react';

function forwardWheelToMain(panel: HTMLElement, main: HTMLElement, event: WheelEvent) {
  const canScroll = panel.scrollHeight > panel.clientHeight + 1;

  if (!canScroll) {
    main.scrollTop += event.deltaY;
    event.preventDefault();
    return;
  }

  const atTop = panel.scrollTop <= 0;
  const atBottom = panel.scrollTop + panel.clientHeight >= panel.scrollHeight - 1;

  if ((event.deltaY < 0 && atTop) || (event.deltaY > 0 && atBottom)) {
    main.scrollTop += event.deltaY;
    event.preventDefault();
  }
}

function scrollMainToHash(main: HTMLElement, hash: string) {
  const id = decodeURIComponent(hash.replace(/^#/, ''));
  if (!id) return;

  const target = document.getElementById(id);
  if (!target || !main.contains(target)) return;

  const offset = 16;
  const mainRect = main.getBoundingClientRect();
  const targetRect = target.getBoundingClientRect();
  main.scrollTop += targetRect.top - mainRect.top - offset;
}

export function DocsScrollBridge() {
  useEffect(() => {
    const shell = document.querySelector<HTMLElement>('.pi-doc-shell');
    const main = document.querySelector<HTMLElement>('.pi-doc-main');
    if (!shell || !main) return;

    const panels = Array.from(
      document.querySelectorAll<HTMLElement>('.pi-doc-sidebar, .pi-doc-right-rail'),
    );

    const onShellWheel = (event: WheelEvent) => {
      if (!(event.target instanceof Node) || main.contains(event.target)) {
        return;
      }

      for (const panel of panels) {
        if (panel.contains(event.target)) {
          forwardWheelToMain(panel, main, event);
          return;
        }
      }

      main.scrollTop += event.deltaY;
      event.preventDefault();
    };

    const onAnchorClick = (event: MouseEvent) => {
      const target = event.target;
      if (!(target instanceof Element)) return;

      const anchor = target.closest<HTMLAnchorElement>('a[href^="#"]');
      if (!anchor || !shell.contains(anchor)) return;

      const hash = anchor.getAttribute('href');
      if (!hash || hash === '#') return;

      event.preventDefault();
      history.pushState(null, '', hash);
      scrollMainToHash(main, hash);
    };

    const onHashChange = () => scrollMainToHash(main, window.location.hash);

    const panelCleanups = panels.map((panel) => {
      const onPanelWheel = (event: WheelEvent) => forwardWheelToMain(panel, main, event);
      panel.addEventListener('wheel', onPanelWheel, { passive: false });
      return () => panel.removeEventListener('wheel', onPanelWheel);
    });

    shell.addEventListener('wheel', onShellWheel, { passive: false });
    shell.addEventListener('click', onAnchorClick);
    window.addEventListener('hashchange', onHashChange);

    if (window.location.hash) {
      requestAnimationFrame(() => scrollMainToHash(main, window.location.hash));
    }

    return () => {
      shell.removeEventListener('wheel', onShellWheel);
      shell.removeEventListener('click', onAnchorClick);
      window.removeEventListener('hashchange', onHashChange);
      panelCleanups.forEach((cleanup) => cleanup());
    };
  }, []);

  return null;
}

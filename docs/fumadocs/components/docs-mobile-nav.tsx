'use client';

import { useEffect, useState } from 'react';

type DocsMobileNavProps = {
  openLabel: string;
  closeLabel: string;
};

function getShell() {
  return document.querySelector<HTMLElement>('.pi-doc-shell');
}

export function DocsMobileNavToggle({ openLabel, closeLabel }: DocsMobileNavProps) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const shell = getShell();
    if (!shell) return;

    if (open) {
      shell.classList.add('pi-doc-nav-open');
      document.body.style.overflow = 'hidden';
    } else {
      shell.classList.remove('pi-doc-nav-open');
      document.body.style.overflow = '';
    }

    return () => {
      shell.classList.remove('pi-doc-nav-open');
      document.body.style.overflow = '';
    };
  }, [open]);

  useEffect(() => {
    const shell = getShell();
    if (!shell) return;

    const close = () => setOpen(false);
    shell.querySelectorAll<HTMLAnchorElement>('.pi-doc-sidebar a').forEach((link) => {
      link.addEventListener('click', close);
    });

    return () => {
      shell.querySelectorAll<HTMLAnchorElement>('.pi-doc-sidebar a').forEach((link) => {
        link.removeEventListener('click', close);
      });
    };
  }, [open]);

  return (
    <>
      <button
        type="button"
        className="pi-doc-menu-button"
        aria-expanded={open}
        aria-controls="pi-doc-sidebar"
        onClick={() => setOpen((value) => !value)}
      >
        {open ? closeLabel : openLabel}
      </button>
      {open ? (
        <button
          type="button"
          className="pi-doc-nav-backdrop"
          aria-label={closeLabel}
          onClick={() => setOpen(false)}
        />
      ) : null}
    </>
  );
}

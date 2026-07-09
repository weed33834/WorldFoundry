export type SidebarBadge = {
  kind: 'integrated' | 'normalizer' | 'planned' | 'blocked';
};

export type SidebarPageLink = {
  url: string;
  label: string;
};

export type SidebarNavPage = {
  type: 'page';
  link: SidebarPageLink;
  depth: 0 | 1 | 2;
  badges: SidebarBadge[];
};

export type SidebarNavDivider = {
  type: 'divider';
  label: string;
};

export type SidebarNavItem = SidebarNavPage | SidebarNavDivider;

export function isSidebarItemActive(item: SidebarNavPage, currentUrl: string) {
  return item.link.url === currentUrl;
}

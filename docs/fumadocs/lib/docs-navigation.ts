import type { Locale } from '@/lib/i18n';

export type DocsNavGroupId =
  | 'start'
  | 'inference'
  | 'training'
  | 'evaluation'
  | 'integration'
  | 'tools'
  | 'architecture'
  | 'maintainers';

export type DocsSlug = readonly string[];

export type DocsNavGroup = {
  id: DocsNavGroupId;
  slugs: readonly DocsSlug[];
};

export type DocsChromeLabels = {
  docs: string;
  home: string;
  kicker: string;
  language: string;
  markdown: string;
  nav: string;
  navGroups: Record<DocsNavGroupId, string>;
  openEnvision: string;
  onThisPage: string;
  previousPage: string;
  nextPage: string;
  relatedPages: string;
  expandBenchmarkList: string;
  collapseBenchmarkList: string;
  expandMetricsList: string;
  collapseMetricsList: string;
  expandArchitectureList: string;
  collapseArchitectureList: string;
  sidebar: string;
  source: string;
  lastUpdated: string;
  openMenu: string;
  closeMenu: string;
};

/** Sidebar labels keyed by slug path (e.g. `guides/inference`). Falls back to page title. */
export type DocsNavPageLabels = Partial<Record<string, string>>;

export const docsNavPageLabels: Record<Locale, DocsNavPageLabels> = {
  en: {
    'reference/environments': 'Environment',
    'guides/inference': 'Run inference',
    'guides/supported-models': 'Models',
    'guides/local-assets': 'Local assets',
    'guides/tui': 'TUI',
    'reference/cli': 'CLI',
    'guides/studio': 'Studio',
    evaluation: 'Overview',
    'evaluation/benchmark-hub': 'Benchmark Hub',
    'evaluation/metrics': 'Metrics',
    'evaluation/metrics/scorers': 'Scorers & quality',
    'evaluation/metrics/distribution': 'Distribution',
    'evaluation/metrics/perceptual': 'Perceptual',
    'evaluation/metrics/editing': 'Editing',
    'evaluation/metrics/reference': 'Registry',
    'evaluation/embodied-official-runtime': 'Embodied setup',
    'maintainers/plan': 'Plan',
  },
  zh: {
    'reference/environments': '环境配置',
    'guides/inference': '运行推理',
    'guides/supported-models': '模型',
    'guides/local-assets': '本地资产',
    'guides/tui': 'TUI',
    'reference/cli': 'CLI',
    'guides/studio': 'Studio',
    evaluation: '概览',
    'evaluation/benchmark-hub': 'Benchmark Hub',
    'evaluation/metrics': '指标',
    'evaluation/metrics/scorers': 'Scorer 与质量',
    'evaluation/metrics/distribution': '分布指标',
    'evaluation/metrics/perceptual': '感知成对',
    'evaluation/metrics/editing': '编辑',
    'evaluation/metrics/reference': 'Registry',
    'evaluation/embodied-official-runtime': 'Embodied 环境',
    'maintainers/plan': '规划',
  },
};

export function getNavPageLabel(slugs: readonly string[], locale: Locale, fallback: string) {
  const key = slugs.join('/');
  return docsNavPageLabels[locale][key] ?? fallback;
}

// Keep the first sidebar group user-facing: setup, assets, TUI, and CLI before
// deeper inference/evaluation reference material.
export const docsNavGroups = [
  {
    id: 'start',
    slugs: [
      [],
      ['quickstart'],
      ['reference', 'environments'],
      ['guides', 'local-assets'],
      ['guides', 'tui'],
      ['reference', 'cli'],
    ],
  },
  {
    id: 'inference',
    slugs: [
      ['guides', 'inference'],
      ['guides', 'supported-models'],
      ['guides', 'studio'],
    ],
  },

  {
    id: 'evaluation',
    slugs: [
      ['evaluation'],
      ['evaluation', 'benchmark-hub'],
      ['evaluation', 'metrics'],
      ['evaluation', 'embodied-official-runtime'],
    ],
  },
  {
    id: 'integration',
    slugs: [
      ['guides', 'add-model'],
      ['guides', 'add-benchmark'],
    ],
  },
  {
    id: 'tools',
    slugs: [],
  },
  {
    id: 'architecture',
    slugs: [['maintainers', 'architecture']],
  },
  {
    id: 'maintainers',
    slugs: [
      ['maintainers', 'contributing'],
      ['maintainers', 'plan'],
    ],
  },
] as const satisfies readonly DocsNavGroup[];

export const docsLabels: Record<Locale, DocsChromeLabels> = {
  en: {
    docs: 'Docs',
    home: 'Home',
    kicker: 'WorldFoundry docs',
    language: 'Language',
    markdown: 'Markdown',
    nav: 'Main navigation',
    navGroups: {
      architecture: 'Architecture',
      evaluation: 'Evaluation',
      inference: 'Inference',
      integration: 'Integration',
      maintainers: 'Maintainers',
      start: 'Start Here',
      tools: 'Tools',
      training: 'Training',
    },
    openEnvision: 'OpenEnvision',
    onThisPage: 'On this page',
    previousPage: 'Previous',
    nextPage: 'Next',
    relatedPages: 'Related',
    expandBenchmarkList: 'Expand benchmark list',
    collapseBenchmarkList: 'Collapse benchmark list',
    expandMetricsList: 'Expand metrics list',
    collapseMetricsList: 'Collapse metrics list',
    expandArchitectureList: 'Expand architecture list',
    collapseArchitectureList: 'Collapse architecture list',
    sidebar: 'Documentation',
    source: 'Source',
    lastUpdated: 'Last updated',
    openMenu: 'Menu',
    closeMenu: 'Close',
  },
  zh: {
    docs: '文档',
    home: '首页',
    kicker: 'WorldFoundry 文档',
    language: '语言',
    markdown: 'Markdown',
    nav: '主导航',
    navGroups: {
      architecture: '架构',
      evaluation: '评测',
      inference: '推理',
      integration: '接入',
      maintainers: '维护者',
      start: '先用起来',
      tools: '工具',
      training: '训练',
    },
    openEnvision: 'OpenEnvision',
    onThisPage: '本页内容',
    previousPage: '上一页',
    nextPage: '下一页',
    relatedPages: '相关页面',
    expandBenchmarkList: '展开 benchmark 列表',
    collapseBenchmarkList: '收起 benchmark 列表',
    expandMetricsList: '展开指标列表',
    collapseMetricsList: '收起指标列表',
    expandArchitectureList: '展开架构列表',
    collapseArchitectureList: '收起架构列表',
    sidebar: '文档',
    source: '源码',
    lastUpdated: '最后更新',
    openMenu: '菜单',
    closeMenu: '关闭',
  },
};

const tableDenseDocsPages = new Set([
  'evaluation/benchmark-hub',
  'guides/supported-models',
]);

export function isTableDenseDocsPage(slugs: readonly string[]) {
  return tableDenseDocsPages.has(slugs.join('/'));
}

export function isBenchmarkHubDocsPage(slugs: readonly string[]) {
  return slugs[0] === 'evaluation' && slugs[1] === 'benchmark-hub';
}

export function isMetricsDocsPage(slugs: readonly string[]) {
  return slugs[0] === 'evaluation' && slugs[1] === 'metrics';
}

export { getBenchmarkHubSectionLabel } from '@/lib/benchmark-catalog';

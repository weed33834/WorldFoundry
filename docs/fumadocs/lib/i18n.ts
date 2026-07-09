import { defineI18n } from 'fumadocs-core/i18n';

export const i18n = defineI18n({
  defaultLanguage: 'en',
  languages: ['en', 'zh'],
  hideLocale: 'default-locale',
});

export type Locale = (typeof i18n.languages)[number];

export const defaultLocale = i18n.defaultLanguage;

export const localeNames: Record<Locale, string> = {
  en: 'English',
  zh: '中文',
};

export function isLocale(value: string | undefined): value is Locale {
  return i18n.languages.includes(value as Locale);
}

export function isDefaultLocale(locale: string | undefined) {
  return locale === undefined || locale === defaultLocale;
}

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createMDX } from 'fumadocs-mdx/next';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const withMDX = createMDX();
const configuredBasePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
const basePath =
  configuredBasePath && configuredBasePath !== '/'
    ? `/${configuredBasePath.replace(/^\/+|\/+$/g, '')}`
    : '';

/** @type {import('next').NextConfig} */
const config = {
  allowedDevOrigins: ['127.0.0.1', 'localhost'],
  ...(basePath
    ? {
        assetPrefix: basePath,
        basePath,
      }
    : {}),
  output: 'export',
  outputFileTracingRoot: path.resolve(__dirname, '..', '..'),
  reactStrictMode: true,
  images: {
    unoptimized: true,
  },
};

export default withMDX(config);

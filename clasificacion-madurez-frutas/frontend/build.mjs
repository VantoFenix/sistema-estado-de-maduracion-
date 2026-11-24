import { mkdir, readFile, writeFile } from 'node:fs/promises';

const indexPath = new URL('./index.html', import.meta.url);
const outputPath = new URL('./dist/', import.meta.url);
const configuredUrl = process.env.API_BASE_URL?.trim();
let html = await readFile(indexPath, 'utf8');

if (configuredUrl) {
  const url = new URL(configuredUrl);
  if (url.protocol !== 'https:' || url.username || url.password ||
      url.pathname !== '/' || url.search || url.hash) {
    throw new Error('API_BASE_URL debe ser un origen HTTPS sin credenciales ni ruta.');
  }
  const placeholder = 'window.API_BASE_URL = window.API_BASE_URL || "";';
  if (!html.includes(placeholder)) {
    throw new Error('Falta el marcador de API_BASE_URL en index.html.');
  }
  html = html.replace(placeholder, () => `window.API_BASE_URL = ${JSON.stringify(url.origin)};`);
}

await mkdir(outputPath, { recursive: true });
await writeFile(new URL('./index.html', outputPath), html, 'utf8');

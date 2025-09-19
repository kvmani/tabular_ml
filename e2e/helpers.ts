import { Page } from '@playwright/test';

const ALLOWED_PROTOCOLS = ['http://127.0.0.1', 'http://localhost', 'https://127.0.0.1', 'https://localhost', 'data:', 'blob:'];

export async function blockExternalRequests(page: Page) {
  await page.route('**/*', (route) => {
    const url = route.request().url();
    const allowed = ALLOWED_PROTOCOLS.some((prefix) => url.startsWith(prefix));
    if (allowed) {
      return route.continue();
    }
    return route.abort();
  });
}

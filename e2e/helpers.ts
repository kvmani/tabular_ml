import type { Locator, Page, Response } from '@playwright/test';
import fs from 'fs/promises';
import path from 'path';

const ALLOWED_PROTOCOLS = ['http://127.0.0.1', 'http://localhost', 'https://127.0.0.1', 'https://localhost', 'data:', 'blob:'];

const EVENT_STREAM_HEADER = 'text/event-stream';

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

export async function ensureDirectory(targetDir: string) {
  await fs.mkdir(targetDir, { recursive: true });
}

export async function captureScreenshot(page: Page, filePath: string) {
  await ensureDirectory(path.dirname(filePath));
  await page.screenshot({ path: filePath, fullPage: true });
}

export async function waitForPlotlyCanvas(plotContainer: Locator, options: { timeout?: number } = {}) {
  const timeout = options.timeout ?? 20000;
  await plotContainer.locator('canvas, svg').first().waitFor({ state: 'visible', timeout });
}

type EventStreamMatcher = string | RegExp | ((url: string) => boolean);

interface WaitForEventStreamOptions {
  timeout?: number;
  optional?: boolean;
}

const normaliseMatcher = (matcher: EventStreamMatcher) => {
  if (typeof matcher === 'function') {
    return matcher;
  }
  if (matcher instanceof RegExp) {
    return (url: string) => matcher.test(url);
  }
  return (url: string) => url.includes(matcher);
};

export async function waitForEventStream(
  page: Page,
  matcher: EventStreamMatcher,
  options: WaitForEventStreamOptions = {}
): Promise<Response | null> {
  const predicate = normaliseMatcher(matcher);
  const timeout = options.timeout ?? 15000;
  try {
    const response = await page.waitForResponse((candidate) => {
      if (!predicate(candidate.url())) {
        return false;
      }
      const header = candidate.headers()['content-type'] || candidate.headers()['Content-Type'] || '';
      return header.includes(EVENT_STREAM_HEADER);
    }, { timeout });
    return response;
  } catch (error) {
    if (options.optional) {
      return null;
    }
    throw error;
  }
}

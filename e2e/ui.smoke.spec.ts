import { test, expect } from '@playwright/test';
import path from 'path';
import fs from 'fs/promises';

import { blockExternalRequests } from './helpers';

const ARTIFACT_ROOT = path.resolve(__dirname, '../artifacts/ui');

async function capture(page, name) {
  await fs.mkdir(ARTIFACT_ROOT, { recursive: true });
  const shotPath = path.join(ARTIFACT_ROOT, name);
  await page.screenshot({ path: shotPath, fullPage: true });
}

test.describe('offline UI smoke', () => {
  test('load dataset, split, train, evaluate', async ({ page, baseURL }) => {
    await blockExternalRequests(page);
    const targetUrl = baseURL ?? 'http://127.0.0.1:8000';

    await page.goto(targetUrl);
    await page.waitForSelector('[data-testid="dataset-selector"]');
    await capture(page, 'step-1-home.png');

    await page.getByTestId('load-sample-titanic').click();
    await expect(page.getByTestId('dataset-preview')).toBeVisible({ timeout: 10000 });
    await capture(page, 'step-2-sample.png');

    await page.getByTestId('split-target').selectOption('Survived');
    await page.getByTestId('create-split').click();
    await expect(page.getByText(/Using split:/)).toBeVisible({ timeout: 20000 });
    await capture(page, 'step-3-split.png');

    await page.getByTestId('train-target').selectOption('Survived');
    await page.getByTestId('train-button').click();
    await expect(page.getByText(/Model trained successfully/)).toBeVisible({ timeout: 30000 });
    await capture(page, 'step-4-train.png');

    await page.getByTestId('evaluate-button').click();
    await expect(page.getByTestId('metrics-summary')).toBeVisible({ timeout: 30000 });
    await capture(page, 'step-5-evaluate.png');

    const metricsText = (await page.getByTestId('metrics-summary').innerText()).toLowerCase();
    expect(metricsText).toContain('validation');
    expect(metricsText).toContain('test');

    await expect(page.getByTestId('confusion-matrix')).toBeVisible({ timeout: 30000 });
  });
});

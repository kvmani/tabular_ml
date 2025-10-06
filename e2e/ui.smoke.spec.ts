import { test, expect } from '@playwright/test';
import path from 'path';

import {
  blockExternalRequests,
  captureScreenshot,
  waitForEventStream,
  waitForPlotlyCanvas
} from './helpers';

const SCREENSHOT_ROOT = path.resolve(__dirname, '../docs/screenshots/2025-10-06_e2e-smoke');
const FIXTURE_PATH = path.resolve(__dirname, './fixtures/iris_small.csv');

test.describe('offline UI smoke', () => {
  test('default preview, upload, visualise, train, and evaluate', async ({ page, baseURL }) => {
    await blockExternalRequests(page);
    const targetUrl = baseURL ?? 'http://127.0.0.1:8000';

    await page.goto(targetUrl, { waitUntil: 'networkidle' });

    const datasetPreview = page.getByTestId('dataset-preview');
    await expect(datasetPreview).toBeVisible({ timeout: 30000 });
    await expect(datasetPreview).toContainText(/PassengerId/i, { timeout: 30000 });
    const columnList = page.getByTestId('column-list');
    await expect(columnList).toContainText(/Survived/i);
    await captureScreenshot(page, path.join(SCREENSHOT_ROOT, '01_home.png'));

    const uploadForm = page.locator('form.upload-form');
    await uploadForm.locator('input[type="text"]').fill('Iris fixture (Playwright)');
    await uploadForm.locator('input[type="file"]').setInputFiles(FIXTURE_PATH);
    await uploadForm.locator('button:has-text("Upload dataset")').click();
    await expect(page.locator('.notification')).toContainText(/Uploaded Iris fixture/i, { timeout: 45000 });
    await expect(datasetPreview).toContainText(/sepal_length/i, { timeout: 45000 });
    await expect(columnList).toContainText(/species/i, { timeout: 45000 });
    await captureScreenshot(page, path.join(SCREENSHOT_ROOT, '02_dataset_uploaded.png'));

    const histogramSection = page.locator('.viz-grid > div').first();
    await expect(histogramSection.locator('select option[value="sepal_length"]')).toBeVisible({ timeout: 15000 });
    await histogramSection.locator('select').first().selectOption('sepal_length');
    await histogramSection.locator('button:has-text("Generate histogram")').click();
    const histogramPlot = histogramSection.locator('.js-plotly-plot').first();
    await waitForPlotlyCanvas(histogramPlot);
    await captureScreenshot(page, path.join(SCREENSHOT_ROOT, '03_histogram.png'));

    const scatterSection = page.locator('.viz-grid > div').nth(1);
    await scatterSection.locator('select').nth(0).selectOption('sepal_length');
    await scatterSection.locator('select').nth(1).selectOption('petal_length');
    await scatterSection.locator('select').nth(2).selectOption('species');
    await scatterSection.locator('button:has-text("Generate scatter")').click();
    const scatterPlot = scatterSection.locator('.js-plotly-plot').first();
    await waitForPlotlyCanvas(scatterPlot);
    await captureScreenshot(page, path.join(SCREENSHOT_ROOT, '04_scatter.png'));

    await page.getByTestId('train-target').selectOption('species');
    const eventStreamPromise = waitForEventStream(page, '/model/train/stream', {
      optional: true,
      timeout: 10000
    });
    await page.getByTestId('train-button').click();
    await captureScreenshot(page, path.join(SCREENSHOT_ROOT, '05_training-start.png'));
    await eventStreamPromise;
    const statusIndicator = page.getByTestId('training-status');
    await expect(statusIndicator).toBeVisible({ timeout: 30000 });
    const livePlotContainer = page.getByTestId('training-history-plot');
    await waitForPlotlyCanvas(livePlotContainer.locator('.js-plotly-plot').first(), { timeout: 120000 });
    await expect(page.getByTestId('latest-epoch')).toContainText(/Latest update/i, { timeout: 120000 });
    await expect(page.locator('.notification')).toContainText(/Model trained successfully/i, { timeout: 120000 });
    await expect(page.getByTestId('metrics-summary')).toBeVisible({ timeout: 120000 });
    await captureScreenshot(page, path.join(SCREENSHOT_ROOT, '06_training-complete.png'));

    await page.getByTestId('evaluate-button').click();
    await expect(page.getByTestId('metrics-summary')).toContainText(/Validation/i, { timeout: 120000 });
    const evaluationPlots = page.locator('.evaluation-grid .js-plotly-plot');
    await waitForPlotlyCanvas(evaluationPlots.first());
    await captureScreenshot(page, path.join(SCREENSHOT_ROOT, '07_evaluation.png'));
    await expect(page.getByTestId('confusion-matrix')).toBeVisible({ timeout: 120000 });
    await captureScreenshot(page, path.join(SCREENSHOT_ROOT, '08_confusion-matrix.png'));
  });
});

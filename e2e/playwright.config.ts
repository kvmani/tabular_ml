import { defineConfig, devices } from '@playwright/test';
import path from 'path';

const ARTIFACTS_DIR = path.resolve(__dirname, '../artifacts/ui');

export default defineConfig({
  testDir: './e2e',
  timeout: 120000,
  outputDir: path.resolve(ARTIFACTS_DIR, 'output'),
  reporter: [
    ['list'],
    ['html', { open: 'never', outputFolder: path.resolve(ARTIFACTS_DIR, 'report') }]
  ],
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:5173',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    viewport: { width: 1440, height: 900 }
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] }
    }
  ]
});

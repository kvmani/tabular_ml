import { defineConfig, devices } from '@playwright/test';
import path from 'path';

const PROJECT_ROOT = path.resolve(__dirname, '..');
const ARTIFACTS_DIR = path.resolve(PROJECT_ROOT, 'artifacts/ui');

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
  webServer: [
    {
      command: 'uvicorn backend.app.main:app --host 127.0.0.1 --port 8000',
      url: 'http://127.0.0.1:8000/health',
      reuseExistingServer: !process.env.CI,
      timeout: 120000,
      cwd: PROJECT_ROOT
    },
    {
      command: 'npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173',
      url: 'http://127.0.0.1:5173',
      reuseExistingServer: !process.env.CI,
      timeout: 120000,
      cwd: PROJECT_ROOT
    }
  ],
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] }
    }
  ]
});

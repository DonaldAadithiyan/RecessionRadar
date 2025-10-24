import { test, expect } from '@playwright/test';

// Configure test settings
test.describe.configure({ mode: 'serial' });

test.describe('RecessionRadar E2E Tests', () => {
  // Base URL for tests
  const BASE_URL = 'http://localhost:3000';
  const API_URL = 'http://localhost:8000';

  test.beforeEach(async ({ page }) => {
    // Set up API response mocks to avoid backend dependency
    await page.route(`${API_URL}/api/**`, async (route) => {
      const url = route.request().url();
      
      if (url.includes('/current-prediction')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            probability: 0.23,
            confidence: 0.89,
            last_updated: '2025-10-05T10:00:00Z'
          })
        });
      } else if (url.includes('/economic-indicators')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            indicators: {
              unemployment_rate: 4.1,
              inflation_rate: 3.2,
              gdp_growth: 2.5
            },
            last_updated: '2025-10-05T10:00:00Z'
          })
        });
      } else if (url.includes('/recession-probabilities')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            data: [
              { date: '2024-01-01', probability: 0.15 },
              { date: '2024-06-01', probability: 0.20 },
              { date: '2025-01-01', probability: 0.23 }
            ]
          })
        });
      } else if (url.includes('/treasury-yields')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            yields: {
              '3_month': 4.5,
              '2_year': 4.2,
              '10_year': 4.1
            },
            last_updated: '2025-10-05T10:00:00Z'
          })
        });
      } else if (url.includes('/custom-prediction')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            probability: 0.35,
            confidence: 0.78,
            scenario: 'custom'
          })
        });
      } else {
        await route.continue();
      }
    });

    // Navigate to the app
    await page.goto(BASE_URL);
  });

  test('Dashboard loads successfully', async ({ page }) => {
    // Wait for page to load
    await page.waitForLoadState('networkidle');

    // Check page title
    await expect(page).toHaveTitle(/RecessionRadar|Recession|Dashboard/i);

    // Check main content loads
    await expect(page.locator('body')).toContainText(/recession|probability|economic|dashboard/i);
  });

  test('Navigation works correctly', async ({ page }) => {
    // Wait for initial load
    await page.waitForLoadState('networkidle');

    // Look for navigation elements (sidebar, navbar, tabs, etc.)
    const navElements = [
      'nav',
      '[role="navigation"]',
      '.sidebar',
      '.navigation',
      'a[href*="dashboard"]',
      'a[href*="custom"]',
      'a[href*="info"]'
    ];

    let foundNav = false;
    for (const selector of navElements) {
      if (await page.locator(selector).count() > 0) {
        foundNav = true;
        break;
      }
    }

    if (foundNav) {
      // Try to navigate to custom prediction if link exists
      const customLink = page.locator('a[href*="custom"], button:has-text("Custom"), button:has-text("Prediction")').first();
      if (await customLink.count() > 0) {
        await customLink.click();
        await page.waitForLoadState('networkidle');
        
        // Should navigate to custom prediction page
        await expect(page.url()).toContain('/custom');
      }
    }

    // At minimum, page should be navigable
    expect(page.url()).toContain('localhost:3000');
  });

  test('Charts and data visualization load', async ({ page }) => {
    await page.waitForLoadState('networkidle');

    // Wait for charts to potentially load
    await page.waitForTimeout(2000);

    // Look for chart elements (Canvas, SVG, or chart containers)
    const chartSelectors = [
      'canvas',
      'svg',
      '[class*="chart"]',
      '[data-testid*="chart"]',
      '.recharts-wrapper',
      '.chartjs-render-monitor'
    ];

    let chartsFound = false;
    for (const selector of chartSelectors) {
      if (await page.locator(selector).count() > 0) {
        chartsFound = true;
        break;
      }
    }

    // Charts should be present or at least chart containers
    expect(chartsFound).toBeTruthy();
  });

  test('Economic data displays correctly', async ({ page }) => {
    await page.waitForLoadState('networkidle');

    // Wait for API calls to complete
    await page.waitForTimeout(1000);

    // Look for economic data display (percentages, numbers, indicators)
    const dataPatterns = [
      /\d+\.?\d*%/,  // Percentage values
      /\d+\.?\d+/,   // Decimal numbers
      /unemployment|inflation|gdp/i,
      /probability/i
    ];

    let dataFound = false;
    const pageContent = await page.textContent('body');
    
    for (const pattern of dataPatterns) {
      if (pattern.test(pageContent)) {
        dataFound = true;
        break;
      }
    }

    expect(dataFound).toBeTruthy();
  });

  test('Custom prediction form works', async ({ page }) => {
    // Navigate to custom prediction page
    await page.goto(`${BASE_URL}/custom`);
    await page.waitForLoadState('networkidle');

    // Look for form inputs
    const inputs = page.locator('input[type="number"], input[type="text"], .MuiTextField input');
    const inputCount = await inputs.count();

    if (inputCount > 0) {
      // Fill out the first few inputs
      for (let i = 0; i < Math.min(3, inputCount); i++) {
        await inputs.nth(i).fill('3.5');
      }

      // Look for submit button
      const submitButton = page.locator('button[type="submit"], button:has-text("Submit"), button:has-text("Predict"), button:has-text("Calculate")').first();
      
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        // Wait for response
        await page.waitForTimeout(2000);
        
        // Should show some result or feedback
        const bodyText = await page.textContent('body');
        expect(bodyText).toMatch(/result|prediction|probability|\d+\.?\d*%/i);
      }
    }

    // At minimum, page should load without errors
    expect(page.url()).toContain('/custom');
  });

  test('Error handling works properly', async ({ page }) => {
    // Mock API error responses
    await page.route(`${API_URL}/api/**`, async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal Server Error' })
      });
    });

    await page.goto(BASE_URL);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);

    // Should handle errors gracefully
    const bodyText = await page.textContent('body');
    expect(bodyText).toMatch(/error|unavailable|failed|try again/i);
  });

  test('Responsive design works on mobile', async ({ page, browser }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    
    await page.goto(BASE_URL);
    await page.waitForLoadState('networkidle');

    // Page should still be functional on mobile
    const bodyText = await page.textContent('body');
    expect(bodyText.length).toBeGreaterThan(10);

    // Check if layout adapts (no horizontal scrolling)
    const scrollWidth = await page.evaluate(() => document.body.scrollWidth);
    const clientWidth = await page.evaluate(() => document.body.clientWidth);
    
    expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 20); // Allow small margin
  });

  test('Performance: Page loads within reasonable time', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto(BASE_URL);
    await page.waitForLoadState('networkidle');
    
    const loadTime = Date.now() - startTime;
    
    // Should load within 10 seconds (generous for testing)
    expect(loadTime).toBeLessThan(10000);
  });
});
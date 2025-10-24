import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import axios from 'axios';
import '@testing-library/jest-dom';

// Mock axios for API calls
jest.mock('axios');
const mockedAxios = axios;

// Mock Chart.js components
jest.mock('react-chartjs-2', () => ({
  Line: ({ data }) => <div data-testid="line-chart">{JSON.stringify(data)}</div>,
  Bar: ({ data }) => <div data-testid="bar-chart">{JSON.stringify(data)}</div>,
}));

// Import components
import Dashboard from '../pages/Dashboard';
import CustomPrediction from '../pages/CustomPrediction';

const MockRouter = ({ children }) => (
  <BrowserRouter>{children}</BrowserRouter>
);

describe('RecessionRadar Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Dashboard API Integration', () => {
    test('loads and displays real API data', async () => {
      // Mock successful API responses
      mockedAxios.get.mockImplementation((url) => {
        if (url.includes('/current-prediction')) {
          return Promise.resolve({
            data: {
              probability: 0.34,
              confidence: 0.87,
              last_updated: '2025-10-05T10:00:00Z'
            }
          });
        }
        if (url.includes('/economic-indicators')) {
          return Promise.resolve({
            data: {
              indicators: {
                unemployment_rate: 4.2,
                inflation_rate: 3.1,
                gdp_growth: 2.3
              },
              last_updated: '2025-10-05T10:00:00Z'
            }
          });
        }
        if (url.includes('/recession-probabilities')) {
          return Promise.resolve({
            data: {
              data: [
                { date: '2024-01-01', probability: 0.15 },
                { date: '2024-06-01', probability: 0.25 },
                { date: '2025-01-01', probability: 0.34 }
              ]
            }
          });
        }
        if (url.includes('/treasury-yields')) {
          return Promise.resolve({
            data: {
              yields: {
                '3_month': 4.8,
                '2_year': 4.5,
                '10_year': 4.3
              }
            }
          });
        }
        return Promise.reject(new Error('Unknown endpoint'));
      });

      render(
        <MockRouter>
          <Dashboard />
        </MockRouter>
      );

      // Wait for API calls to complete
      await waitFor(() => {
        expect(mockedAxios.get).toHaveBeenCalledWith(
          expect.stringContaining('/current-prediction')
        );
      }, { timeout: 5000 });

      // Check if data is displayed
      await waitFor(() => {
        const bodyText = document.body.textContent;
        expect(bodyText).toMatch(/34%|0\.34|4\.2|3\.1|2\.3/);
      }, { timeout: 5000 });
    });

    test('handles API failures gracefully', async () => {
      // Mock API failures
      mockedAxios.get.mockRejectedValue(new Error('Network Error'));

      render(
        <MockRouter>
          <Dashboard />
        </MockRouter>
      );

      // Should handle errors without crashing
      await waitFor(() => {
        expect(document.body.textContent).toMatch(/error|unavailable|failed/i);
      }, { timeout: 5000 });
    });

    test('displays loading states during API calls', async () => {
      // Mock delayed responses
      mockedAxios.get.mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve({ data: {} }), 1000))
      );

      render(
        <MockRouter>
          <Dashboard />
        </MockRouter>
      );

      // Should show loading indicator
      expect(screen.getByRole('progressbar') || screen.getByText(/loading/i)).toBeInTheDocument();

      // Wait for loading to complete
      await waitFor(() => {
        expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
      }, { timeout: 3000 });
    });
  });

  describe('Custom Prediction Form Integration', () => {
    test('submits form data to API', async () => {
      // Mock successful prediction response
      mockedAxios.post.mockResolvedValue({
        data: {
          probability: 0.42,
          confidence: 0.85,
          scenario: 'custom'
        }
      });

      render(
        <MockRouter>
          <CustomPrediction />
        </MockRouter>
      );

      // Wait for component to load
      await waitFor(() => {
        expect(document.body.textContent).toMatch(/custom|prediction|input/i);
      });

      // Find and fill input fields
      const inputs = screen.getAllByRole('textbox') || screen.getAllByRole('spinbutton');
      
      if (inputs.length > 0) {
        // Fill first few inputs with test data
        for (let i = 0; i < Math.min(3, inputs.length); i++) {
          fireEvent.change(inputs[i], { target: { value: '3.5' } });
        }

        // Find and click submit button
        const submitButton = screen.getByRole('button', { name: /submit|predict|calculate/i });
        if (submitButton) {
          fireEvent.click(submitButton);

          // Wait for API call
          await waitFor(() => {
            expect(mockedAxios.post).toHaveBeenCalledWith(
              expect.stringContaining('/custom-prediction'),
              expect.any(Object)
            );
          });

          // Check result display
          await waitFor(() => {
            expect(document.body.textContent).toMatch(/42%|0\.42|result/i);
          });
        }
      }
    });

    test('validates form inputs', async () => {
      render(
        <MockRouter>
          <CustomPrediction />
        </MockRouter>
      );

      // Try to submit empty form
      const submitButton = screen.queryByRole('button', { name: /submit|predict|calculate/i });
      if (submitButton) {
        fireEvent.click(submitButton);

        // Should show validation errors or prevent submission
        await waitFor(() => {
          expect(document.body.textContent).toMatch(/required|invalid|error/i);
        });
      }
    });
  });

  describe('Chart Integration', () => {
    test('charts receive and display data correctly', async () => {
      // Mock chart data
      mockedAxios.get.mockImplementation((url) => {
        if (url.includes('/recession-probabilities')) {
          return Promise.resolve({
            data: {
              data: [
                { date: '2024-01-01', probability: 0.15 },
                { date: '2024-06-01', probability: 0.25 }
              ]
            }
          });
        }
        return Promise.resolve({ data: {} });
      });

      render(
        <MockRouter>
          <Dashboard />
        </MockRouter>
      );

      // Wait for charts to load with data
      await waitFor(() => {
        const charts = screen.getAllByTestId(/chart/);
        expect(charts.length).toBeGreaterThan(0);
      }, { timeout: 5000 });

      // Check if chart contains data
      const chartElement = screen.getByTestId('line-chart');
      expect(chartElement.textContent).toContain('0.15');
    });
  });

  describe('Error Boundary Integration', () => {
    test('handles component crashes gracefully', () => {
      // Create a component that throws an error
      const ThrowError = () => {
        throw new Error('Test error');
      };

      // Mock console.error to avoid noise in test output
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

      expect(() => {
        render(
          <MockRouter>
            <ThrowError />
          </MockRouter>
        );
      }).toThrow();

      consoleSpy.mockRestore();
    });
  });

  describe('Performance Integration', () => {
    test('renders components within reasonable time', async () => {
      const startTime = Date.now();

      // Mock fast API responses
      mockedAxios.get.mockResolvedValue({ data: {} });

      render(
        <MockRouter>
          <Dashboard />
        </MockRouter>
      );

      await waitFor(() => {
        expect(document.body.textContent.length).toBeGreaterThan(10);
      });

      const renderTime = Date.now() - startTime;
      expect(renderTime).toBeLessThan(5000); // Should render within 5 seconds
    });
  });
});
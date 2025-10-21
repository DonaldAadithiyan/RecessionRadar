import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import '@testing-library/jest-dom';

// Import components to test
import Dashboard from '../../../react-frontend/src/pages/Dashboard';
import CustomPrediction from '../../../react-frontend/src/pages/CustomPrediction';
import MetricsCards from '../../../react-frontend/src/components/MetricsCards';

// Mock API services
jest.mock('../../../react-frontend/src/services/api', () => ({
  getTreasuryYields: jest.fn(),
  getRecessionProbabilities: jest.fn(),
  getCurrentPrediction: jest.fn(),
  getEconomicIndicators: jest.fn(),
  getCustomPrediction: jest.fn(),
}));

// Mock Chart.js to avoid canvas rendering issues in tests
jest.mock('react-chartjs-2', () => ({
  Line: () => <div data-testid="line-chart">Mock Line Chart</div>,
  Bar: () => <div data-testid="bar-chart">Mock Bar Chart</div>,
}));

const MockRouter = ({ children }) => (
  <BrowserRouter>{children}</BrowserRouter>
);

describe('RecessionRadar UI Components', () => {
  beforeEach(() => {
    // Reset all mocks before each test
    jest.clearAllMocks();
  });

  describe('Dashboard Component', () => {
    test('renders main dashboard title', async () => {
      const { getTreasuryYields, getRecessionProbabilities, getCurrentPrediction, getEconomicIndicators } = require('../services/api');
      
      // Mock API responses
      getTreasuryYields.mockResolvedValue({ data: [] });
      getRecessionProbabilities.mockResolvedValue({ data: [] });
      getCurrentPrediction.mockResolvedValue({ probability: 0.25 });
      getEconomicIndicators.mockResolvedValue({ indicators: {} });

      render(
        <MockRouter>
          <Dashboard />
        </MockRouter>
      );

      // Check if dashboard title elements are present
      await waitFor(() => {
        expect(screen.getByText(/RecessionRadar/i) || screen.getByText(/Dashboard/i) || screen.getByText(/Economic/i)).toBeInTheDocument();
      });
    });

    test('displays loading state initially', () => {
      render(
        <MockRouter>
          <Dashboard />
        </MockRouter>
      );

      // Should show loading spinner or progress indicator
      expect(screen.getByRole('progressbar') || screen.getByText(/loading/i)).toBeInTheDocument();
    });

    test('handles API error gracefully', async () => {
      const { getTreasuryYields, getRecessionProbabilities, getCurrentPrediction, getEconomicIndicators } = require('../services/api');
      
      // Mock API errors
      getTreasuryYields.mockRejectedValue(new Error('API Error'));
      getRecessionProbabilities.mockRejectedValue(new Error('API Error'));
      getCurrentPrediction.mockRejectedValue(new Error('API Error'));
      getEconomicIndicators.mockRejectedValue(new Error('API Error'));

      render(
        <MockRouter>
          <Dashboard />
        </MockRouter>
      );

      // Should handle errors gracefully - look for error messages or fallback content
      await waitFor(() => {
        expect(
          screen.getByText(/error/i) || 
          screen.getByText(/unavailable/i) || 
          screen.getByText(/failed/i) ||
          screen.getByRole('alert')
        ).toBeInTheDocument();
      }, { timeout: 3000 });
    });
  });

  describe('MetricsCards Component', () => {
    test('renders metrics cards with mock data', () => {
      const mockData = {
        current_probability: 0.25,
        last_updated: '2025-10-05T10:00:00Z',
        economic_indicators: {
          unemployment_rate: 4.1,
          inflation_rate: 3.2
        }
      };

      render(<MetricsCards data={mockData} />);

      // Check if metrics are displayed
      expect(screen.getByText(/25%|0.25/) || screen.getByText(/probability/i)).toBeInTheDocument();
    });

    test('handles missing data gracefully', () => {
      render(<MetricsCards data={null} />);

      // Should not crash and might show placeholder or N/A values
      expect(screen.getByText(/N\/A|--|-|unavailable/i) || screen.getByTestId('metrics-placeholder')).toBeInTheDocument();
    });
  });

  describe('CustomPrediction Component', () => {
    test('renders custom prediction form', () => {
      render(
        <MockRouter>
          <CustomPrediction />
        </MockRouter>
      );

      // Should have form elements for custom prediction
      expect(
        screen.getByRole('button', { name: /predict|submit|calculate/i }) ||
        screen.getByText(/custom|prediction|input/i)
      ).toBeInTheDocument();
    });

    test('form accepts user input', () => {
      render(
        <MockRouter>
          <CustomPrediction />
        </MockRouter>
      );

      // Should have input fields for economic indicators
      const inputs = screen.getAllByRole('textbox') || screen.getAllByRole('spinbutton');
      expect(inputs.length).toBeGreaterThan(0);
    });
  });

  describe('Chart Components Integration', () => {
    test('charts render without crashing', () => {
      const mockChartData = {
        labels: ['Jan', 'Feb', 'Mar'],
        datasets: [{
          label: 'Test Data',
          data: [1, 2, 3]
        }]
      };

      render(
        <MockRouter>
          <Dashboard />
        </MockRouter>
      );

      // Mock charts should render
      expect(screen.getByTestId('line-chart') || screen.getByTestId('bar-chart')).toBeInTheDocument();
    });
  });

  describe('Navigation and Routing', () => {
    test('dashboard page renders correctly', () => {
      render(
        <MockRouter>
          <Dashboard />
        </MockRouter>
      );

      // Dashboard should render without navigation errors
      expect(document.body).toContainElement(screen.getByRole('main') || document.querySelector('[class*="dashboard"]') || document.body.firstChild);
    });
  });

  describe('Responsive Design Elements', () => {
    test('renders Material-UI components correctly', () => {
      render(
        <MockRouter>
          <Dashboard />
        </MockRouter>
      );

      // Should use Material-UI components (they have specific classes or attributes)
      expect(document.querySelector('[class*="MuiContainer"]') || document.querySelector('[class*="MuiGrid"]')).toBeInTheDocument();
    });
  });
});
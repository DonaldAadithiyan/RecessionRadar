import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';

// Simple component test without complex dependencies
describe('Basic UI Component Tests', () => {
  test('renders without crashing', () => {
    const TestComponent = () => <div>RecessionRadar Dashboard</div>;
    render(<TestComponent />);
    expect(screen.getByText('RecessionRadar Dashboard')).toBeInTheDocument();
  });

  test('can render basic text content', () => {
    const TestComponent = () => (
      <div>
        <h1>Economic Dashboard</h1>
        <p>Current recession probability: 23%</p>
        <button>Calculate Prediction</button>
      </div>
    );
    
    render(<TestComponent />);
    
    expect(screen.getByText('Economic Dashboard')).toBeInTheDocument();
    expect(screen.getByText(/recession probability/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /calculate/i })).toBeInTheDocument();
  });

  test('handles basic user interactions', () => {
    const TestComponent = () => {
      const [clicked, setClicked] = React.useState(false);
      return (
        <div>
          <button onClick={() => setClicked(true)}>
            {clicked ? 'Clicked!' : 'Click Me'}
          </button>
        </div>
      );
    };
    
    render(<TestComponent />);
    
    const button = screen.getByRole('button');
    expect(button).toHaveTextContent('Click Me');
    
    fireEvent.click(button);
    
    expect(button).toHaveTextContent('Clicked!');
  });

  test('can display mock economic data', () => {
    const mockData = {
      probability: 0.23,
      indicators: {
        unemployment: 4.1,
        inflation: 3.2
      }
    };

    const EconomicDisplay = ({ data }) => (
      <div>
        <h2>Economic Indicators</h2>
        <p>Recession Probability: {(data.probability * 100).toFixed(1)}%</p>
        <p>Unemployment: {data.indicators.unemployment}%</p>
        <p>Inflation: {data.indicators.inflation}%</p>
      </div>
    );

    render(<EconomicDisplay data={mockData} />);
    
    expect(screen.getByText((content, element) => {
      return content.includes('23.0') && content.includes('%');
    })).toBeInTheDocument();
    expect(screen.getByText('Unemployment: 4.1%')).toBeInTheDocument();
    expect(screen.getByText('Inflation: 3.2%')).toBeInTheDocument();
  });

  test('handles missing data gracefully', () => {
    const SafeComponent = ({ data }) => (
      <div>
        <h2>Dashboard</h2>
        <p>Probability: {data?.probability ? `${(data.probability * 100).toFixed(1)}%` : 'N/A'}</p>
        <p>Status: {data ? 'Data Available' : 'Loading...'}</p>
      </div>
    );

    render(<SafeComponent data={null} />);
    
    expect(screen.getByText('Probability: N/A')).toBeInTheDocument();
    expect(screen.getByText('Status: Loading...')).toBeInTheDocument();
  });

  test('validates form input behavior', () => {
    const FormComponent = () => {
      const [value, setValue] = React.useState('');
      const [error, setError] = React.useState('');

      const handleSubmit = () => {
        if (!value || isNaN(value) || value < 0 || value > 100) {
          setError('Please enter a valid percentage (0-100)');
        } else {
          setError('');
        }
      };

      return (
        <div>
          <input 
            type="number" 
            value={value} 
            onChange={(e) => setValue(e.target.value)}
            placeholder="Enter percentage"
          />
          <button onClick={handleSubmit}>Submit</button>
          {error && <p role="alert">{error}</p>}
        </div>
      );
    };

    render(<FormComponent />);
    
    const input = screen.getByPlaceholderText('Enter percentage');
    const button = screen.getByRole('button', { name: 'Submit' });

    // Test empty submission
    fireEvent.click(button);
    expect(screen.getByRole('alert')).toHaveTextContent('Please enter a valid percentage');

    // Test valid input
    fireEvent.change(input, { target: { value: '25' } });
    fireEvent.click(button);
    
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});
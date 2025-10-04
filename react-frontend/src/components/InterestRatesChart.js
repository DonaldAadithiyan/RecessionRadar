import React, { useState, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import { Paper, Typography, Box, Button, ButtonGroup } from '@mui/material';
import { getHistoricalEconomicData } from '../services/api';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  zoomPlugin
);

const InterestRatesChart = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedRange, setSelectedRange] = useState('ALL');
  const chartRef = useRef();

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const result = await getHistoricalEconomicData();
      setData(result);
      setError(null);
    } catch (err) {
      setError('Failed to load interest rates data');
      console.error('Error fetching interest rates data:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRangeChange = (months) => {
    setSelectedRange(months === 6 ? '6M' : months === 12 ? '1Y' : months === 24 ? '2Y' : months === 60 ? '5Y' : months === 120 ? '10Y' : 'ALL');
    // Reset zoom when changing time range for better UX
    setTimeout(() => {
      if (chartRef.current) {
        chartRef.current.resetZoom();
      }
    }, 100);
  };

  const resetZoom = () => {
    if (chartRef.current) {
      chartRef.current.resetZoom();
    }
    setSelectedRange('ALL');
  };

  if (loading) return <div>Loading interest rates data...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!data) return <div>No data available</div>;

  // Filter data based on selected time range
  const getFilteredData = () => {
    if (selectedRange === 'ALL') return data;

    const monthsMap = { '6M': 6, '1Y': 12, '2Y': 24, '5Y': 60, '10Y': 120 };
    const months = monthsMap[selectedRange];
    
    if (!months) return data;

    const cutoffDate = new Date();
    cutoffDate.setMonth(cutoffDate.getMonth() - months);
    
    const startIndex = data.dates.findIndex(dateStr => new Date(dateStr) >= cutoffDate);
    const filteredStartIndex = startIndex === -1 ? Math.max(0, data.dates.length - months) : startIndex;
    
    return {
      dates: data.dates.slice(filteredStartIndex),
      three_months_rate: data.three_months_rate.slice(filteredStartIndex),
      six_months_rate: data.six_months_rate.slice(filteredStartIndex),
      one_year_rate: data.one_year_rate.slice(filteredStartIndex),
      ten_year_rate: data.ten_year_rate.slice(filteredStartIndex)
    };
  };

  const filteredData = getFilteredData();

  const chartData = {
    labels: filteredData.dates.map(date => new Date(date).toLocaleDateString()),
    datasets: [
      {
        label: '3-Month Treasury Rate',
        data: filteredData.three_months_rate,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        tension: 0.1,
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
      },
      {
        label: '6-Month Treasury Rate',
        data: filteredData.six_months_rate,
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        tension: 0.1,
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
      },
      {
        label: '1-Year Treasury Rate',
        data: filteredData.one_year_rate,
        borderColor: 'rgb(255, 205, 86)',
        backgroundColor: 'rgba(255, 205, 86, 0.5)',
        tension: 0.1,
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
      },
      {
        label: '10-Year Treasury Rate',
        data: filteredData.ten_year_rate,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        tension: 0.1,
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      title: {
        display: true,
        text: 'Historical Interest Rates (1967-2025)',
        color: '#ffffff',
        font: { size: 16 }
      },
      legend: {
        position: 'top',
        labels: { color: '#ffffff' }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        callbacks: {
          title: function(context) {
            return `Date: ${context[0].label}`;
          },
          label: function(context) {
            const value = typeof context.parsed.y === 'number' ? context.parsed.y.toFixed(2) : context.parsed.y;
            return `${context.dataset.label}: ${value}%`;
          }
        }
      },
      zoom: {
        zoom: {
          wheel: {
            enabled: true,
          },
          pinch: {
            enabled: true
          },
          mode: 'x',
        },
        pan: {
          enabled: true,
          mode: 'x',
        }
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Date',
          color: '#ffffff'
        },
        ticks: { color: '#ffffff' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Interest Rate (%)',
          color: '#ffffff'
        },
        ticks: { color: '#ffffff' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      }
    }
  };

  return (
    <Paper sx={{ p: 3, backgroundColor: '#2a2a2a', color: '#ffffff' }}>
      <Typography variant="h6" gutterBottom>
        Historical Interest Rates
      </Typography>
      
      {/* Time Range Controls */}
      <Box sx={{ mb: 3, display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
        <Typography variant="body2">Time Range:</Typography>
        <ButtonGroup size="small" variant="outlined">
          <Button 
            onClick={() => handleRangeChange(6)} 
            sx={{ 
              color: selectedRange === '6M' ? '#000000' : '#ffffff', 
              backgroundColor: selectedRange === '6M' ? '#ffffff' : 'transparent',
              borderColor: '#ffffff',
              '&:hover': { backgroundColor: selectedRange === '6M' ? '#ffffff' : 'rgba(255,255,255,0.1)' }
            }}
          >6M</Button>
          <Button 
            onClick={() => handleRangeChange(12)} 
            sx={{ 
              color: selectedRange === '1Y' ? '#000000' : '#ffffff', 
              backgroundColor: selectedRange === '1Y' ? '#ffffff' : 'transparent',
              borderColor: '#ffffff',
              '&:hover': { backgroundColor: selectedRange === '1Y' ? '#ffffff' : 'rgba(255,255,255,0.1)' }
            }}
          >1Y</Button>
          <Button 
            onClick={() => handleRangeChange(24)} 
            sx={{ 
              color: selectedRange === '2Y' ? '#000000' : '#ffffff', 
              backgroundColor: selectedRange === '2Y' ? '#ffffff' : 'transparent',
              borderColor: '#ffffff',
              '&:hover': { backgroundColor: selectedRange === '2Y' ? '#ffffff' : 'rgba(255,255,255,0.1)' }
            }}
          >2Y</Button>
          <Button 
            onClick={() => handleRangeChange(60)} 
            sx={{ 
              color: selectedRange === '5Y' ? '#000000' : '#ffffff', 
              backgroundColor: selectedRange === '5Y' ? '#ffffff' : 'transparent',
              borderColor: '#ffffff',
              '&:hover': { backgroundColor: selectedRange === '5Y' ? '#ffffff' : 'rgba(255,255,255,0.1)' }
            }}
          >5Y</Button>
          <Button 
            onClick={() => handleRangeChange(120)} 
            sx={{ 
              color: selectedRange === '10Y' ? '#000000' : '#ffffff', 
              backgroundColor: selectedRange === '10Y' ? '#ffffff' : 'transparent',
              borderColor: '#ffffff',
              '&:hover': { backgroundColor: selectedRange === '10Y' ? '#ffffff' : 'rgba(255,255,255,0.1)' }
            }}
          >10Y</Button>
          <Button 
            onClick={resetZoom} 
            sx={{ 
              color: selectedRange === 'ALL' ? '#000000' : '#ffffff', 
              backgroundColor: selectedRange === 'ALL' ? '#ffffff' : 'transparent',
              borderColor: '#ffffff',
              '&:hover': { backgroundColor: selectedRange === 'ALL' ? '#ffffff' : 'rgba(255,255,255,0.1)' }
            }}
          >ALL</Button>
        </ButtonGroup>
        
        {/* Reset Zoom Button */}
        <Button 
          onClick={resetZoom}
          size="small"
          variant="outlined"
          sx={{ 
            color: '#ffffff', 
            borderColor: '#ffffff',
            ml: 2
          }}
        >
          RESET ZOOM
        </Button>
      </Box>

      {/* Chart */}
      <Box sx={{ height: 400 }}>
        <Line ref={chartRef} data={chartData} options={options} />
      </Box>


    </Paper>
  );
};

export default InterestRatesChart;
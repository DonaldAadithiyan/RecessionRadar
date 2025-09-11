import React, { useState, useEffect } from 'react';
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
import { Paper, Typography, Box } from '@mui/material';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function YieldCurveChart({ treasuryYields }) {
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    if (treasuryYields && treasuryYields.yields) {
      // Define the order of yield tenors
      const tenorOrder = [
        '1-Month Rate',
        '3-Month Rate',
        '6-Month Rate',
        '1-Year Rate',
        '2-Year Rate',
        '5-Year Rate',
        '10-Year Rate',
        '30-Year Rate'
      ];

      // Map the tenor to display labels (shorter labels for the chart)
      const tenorLabels = {
        '1-Month Rate': '1M',
        '3-Month Rate': '3M',
        '6-Month Rate': '6M',
        '1-Year Rate': '1Y',
        '2-Year Rate': '2Y',
        '5-Year Rate': '5Y',
        '10-Year Rate': '10Y',
        '30-Year Rate': '30Y'
      };

      // Filter and sort yields based on tenorOrder
      const sortedLabels = [];
      const sortedYields = [];

      tenorOrder.forEach(tenor => {
        if (treasuryYields.yields[tenor] !== undefined) {
          sortedLabels.push(tenorLabels[tenor]);
          sortedYields.push(treasuryYields.yields[tenor]);
        }
      });

      setChartData({
        labels: sortedLabels,
        datasets: [
          {
            label: 'Treasury Yield Curve',
            data: sortedYields,
            borderColor: 'rgb(255, 159, 64)',
            backgroundColor: 'rgba(255, 159, 64, 0.5)',
            tension: 0.4,
          }
        ],
      });
    }
  }, [treasuryYields]);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        title: {
          display: true,
          text: 'Yield (%)',
        },
      },
      x: {
        title: {
          display: true,
          text: 'Maturity',
        },
      },
    },
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Treasury Yield Curve',
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`;
          }
        }
      }
    },
  };

  // Check for yield curve inversion (3-month > 10-year)
  const isInverted = treasuryYields && 
                     treasuryYields.yields && 
                     treasuryYields.yields['3-Month Rate'] > 
                     treasuryYields.yields['10-Year Rate'];

  return (
    <Paper sx={{ p: 3, mb: 4, height: 400 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5" component="h2">
          Treasury Yield Curve
        </Typography>
        {isInverted && (
          <Typography 
            variant="subtitle1" 
            component="div" 
            sx={{ 
              color: 'error.main',
              backgroundColor: 'rgba(255, 99, 132, 0.1)',
              px: 2,
              py: 0.5,
              borderRadius: 1
            }}
          >
            Inverted Yield Curve Detected
          </Typography>
        )}
      </Box>
      
      {chartData ? (
        <Box sx={{ height: 'calc(100% - 60px)' }}>
          <Line options={options} data={chartData} />
        </Box>
      ) : (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
          <Typography>Loading yield curve data...</Typography>
        </Box>
      )}
      
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', textAlign: 'right', mt: 1 }}>
        Last updated: {treasuryYields ? new Date(treasuryYields.updated_at).toLocaleString() : 'Loading...'}
      </Typography>
    </Paper>
  );
}

export default YieldCurveChart;

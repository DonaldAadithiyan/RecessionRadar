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

// Note: If you get an error about the annotation plugin, you'll need to install it:
// npm install chartjs-plugin-annotation

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

function RecessionProbabilityChart({ data }) {
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    if (data && data.dates && data.one_month && data.three_month && data.six_month) {
      setChartData({
      labels: data.dates.map(date => new Date(date).toLocaleDateString()),
      datasets: [
        {
        label: '1-Month Probability',
        data: data.one_month.map(val => val),
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
        tension: 0.4,
        },
        {
        label: '3-Month Probability',
        data: data.three_month.map(val => val),
        borderColor: 'rgb(255, 159, 64)',
        backgroundColor: 'rgba(255, 159, 64, 0.5)',
        tension: 0.4,
        },
        {
        label: '6-Month Probability',
        data: data.six_month.map(val => val),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        tension: 0.4,
        },
      ],
      });
    }
  }, [data]);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        title: {
          display: true,
          text: 'Probability (%)',
          color: 'white'
        },
        ticks: {
          color: 'white'
        },
        grid: {
          color: context => {
            if (context.tick.value === 50) {
              return 'rgba(128, 128, 128, 0.5)'; // Gray reference line at 50%
            }
            if (context.tick.value === 70) {
              return 'rgba(255, 99, 132, 0.5)'; // Red reference line at 70%
            }
            return 'rgba(255, 255, 255, 0.1)'; // Lighter grid lines for dark background
          },
          lineWidth: context => {
            if (context.tick.value === 50 || context.tick.value === 70) {
              return 2;
            }
            return 1;
          },
          drawTicks: false
        }
      },
      x: {
        title: {
          display: true,
          text: 'Date',
          color: 'white'
        },
        ticks: {
          color: 'white',
          maxTicksLimit: 15 // Show fewer x-axis labels
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)' // Lighter grid lines for dark background
        }
      },
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: 'white'
        }
      },
      title: {
        display: true,
        text: 'Recession Probability Over Time',
        color: 'white',
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`;
          }
        }
      },
      annotation: {
        annotations: {
          warningLine: {
            type: 'line',
            yMin: 50,
            yMax: 50,
            borderColor: 'rgba(128, 128, 128, 0.7)',
            borderWidth: 2,
            borderDash: [6, 6],
            label: {
              display: true,
              content: 'Warning Threshold (50%)',
              position: 'end'
            }
          },
          dangerLine: {
            type: 'line',
            yMin: 70,
            yMax: 70,
            borderColor: 'rgba(255, 99, 132, 0.7)',
            borderWidth: 2,
            borderDash: [6, 6],
            label: {
              display: true,
              content: 'High Risk (70%)',
              position: 'end'
            }
          }
        }
      }
    },
  };

  // Calculate dynamic minWidth for horizontal scrolling (e.g., 40px per data point, min 1200px)
  const minWidth = chartData && chartData.labels ? Math.max(chartData.labels.length * 40, 1200) : 1200;

  return (
    <Paper sx={{ p: 3, mb: 4, height: 600, bgcolor: '#212121', color: 'white' }}>
      <Typography variant="h5" component="h2" gutterBottom color="white">
        Historical Recession Probability
      </Typography>
      {chartData ? (
        <Box sx={{ height: 'calc(100% - 40px)', overflowX: 'auto' }}>
          <Box sx={{ minWidth: minWidth, height: '100%' }}>
            <Line options={options} data={chartData} height={1} />
          </Box>
        </Box>
      ) : (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
          <Typography color="white">Loading chart data...</Typography>
        </Box>
      )}
    </Paper>
  );
}

export default RecessionProbabilityChart;
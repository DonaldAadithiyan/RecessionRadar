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
import { Paper, Typography, Box, Button, ButtonGroup, Chip } from '@mui/material';

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

function YieldCurveChart({ treasuryYields }) {
  const [chartData, setChartData] = useState(null);
  const chartRef = useRef();

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
          color: 'white'
        },
        ticks: {
          color: 'white'
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Maturity',
          color: 'white'
        },
        ticks: {
          color: 'white'
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      },
    },
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#e9e4e4ff'
        }
      },
      title: {
        display: true,
        text: 'Treasury Yield Curve Chart',
        color: '#ebebebff',
        font: { size: 16 }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        callbacks: {
          title: function(context) {
            return `Maturity: ${context[0].label}`;
          },
          label: function(context) {
            const rate = context.parsed.y.toFixed(2);
            const maturityMap = {
              '3M': '3-Month Rate',
              '6M': '6-Month Rate', 
              '1Y': '1-Year Rate',
              '2Y': '2-Year Rate',
              '5Y': '5-Year Rate',
              '10Y': '10-Year Rate',
              '30Y': '30-Year Rate'
            };
            return `${maturityMap[context.label] || context.label}: ${rate}%`;
          },
          afterBody: function(context) {
            if (context.length > 0) {
              // Calculate spreads for common comparisons
              const data = context[0].chart.data.datasets[0].data;
              const labels = context[0].chart.data.labels;
              
              let spreads = [];
              
              // Find 3M and 10Y indices for inversion check
              const threeMonthIndex = labels.indexOf('3M');
              const tenYearIndex = labels.indexOf('10Y');
              
              if (threeMonthIndex !== -1 && tenYearIndex !== -1) {
                const spread3M10Y = data[tenYearIndex] - data[threeMonthIndex];
                spreads.push(`3M-10Y Spread: ${spread3M10Y.toFixed(2)}%`);
                if (spread3M10Y < 0) {
                  spreads.push('⚠️ Inverted Yield Curve!');
                }
              }
              
              return spreads;
            }
            return [];
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
          mode: 'y',
        },
        pan: {
          enabled: true,
          mode: 'y',
        }
      }
    },
  };

  // Check for yield curve inversion (3-month > 10-year)
  const isInverted = treasuryYields && 
                     treasuryYields.yields && 
                     treasuryYields.yields['3-Month Rate'] > 
                     treasuryYields.yields['10-Year Rate'];

  // Calculate key spreads
  const getSpread = (short, long) => {
    if (treasuryYields && treasuryYields.yields) {
      const shortRate = treasuryYields.yields[short];
      const longRate = treasuryYields.yields[long];
      if (shortRate !== undefined && longRate !== undefined) {
        return (longRate - shortRate).toFixed(2);
      }
    }
    return 'N/A';
  };

  const resetZoom = () => {
    if (chartRef.current) {
      chartRef.current.resetZoom();
    }
  };

  return (
    <Paper sx={{ p: 3, mb: 4, height: 500 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5" component="h2">
          Treasury Yield Curve
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          {isInverted && (
            <Chip 
              label="Inverted Curve" 
              color="error" 
              size="small"
              sx={{ fontWeight: 'bold' }}
            />
          )}
          <Button 
            onClick={resetZoom} 
            size="small" 
            variant="outlined"
          >
            Reset Zoom
          </Button>
        </Box>
      </Box>

      {/* Key Spreads Display */}
      <Box sx={{ mb: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <Chip 
          label={`3M-10Y: ${getSpread('3-Month Rate', '10-Year Rate')}%`}
          color={parseFloat(getSpread('3-Month Rate', '10-Year Rate')) < 0 ? 'error' : 'success'}
          variant="outlined"
          size="small"
        />
        <Chip 
          label={`2Y-10Y: ${getSpread('2-Year Rate', '10-Year Rate')}%`}
          color={parseFloat(getSpread('2-Year Rate', '10-Year Rate')) < 0 ? 'error' : 'default'}
          variant="outlined"
          size="small"
        />
        <Chip 
          label={`5Y-30Y: ${getSpread('5-Year Rate', '30-Year Rate')}%`}
          variant="outlined"
          size="small"
        />
      </Box>
      
      {chartData ? (
        <Box sx={{ height: 'calc(100% - 120px)' }}>
          <Line ref={chartRef} options={options} data={chartData} />
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

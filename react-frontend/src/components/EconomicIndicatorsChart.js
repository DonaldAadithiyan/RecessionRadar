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
import { Paper, Typography, Box, Button, ButtonGroup, FormControl, InputLabel, Select, MenuItem, Chip } from '@mui/material';
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

const EconomicIndicatorsChart = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedIndicators, setSelectedIndicators] = useState(['cpi', 'ppi', 'unemployment_rate']);
  const [selectedRange, setSelectedRange] = useState('ALL');
  const chartRef = useRef();

  const indicatorConfig = {
    cpi: { 
      label: 'Consumer Price Index (CPI)', 
      color: 'rgb(255, 99, 132)', 
      backgroundColor: 'rgba(255, 99, 132, 0.5)',
      yAxisID: 'y'
    },
    ppi: { 
      label: 'Producer Price Index (PPI)', 
      color: 'rgb(54, 162, 235)', 
      backgroundColor: 'rgba(54, 162, 235, 0.5)',
      yAxisID: 'y'
    },
    industrial_production: { 
      label: 'Industrial Production', 
      color: 'rgb(255, 205, 86)', 
      backgroundColor: 'rgba(255, 205, 86, 0.5)',
      yAxisID: 'y1'
    },
    unemployment_rate: { 
      label: 'Unemployment Rate (%)', 
      color: 'rgb(75, 192, 192)', 
      backgroundColor: 'rgba(75, 192, 192, 0.5)',
      yAxisID: 'y2'
    },
    share_price: { 
      label: 'Share Price Index', 
      color: 'rgb(153, 102, 255)', 
      backgroundColor: 'rgba(153, 102, 255, 0.5)',
      yAxisID: 'y3'
    },
    gdp_per_capita: { 
      label: 'GDP Per Capita', 
      color: 'rgb(255, 159, 64)', 
      backgroundColor: 'rgba(255, 159, 64, 0.5)',
      yAxisID: 'y4'
    },
    oecd_cli_index: { 
      label: 'OECD CLI Index', 
      color: 'rgb(199, 199, 199)', 
      backgroundColor: 'rgba(199, 199, 199, 0.5)',
      yAxisID: 'y1'
    },
    csi_index: { 
      label: 'Consumer Sentiment Index', 
      color: 'rgb(83, 102, 255)', 
      backgroundColor: 'rgba(83, 102, 255, 0.5)',
      yAxisID: 'y1'
    },
    ten_year_rate: { 
      label: '10-Year Treasury Rate (%)', 
      color: 'rgb(255, 99, 255)', 
      backgroundColor: 'rgba(255, 99, 255, 0.5)',
      yAxisID: 'y2'
    },
    three_months_rate: { 
      label: '3-Month Treasury Rate (%)', 
      color: 'rgb(99, 255, 132)', 
      backgroundColor: 'rgba(99, 255, 132, 0.5)',
      yAxisID: 'y2'
    },
    six_months_rate: { 
      label: '6-Month Treasury Rate (%)', 
      color: 'rgb(132, 99, 255)', 
      backgroundColor: 'rgba(132, 99, 255, 0.5)',
      yAxisID: 'y2'
    },
    one_year_rate: { 
      label: '1-Year Treasury Rate (%)', 
      color: 'rgb(255, 132, 99)', 
      backgroundColor: 'rgba(255, 132, 99, 0.5)',
      yAxisID: 'y2'
    }
  };

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
      setError('Failed to load economic data');
      console.error('Error fetching economic data:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleIndicatorChange = (event) => {
    setSelectedIndicators(event.target.value);
  };

  const handleRangeChange = (months) => {
    setSelectedRange(months);
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
  };

  if (loading) return <div>Loading economic indicators data...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!data) return <div>No data available</div>;

  // Filter data based on selected time range
  const getFilteredData = () => {
    if (!data || selectedRange === 'ALL') return data;
    
    const totalPoints = data.dates.length;
    const startIndex = Math.max(0, totalPoints - selectedRange);
    
    const filteredData = {
      dates: data.dates.slice(startIndex),
    };
    
    // Filter all indicator data
    Object.keys(indicatorConfig).forEach(indicator => {
      if (data[indicator]) {
        filteredData[indicator] = data[indicator].slice(startIndex);
      }
    });
    
    return filteredData;
  };

  const filteredData = getFilteredData();

  const chartData = {
    labels: filteredData.dates.map(date => new Date(date).toLocaleDateString()),
    datasets: selectedIndicators.map(indicator => ({
      label: indicatorConfig[indicator].label,
      data: filteredData[indicator],
      borderColor: indicatorConfig[indicator].color,
      backgroundColor: indicatorConfig[indicator].backgroundColor,
      tension: 0.1,
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 4,
      yAxisID: indicatorConfig[indicator].yAxisID
    }))
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
        text: 'Historical Economic Indicators',
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
            return `${context.dataset.label}: ${value}`;
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
        type: 'linear',
        display: selectedIndicators.some(ind => indicatorConfig[ind].yAxisID === 'y'),
        position: 'left',
        title: {
          display: true,
          text: 'Price Indices',
          color: '#ffffff'
        },
        ticks: { color: '#ffffff' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      },
      y1: {
        type: 'linear',
        display: selectedIndicators.some(ind => indicatorConfig[ind].yAxisID === 'y1'),
        position: 'right',
        title: {
          display: true,
          text: 'Production Indices',
          color: '#ffffff'
        },
        ticks: { color: '#ffffff' },
        grid: { drawOnChartArea: false }
      },
      y2: {
        type: 'linear',
        display: selectedIndicators.some(ind => indicatorConfig[ind].yAxisID === 'y2'),
        position: 'right',
        title: {
          display: true,
          text: 'Rates (%)',
          color: '#ffffff'
        },
        ticks: { color: '#ffffff' },
        grid: { drawOnChartArea: false }
      },
      y3: {
        type: 'linear',
        display: selectedIndicators.some(ind => indicatorConfig[ind].yAxisID === 'y3'),
        position: 'right',
        title: {
          display: true,
          text: 'Share Prices',
          color: '#ffffff'
        },
        ticks: { color: '#ffffff' },
        grid: { drawOnChartArea: false }
      },
      y4: {
        type: 'linear',
        display: selectedIndicators.some(ind => indicatorConfig[ind].yAxisID === 'y4'),
        position: 'right',
        title: {
          display: true,
          text: 'GDP ($)',
          color: '#ffffff'
        },
        ticks: { color: '#ffffff' },
        grid: { drawOnChartArea: false }
      }
    }
  };

  return (
    <Paper sx={{ p: 3, backgroundColor: '#2a2a2a', color: '#ffffff' }}>
      <Typography variant="h6" gutterBottom>
        Economic Indicators Historical Data
      </Typography>
      
      {/* Indicator Selection */}
      <Box sx={{ mb: 3, width: '100%', overflow: 'visible' }}>
        <FormControl sx={{ 
          width: Math.min(220 + (selectedIndicators.length * 60), window.innerWidth * 0.8), 
          maxWidth: '90vw',
          minWidth: 220,
          mb: 2,
          transition: 'width 0.3s ease-in-out'
        }}>
          <InputLabel 
            sx={{ 
              color: '#ffffff',
              backgroundColor: '#2a2a2a',
              padding: '0 8px',
              '&.Mui-focused': { color: '#ffffff' },
              '&.MuiInputLabel-shrink': { 
                backgroundColor: '#2a2a2a',
                padding: '0 8px'
              }
            }}
          >
            Select Indicators
          </InputLabel>
          <Select
            multiple
            value={selectedIndicators}
            onChange={handleIndicatorChange}
            sx={{ 
              color: '#ffffff', 
              '.MuiOutlinedInput-notchedOutline': { borderColor: '#ffffff' },
              '.MuiSelect-select': { 
                minHeight: '48px',
                padding: '12px 14px',
                overflow: 'visible',
                display: 'flex',
                alignItems: 'center'
              }
            }}
            MenuProps={{
              PaperProps: {
                sx: {
                  maxHeight: 300,
                  overflow: 'auto'
                }
              }
            }}
            renderValue={(selected) => (
              <Box sx={{ 
                display: 'flex', 
                flexWrap: 'nowrap', 
                gap: 0.5, 
                overflow: 'hidden', 
                minHeight: '32px',
                alignItems: 'center',
                width: '100%'
              }}>
                {selected.map((value) => (
                  <Chip 
                    key={value} 
                    label={(() => {
                      let label = indicatorConfig[value].label;
                      // Handle specific cases to avoid redundancy
                      if (label.includes('Consumer Price Index (CPI)')) return 'CPI';
                      if (label.includes('Producer Price Index (PPI)')) return 'PPI';
                      if (label.includes('Unemployment Rate (%)')) return 'Unemp.';
                      if (label.includes('10-Year Treasury Rate (%)')) return '10Y';
                      if (label.includes('3-Month Treasury Rate (%)')) return '3M';
                      if (label.includes('6-Month Treasury Rate (%)')) return '6M';
                      // Generic replacements for other cases
                      return label
                        .replace('Industrial Production', 'Ind. Prod.')
                        .replace('Share Price Index', 'Shares')
                        .replace('GDP Per Capita', 'GDP/Cap')
                        .replace('Consumer Sentiment Index', 'CSI')
                        .replace('OECD CLI Index', 'OECD CLI');
                    })()}
                    size="small"
                    sx={{ 
                      backgroundColor: indicatorConfig[value].color, 
                      color: '#ffffff',
                      fontSize: '0.8rem',
                      height: '28px',
                      whiteSpace: 'nowrap',
                      flexShrink: 1,
                      minWidth: 'auto',
                      '& .MuiChip-label': {
                        paddingLeft: '8px',
                        paddingRight: '8px',
                        fontWeight: 500
                      }
                    }}
                  />
                ))}
              </Box>
            )}
          >
            {Object.entries(indicatorConfig).map(([key, config]) => (
              <MenuItem key={key} value={key}>
                {config.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      {/* Time Range Controls */}
      <Box sx={{ mb: 3, display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
        <Typography variant="body2">Time Range:</Typography>
        <ButtonGroup size="small" variant="outlined">
          <Button 
            onClick={() => handleRangeChange(6)} 
            sx={{ 
              color: selectedRange === 6 ? '#000000' : '#ffffff', 
              borderColor: '#ffffff',
              backgroundColor: selectedRange === 6 ? '#ffffff' : 'transparent'
            }}
          >
            6M
          </Button>
          <Button 
            onClick={() => handleRangeChange(12)} 
            sx={{ 
              color: selectedRange === 12 ? '#000000' : '#ffffff', 
              borderColor: '#ffffff',
              backgroundColor: selectedRange === 12 ? '#ffffff' : 'transparent'
            }}
          >
            1Y
          </Button>
          <Button 
            onClick={() => handleRangeChange(24)} 
            sx={{ 
              color: selectedRange === 24 ? '#000000' : '#ffffff', 
              borderColor: '#ffffff',
              backgroundColor: selectedRange === 24 ? '#ffffff' : 'transparent'
            }}
          >
            2Y
          </Button>
          <Button 
            onClick={() => handleRangeChange(60)} 
            sx={{ 
              color: selectedRange === 60 ? '#000000' : '#ffffff', 
              borderColor: '#ffffff',
              backgroundColor: selectedRange === 60 ? '#ffffff' : 'transparent'
            }}
          >
            5Y
          </Button>
          <Button 
            onClick={() => handleRangeChange(120)} 
            sx={{ 
              color: selectedRange === 120 ? '#000000' : '#ffffff', 
              borderColor: '#ffffff',
              backgroundColor: selectedRange === 120 ? '#ffffff' : 'transparent'
            }}
          >
            10Y
          </Button>
          <Button 
            onClick={() => handleRangeChange('ALL')} 
            sx={{ 
              color: selectedRange === 'ALL' ? '#000000' : '#ffffff', 
              borderColor: '#ffffff',
              backgroundColor: selectedRange === 'ALL' ? '#ffffff' : 'transparent'
            }}
          >
            ALL
          </Button>
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
          Reset Zoom
        </Button>
      </Box>

      {/* Chart */}
      <Box sx={{ height: 500 }}>
        <Line ref={chartRef} data={chartData} options={options} />
      </Box>


    </Paper>
  );
};

export default EconomicIndicatorsChart;
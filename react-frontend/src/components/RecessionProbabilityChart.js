import React, { useState, useEffect, useRef, useCallback } from 'react';
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
  TimeScale,
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import zoomPlugin from 'chartjs-plugin-zoom';
import { Paper, Typography, Box, Button, ButtonGroup, TextField, FormControlLabel, Checkbox, FormGroup } from '@mui/material';

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
  Legend,
  TimeScale,
  zoomPlugin
);

function RecessionProbabilityChart({ data }) {
  const [chartData, setChartData] = useState(null);
  const [filteredData, setFilteredData] = useState(null);
  const [selectedRange, setSelectedRange] = useState('ALL'); // Default to all available data
  const [customDateRange, setCustomDateRange] = useState({
    startDate: null,
    endDate: null
  });
  const [showCustomRange, setShowCustomRange] = useState(false);
  const [visibilityState, setVisibilityState] = useState({
    oneMonth: true,
    threeMonth: true,
    sixMonth: true
  });
  const chartRef = useRef();

  const filterDataByMonths = useCallback((months) => {
    if (!data || !data.dates) return data;
    
    if (months === 'ALL') return data;
    
    const totalPoints = data.dates.length;
    const startIndex = Math.max(0, totalPoints - months);
    
    return {
      dates: data.dates.slice(startIndex),
      one_month: data.one_month.slice(startIndex),
      three_month: data.three_month.slice(startIndex),
      six_month: data.six_month.slice(startIndex),
      base: data.base ? data.base.slice(startIndex) : []
    };
  }, [data]);

  const filterDataByDateRange = (startDate, endDate) => {
    if (!data || !data.dates || !startDate || !endDate) return data;
    
    const start = new Date(startDate);
    const end = new Date(endDate);
    
    const filteredIndices = data.dates.reduce((acc, date, index) => {
      const dateVal = new Date(date);
      if (dateVal >= start && dateVal <= end) {
        acc.push(index);
      }
      return acc;
    }, []);
    
    return {
      dates: filteredIndices.map(i => data.dates[i]),
      one_month: filteredIndices.map(i => data.one_month[i]),
      three_month: filteredIndices.map(i => data.three_month[i]),
      six_month: filteredIndices.map(i => data.six_month[i]),
      base: data.base ? filteredIndices.map(i => data.base[i]) : []
    };
  };

  const handleRangeChange = (months) => {
    setSelectedRange(months);
    setShowCustomRange(false);
    setCustomDateRange({ startDate: null, endDate: null });
    
    const newFilteredData = filterDataByMonths(months);
    setFilteredData(newFilteredData);
    
    // Reset zoom when changing time range for better UX
    setTimeout(() => {
      if (chartRef.current) {
        chartRef.current.resetZoom();
      }
    }, 100);
  };

  const handleCustomRangeToggle = () => {
    setShowCustomRange(!showCustomRange);
    if (!showCustomRange) {
      setSelectedRange('CUSTOM');
    }
  };

  const handleCustomDateRangeChange = (startDate, endDate) => {
    setCustomDateRange({ startDate, endDate });
    if (startDate && endDate) {
      // Validate date range
      const start = new Date(startDate);
      const end = new Date(endDate);
      
      if (start <= end) {
        const newFilteredData = filterDataByDateRange(startDate, endDate);
        setFilteredData(newFilteredData);
        
        // Reset zoom for better UX
        setTimeout(() => {
          if (chartRef.current) {
            chartRef.current.resetZoom();
          }
        }, 100);
      }
    }
  };

  const handleVisibilityToggle = (datasetKey) => {
    setVisibilityState(prev => ({
      ...prev,
      [datasetKey]: !prev[datasetKey]
    }));
  };

  // Get date constraints for the date inputs
  const getDateConstraints = () => {
    if (!data || !data.dates || data.dates.length === 0) {
      return { min: '', max: '' };
    }
    
    const minDate = new Date(data.dates[0]).toISOString().split('T')[0];
    const maxDate = new Date(data.dates[data.dates.length - 1]).toISOString().split('T')[0];
    
    return { min: minDate, max: maxDate };
  };

  const resetZoom = () => {
    if (chartRef.current) {
      chartRef.current.resetZoom();
    }
  };

  useEffect(() => {
    if (data && data.dates) {
      // Default to all available data
      const defaultFilteredData = filterDataByMonths('ALL');
      setFilteredData(defaultFilteredData);
    }
  }, [data, filterDataByMonths]);

  useEffect(() => {
    const dataToUse = filteredData || data;
    if (dataToUse && dataToUse.dates && dataToUse.one_month && dataToUse.three_month && dataToUse.six_month) {
      // Process data to handle null/undefined values and ensure numbers
      const processedData = {
        dates: dataToUse.dates,
        one_month: dataToUse.one_month.map(val => val === null || val === undefined || val === '' ? null : Number(val)),
        three_month: dataToUse.three_month.map(val => val === null || val === undefined || val === '' ? null : Number(val)),
        six_month: dataToUse.six_month.map(val => val === null || val === undefined || val === '' ? null : Number(val))
      };
      
      // Create datasets array based on visibility state
      const datasets = [];
      
      if (visibilityState.oneMonth) {
        datasets.push({
          label: '1-Month Probability',
          data: processedData.one_month,
          borderColor: '#0096FF',
          backgroundColor: 'rgba(0, 150, 255, 0.2)',
          tension: 0.1,
          borderWidth: 4,
          pointRadius: 0,
          pointHoverRadius: 4,
          fill: false,
          yAxisID: 'y',
        });
      }
      
      if (visibilityState.threeMonth) {
        datasets.push({
          label: '3-Month Probability',
          data: processedData.three_month,
          borderColor: 'rgb(255, 159, 64)',
          backgroundColor: 'rgba(255, 159, 64, 0.5)',
          tension: 0.4,
          borderWidth: 2,
          pointRadius: 1,
        });
      }
      
      if (visibilityState.sixMonth) {
        datasets.push({
          label: '6-Month Probability',
          data: processedData.six_month,
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
          tension: 0.4,
          borderWidth: 2,
          pointRadius: 1,
        });
      }

      setChartData({
        labels: processedData.dates.map(date => new Date(date)),
        datasets: datasets,
      });
    }
  }, [data, filteredData, visibilityState]);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    spanGaps: true, // Connect points even when there are null values
    layout: {
      padding: {
        top: 10,
        bottom: 10
      }
    },
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
        type: 'time',
        time: {
          parser: 'YYYY-MM-DD',
          displayFormats: {
            day: 'MMM dd',
            week: 'MMM dd',
            month: 'MMM yyyy',
            quarter: 'MMM yyyy',
            year: 'yyyy'
          },
          tooltipFormat: 'MMM dd, yyyy',
          unit: (() => {
            if (selectedRange === 'ALL') return 'year';
            const numRange = typeof selectedRange === 'number' ? selectedRange : 
              (selectedRange === '6M' ? 6 : selectedRange === '1Y' ? 12 : 
               selectedRange === '2Y' ? 24 : selectedRange === '5Y' ? 60 : 120);
            return numRange <= 6 ? 'month' : numRange <= 24 ? 'month' : 'year';
          })()
        },
        title: {
          display: true,
          text: 'Date',
          color: 'white',
          font: {
            size: 14,
            weight: 'bold'
          }
        },
        ticks: {
          color: 'white',
          maxTicksLimit: selectedRange === 'ALL' ? 15 : 12,
          maxRotation: 45,
          minRotation: 0,
          font: {
            size: 11
          },
          autoSkip: true,
          autoSkipPadding: selectedRange === 'ALL' ? 30 : 20,
          source: 'auto',
          major: {
            enabled: true
          }
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
          lineWidth: 1
        }
      },
    },
    plugins: {
      legend: {
        position: 'top',
        align: 'start',
        maxHeight: 50,
        labels: {
          color: 'white',
          usePointStyle: false,
          padding: 20,
          font: {
            size: 12
          },
          boxWidth: 40,
          boxHeight: 3
        }
      },
      title: {
        display: true,
        text: 'Recession Probability Over Time',
        color: '#ffffff',
        align: 'start',
        font: {
          size: 16,
          weight: 'bold'
        },
        position: 'top',
        padding: {
          top: 10,
          bottom: 15
        }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.9)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: 'rgba(255, 255, 255, 0.3)',
        borderWidth: 1,
        cornerRadius: 8,
        callbacks: {
          title: function(context) {
            const date = new Date(context[0].parsed.x);
            return `Date: ${date.toLocaleDateString('en-US', { 
              year: 'numeric', 
              month: 'short', 
              day: 'numeric' 
            })}`;
          },
          label: function(context) {
            const probability = context.parsed.y.toFixed(1);
            let riskLevel = 'Low Risk';
            if (probability >= 70) riskLevel = 'High Risk';
            else if (probability >= 50) riskLevel = 'Moderate Risk';
            else if (probability >= 30) riskLevel = 'Elevated Risk';
            
            return [
              `${context.dataset.label}: ${probability}%`,
              `Risk Level: ${riskLevel}`
            ];
          },
          afterBody: function(context) {
            return 'Red line indicates high-risk threshold (70%)';
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


  const getTimeRangeDisplay = () => {
    // If Custom Range toggle is active (or selectedRange is CUSTOM), show Custom Range in the title
    if (showCustomRange || selectedRange === 'CUSTOM') {
      if (customDateRange.startDate && customDateRange.endDate) {
        const startDate = new Date(customDateRange.startDate);
        const endDate = new Date(customDateRange.endDate);
        return ` (${startDate.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })} - ${endDate.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })})`;
      }
      return ' (Custom Range)';
    }
    if (selectedRange !== 'ALL') {
      return ` (Last ${selectedRange === 6 ? '6 months' : selectedRange === 12 ? '1 year' : 
        selectedRange === 24 ? '2 years' : selectedRange === 60 ? '5 years' : '10 years'})`;
    }
    return ' (All Available Data)';
  };

  return (
    <Paper sx={{ p: 3, mb: 4, width: '100%', bgcolor: '#212121', color: 'white', display: 'flex', flexDirection: 'column', minHeight: showCustomRange ? 640 : 480, boxSizing: 'border-box' }}>
      <Typography variant="h5" component="h2" gutterBottom color="white" sx={{ mb: 2, textAlign: 'center' }}>
        Historical Recession Probability{getTimeRangeDisplay()}
      </Typography>
      
      {/* Time Range Controls */}
      <Box sx={{ mb: 2, display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
        <Typography variant="body2">Time Range:</Typography>
        <ButtonGroup size="small" variant="outlined">
          <Button 
            onClick={() => handleRangeChange(6)} 
            sx={{ 
              color: selectedRange === 6 ? '#000' : '#ffffff', 
              borderColor: '#ffffff',
              backgroundColor: selectedRange === 6 ? '#ffffff' : 'transparent'
            }}
          >6M</Button>
          <Button 
            onClick={() => handleRangeChange(12)} 
            sx={{ 
              color: selectedRange === 12 ? '#000' : '#ffffff', 
              borderColor: '#ffffff',
              backgroundColor: selectedRange === 12 ? '#ffffff' : 'transparent'
            }}
          >1Y</Button>
          <Button 
            onClick={() => handleRangeChange(24)} 
            sx={{ 
              color: selectedRange === 24 ? '#000' : '#ffffff', 
              borderColor: '#ffffff',
              backgroundColor: selectedRange === 24 ? '#ffffff' : 'transparent'
            }}
          >2Y</Button>
          <Button 
            onClick={() => handleRangeChange(60)} 
            sx={{ 
              color: selectedRange === 60 ? '#000' : '#ffffff', 
              borderColor: '#ffffff',
              backgroundColor: selectedRange === 60 ? '#ffffff' : 'transparent'
            }}
          >5Y</Button>
          <Button 
            onClick={() => handleRangeChange(120)} 
            sx={{ 
              color: selectedRange === 120 ? '#000' : '#ffffff', 
              borderColor: '#ffffff',
              backgroundColor: selectedRange === 120 ? '#ffffff' : 'transparent'
            }}
          >10Y</Button>
          <Button 
            onClick={() => handleRangeChange('ALL')} 
            sx={{ 
              color: selectedRange === 'ALL' ? '#000' : '#ffffff', 
              borderColor: '#ffffff',
              backgroundColor: selectedRange === 'ALL' ? '#ffffff' : 'transparent'
            }}
          >ALL</Button>
        </ButtonGroup>
        
        {/* Custom Range Toggle */}
        <Button 
          onClick={handleCustomRangeToggle}
          size="small"
          variant="outlined"
          sx={{ 
            color: selectedRange === 'CUSTOM' ? '#000' : '#ffffff', 
            borderColor: '#ffffff',
            backgroundColor: selectedRange === 'CUSTOM' ? '#ffffff' : 'transparent',
            ml: 2
          }}
        >
          Custom Range
        </Button>
        
        {/* Reset Zoom Button */}
        <Button 
          onClick={resetZoom}
          size="small"
          variant="outlined"
          sx={{ 
            color: '#ffffff', 
            borderColor: '#ffffff',
            ml: 1
          }}
        >
          Reset Zoom
        </Button>
      </Box>

      {/* Data Visibility Controls */}
      <Box sx={{ mb: 2, p: 1.5, bgcolor: 'rgba(255,255,255,0.05)', borderRadius: 1 }}>
        <Typography variant="body2" sx={{ mb: 1.5, color: 'white' }}>
          Display Options:
        </Typography>
        <FormGroup row>
          <FormControlLabel
            control={
              <Checkbox
                checked={visibilityState.oneMonth}
                onChange={() => handleVisibilityToggle('oneMonth')}
                sx={{
                  color: '#0096FF',
                  '&.Mui-checked': {
                    color: '#0096FF',
                  }
                }}
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ 
                  width: 20, 
                  height: 3, 
                  backgroundColor: '#0096FF', 
                  borderRadius: 1 
                }} />
                <Typography variant="body2" color="white">1-Month Probability</Typography>
              </Box>
            }
          />
          <FormControlLabel
            control={
              <Checkbox
                checked={visibilityState.threeMonth}
                onChange={() => handleVisibilityToggle('threeMonth')}
                sx={{
                  color: 'rgb(255, 159, 64)',
                  '&.Mui-checked': {
                    color: 'rgb(255, 159, 64)',
                  }
                }}
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ 
                  width: 20, 
                  height: 3, 
                  backgroundColor: 'rgb(255, 159, 64)', 
                  borderRadius: 1 
                }} />
                <Typography variant="body2" color="white">3-Month Probability</Typography>
              </Box>
            }
          />
          <FormControlLabel
            control={
              <Checkbox
                checked={visibilityState.sixMonth}
                onChange={() => handleVisibilityToggle('sixMonth')}
                sx={{
                  color: 'rgb(255, 99, 132)',
                  '&.Mui-checked': {
                    color: 'rgb(255, 99, 132)',
                  }
                }}
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ 
                  width: 20, 
                  height: 3, 
                  backgroundColor: 'rgb(255, 99, 132)', 
                  borderRadius: 1 
                }} />
                <Typography variant="body2" color="white">6-Month Probability</Typography>
              </Box>
            }
          />
        </FormGroup>
      </Box>

      {/* Custom Date Range Picker */}
      {showCustomRange && (
        <Box sx={{ mb: 2, p: 1.5, bgcolor: 'rgba(255,255,255,0.05)', borderRadius: 1 }}>
          <Typography variant="body2" sx={{ mb: 1.5, color: 'white' }}>
            Select Custom Date Range:
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
            <TextField
              label="Start Date"
              type="date"
              value={customDateRange.startDate || ''}
              onChange={(e) => {
                const updatedRange = { ...customDateRange, startDate: e.target.value };
                setCustomDateRange(updatedRange);
                if (updatedRange.startDate && updatedRange.endDate) {
                  handleCustomDateRangeChange(updatedRange.startDate, updatedRange.endDate);
                }
              }}
              inputProps={{
                min: getDateConstraints().min,
                max: getDateConstraints().max
              }}
              InputLabelProps={{
                shrink: true,
                style: { color: 'rgba(255,255,255,0.7)' }
              }}
              InputProps={{
                style: { color: 'white' }
              }}
              sx={{
                '& .MuiOutlinedInput-root': {
                  '& fieldset': {
                    borderColor: 'rgba(255,255,255,0.3)',
                  },
                  '&:hover fieldset': {
                    borderColor: 'rgba(255,255,255,0.5)',
                  },
                  '&.Mui-focused fieldset': {
                    borderColor: 'rgba(255,255,255,0.7)',
                  },
                },
                '& input[type="date"]::-webkit-calendar-picker-indicator': {
                  filter: 'invert(1)',
                  cursor: 'pointer'
                }
              }}
              size="small"
            />
            <TextField
              label="End Date"
              type="date"
              value={customDateRange.endDate || ''}
              onChange={(e) => {
                const updatedRange = { ...customDateRange, endDate: e.target.value };
                setCustomDateRange(updatedRange);
                if (updatedRange.startDate && updatedRange.endDate) {
                  handleCustomDateRangeChange(updatedRange.startDate, updatedRange.endDate);
                }
              }}
              inputProps={{
                min: customDateRange.startDate || getDateConstraints().min,
                max: getDateConstraints().max
              }}
              InputLabelProps={{
                shrink: true,
                style: { color: 'rgba(255,255,255,0.7)' }
              }}
              InputProps={{
                style: { color: 'white' }
              }}
              sx={{
                '& .MuiOutlinedInput-root': {
                  '& fieldset': {
                    borderColor: 'rgba(255,255,255,0.3)',
                  },
                  '&:hover fieldset': {
                    borderColor: 'rgba(255,255,255,0.5)',
                  },
                  '&.Mui-focused fieldset': {
                    borderColor: 'rgba(255,255,255,0.7)',
                  },
                },
                '& input[type="date"]::-webkit-calendar-picker-indicator': {
                  filter: 'invert(1)',
                  cursor: 'pointer'
                }
              }}
              size="small"
            />
          </Box>
        </Box>
      )}

      {chartData ? (
        <Box sx={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: showCustomRange ? 480 : 320 }}>
          <Box sx={{ flex: 1, minHeight: showCustomRange ? 420 : 260 }}>
            <Line ref={chartRef} options={options} data={chartData} />
          </Box>
        </Box>
      ) : (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flex: 1 }}>
          <Typography color="white">Loading chart data...</Typography>
        </Box>
      )}
    </Paper>
  );
}

export default RecessionProbabilityChart;
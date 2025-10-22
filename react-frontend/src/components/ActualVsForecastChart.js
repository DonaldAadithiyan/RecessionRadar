import React, { useEffect, useState, useMemo } from 'react';
import { Box, Paper, Typography, Select, MenuItem, FormControl, InputLabel, CircularProgress } from '@mui/material';
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

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

function ActualVsForecastChart() {
  const [dataRows, setDataRows] = useState(null);
  const [indicators, setIndicators] = useState([]);
  const [selected, setSelected] = useState('gdp_per_capita');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const url = '/data/all_indicators_test_preds.json'; 
    setLoading(true);
    setError(null);
    fetch(url)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(json => {
        if (!Array.isArray(json) || json.length === 0) {
          throw new Error('Empty or invalid JSON');
        }

        setDataRows(json);

        // find indicators by searching keys that end with _actual
        const keys = Object.keys(json[0]);
        const found = keys
          .filter(k => k.endsWith('_actual'))
          .map(k => k.replace(/_actual$/, ''))
          .sort();

        setIndicators(found);
  // pick gdp_per_capita if present, else first available indicator
  setSelected(found.includes('gdp_per_capita') ? 'gdp_per_capita' : (found[0] || ''));
      })
      .catch(err => {
        console.error('Failed to fetch test predictions JSON:', err);
        setError(err.message);
      })
      .finally(() => setLoading(false));
  }, []);

  

  const categories = useMemo(() => {
    if (!dataRows) return [];
    return dataRows.map(r => r.date);
  }, [dataRows]);

  const chartData = useMemo(() => {
    if (!dataRows || !selected) return null;

    const actualKey = `${selected}_actual`;
    const forecastedKey = `${selected}_hybrid`;

    const actual = [];
    const forecasted = [];

    for (const row of dataRows) {
      const a = row[actualKey];
      // prefer hybrid, fall back to arima or prophet if hybrid missing
      let h = row[forecastedKey];
      if (h === undefined) {
        if (row[`${selected}_arima`] !== undefined) h = row[`${selected}_arima`];
        else if (row[`${selected}_prophet`] !== undefined) h = row[`${selected}_prophet`];
        else h = null;
      }

      actual.push(typeof a === 'number' ? a : (a ? Number(a) : null));
      forecasted.push(typeof h === 'number' ? h : (h ? Number(h) : null));
    }

    return {
      labels: categories,
      datasets: [
        {
          label: `${selected} Actual`,
          data: actual,
          borderColor: '#1976d2',
          backgroundColor: 'rgba(25,118,210,0.1)',
          tension: 0.2,
        },
        {
          label: `${selected} Forecasted`,
          data: forecasted,
          borderColor: '#2e7d32',
          backgroundColor: 'rgba(46,125,50,0.08)',
          tension: 0.2,
        }
      ]
    };
  }, [dataRows, categories, selected]);

  const options = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: selected ? `${selected} — Actual vs Forecast` : 'Actual vs Forecast' }
    },
    scales: {
      x: {
        type: 'category',
        ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12 }
      }
    }
  }), [selected]);

  return (
    <Paper sx={{ p: 2, mb: 4, height: 480 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Actual vs Forecast</Typography>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <FormControl size="small" sx={{ minWidth: 220 }}>
            <InputLabel id="avf-select-label">Indicator</InputLabel>
            <Select
              labelId="avf-select-label"
              value={selected}
              label="Indicator"
              onChange={e => setSelected(e.target.value)}
            >
              {indicators.map(ind => (
                <MenuItem key={ind} value={ind}>{ind}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
      </Box>

      {loading && (
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '84%' }}>
          <CircularProgress />
        </Box>
      )}

      {error && (
        <Typography color="error">Failed to load prediction data: {error}</Typography>
      )}

      {!loading && !error && chartData && (
        <Box sx={{ height: '400px' }}>
          <Line data={chartData} options={options} />
        </Box>
      )}
    </Paper>
  );
}

export default ActualVsForecastChart;

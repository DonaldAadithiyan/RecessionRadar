// File: react-frontend/src/components/ActualVsForecastChart.js
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
  const [dataRowsProphet, setDataRowsProphet] = useState(null);
  const [indicators, setIndicators] = useState([]);
  const [selected, setSelected] = useState('gdp_per_capita');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Use PUBLIC_URL so fetch works when app is served from root or a subpath
    const base = (process.env.PUBLIC_URL && process.env.PUBLIC_URL !== '') ? process.env.PUBLIC_URL : '';
    const urlBase = `${base}/data/all_indicators_test_preds_hybridarima.json`;
    const urlProphet = `${base}/data/all_indicators_test_preds_hybridprophet.json`;

    setLoading(true);
    setError(null);

    // fetch base and prophet in parallel; prophet file is optional
    const pBase = fetch(urlBase).then(res => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    });

    const pProphet = fetch(urlProphet).then(res => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    }).catch(err => {
      // prophet is optional locally; continue if missing
      console.debug('Prophet combined JSON not available at', urlProphet, err.message);
      return null;
    });

    Promise.all([pBase, pProphet])
      .then(([baseJson, prophetJson]) => {
        if (!Array.isArray(baseJson) || baseJson.length === 0) {
          throw new Error('Empty or invalid base JSON');
        }

        setDataRows(baseJson);
        setDataRowsProphet(Array.isArray(prophetJson) ? prophetJson : null);

        // detect indicators via keys ending with _actual
        const keys = Object.keys(baseJson[0]);
        const found = keys
          .filter(k => k.endsWith('_actual'))
          .map(k => k.replace(/_actual$/, ''))
          .sort();

        setIndicators(found);
        setSelected(found.includes('gdp_per_capita') ? 'gdp_per_capita' : (found[0] || ''));
      })
      .catch(err => {
        console.error('Failed to fetch prediction JSONs:', err);
        setError(err.message || String(err));
      })
      .finally(() => setLoading(false));
  }, []);

  const categories = useMemo(() => {
    if (!dataRows) return [];
    return dataRows.map(r => r.date);
  }, [dataRows]);

  const chartData = useMemo(() => {
    if (!dataRows || !selected) return null;

    // ---- Indicator -> preferred hybrid model mapping (matches the image provided)
    // Keys must match indicator names discovered from JSON (case-sensitive, without suffix)
    // Values: 'arima' to use ARIMA-hybrid, 'prophet' to use Prophet-hybrid
    const indicatorModelMap = {
      // From image: 1-year rate -> Prophet-Hybrid
      '1_year_rate': 'prophet',

      // 3-month rate (JSON key is '3_months_rate') -> ARIMA-Hybrid
      '3_months_rate': 'arima',

      // 6-month rate -> Prophet-Hybrid
      '6_months_rate': 'prophet',

      // CPI -> Prophet-Hybrid
      'CPI': 'prophet',

      // INDPRO -> ARIMA-Hybrid
      'INDPRO': 'arima',

      // 10-year rate -> Prophet-Hybrid
      '10_year_rate': 'prophet',

      // Share Price -> Prophet-Hybrid (JSON key: 'share_price')
      'share_price': 'prophet',

      // Unemployment Rate -> ARIMA-Hybrid
      'unemployment_rate': 'arima',

      // PPI -> ARIMA-Hybrid
      'PPI': 'arima',

      // OECD CLI Index -> Prophet-Hybrid (JSON key: 'OECD_CLI_index')
      'OECD_CLI_index': 'prophet',

      // CSI Index -> Prophet-Hybrid (JSON key: 'CSI_index')
      'CSI_index': 'prophet',

      // GDP per Capita -> Prophet-Hybrid (JSON key: 'gdp_per_capita')
      'gdp_per_capita': 'prophet'
    };

    const preferredModel = indicatorModelMap[selected] || 'arima';

    // build date->row maps for robust joining
    const arimaMap = {};
    for (const r of dataRows) if (r && r.date) arimaMap[r.date] = r;

    const prophetMap = {};
    if (dataRowsProphet) {
      for (const r of dataRowsProphet) if (r && r.date) prophetMap[r.date] = r;
    }

    const actual = [];
    const forecasted = [];

    for (const date of categories) {
      const baseRow = arimaMap[date];
      const propRow = prophetMap[date];

      const a = baseRow ? baseRow[`${selected}_actual`] : null;

      let h = null;

      // If mapped to prophet, try prophet hybrid first
      if (preferredModel === 'prophet' && propRow) {
        h = propRow[`${selected}_hybrid`];
        if (h === undefined) {
          console.debug(`Prophet hybrid missing for ${selected} on ${date}, falling back`);
          h = propRow[`${selected}_prophet`];
        }
      }

      // Fallback to base hybrid (ARIMA-hybrid), then explicit arima/prophet columns
      if (h === undefined || h === null) {
        if (baseRow) h = baseRow[`${selected}_hybrid`];
        if (h === undefined || h === null) {
          if (baseRow && baseRow[`${selected}_arima`] !== undefined) h = baseRow[`${selected}_arima`];
          else if (propRow && propRow[`${selected}_prophet`] !== undefined) h = propRow[`${selected}_prophet`];
          else h = null;
        }
      }

      actual.push(typeof a === 'number' ? a : (a ? Number(a) : null));
      forecasted.push(typeof h === 'number' ? h : (h ? Number(h) : null));
    }

    const modelLabel = preferredModel === 'prophet' ? 'Prophet-Hybrid' : 'ARIMA-Hybrid';

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
          label: `${selected} Forecasted (${modelLabel})`,
          data: forecasted,
          borderColor: '#2e7d32',
          backgroundColor: 'rgba(46,125,50,0.08)',
          tension: 0.2,
        }
      ]
    };
  }, [dataRows, categories, selected, dataRowsProphet]);

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
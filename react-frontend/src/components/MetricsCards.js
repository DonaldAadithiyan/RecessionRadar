import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import TrendingFlatIcon from '@mui/icons-material/TrendingFlat';

function ProbabilityCard({ title, value, timeframe }) {
  // Determine color and icon based on probability value
  let color, icon;
  
  if (value < 0.3) {
    color = 'success.main';
    icon = <TrendingDownIcon fontSize="large" sx={{ color }} />;
  } else if (value < 0.6) {
    color = 'warning.main';
    icon = <TrendingFlatIcon fontSize="large" sx={{ color }} />;
  } else {
    color = 'error.main';
    icon = <TrendingUpIcon fontSize="large" sx={{ color }} />;
  }

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" component="div" gutterBottom>
            {title}
          </Typography>
          {icon}
        </Box>
        
        <Typography variant="h3" component="div" sx={{ fontWeight: 'bold', color }}>
          {Math.round(value * 100)}%
        </Typography>
        
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          Probability within {timeframe}
        </Typography>
      </CardContent>
    </Card>
  );
}

function MetricsCards({ predictions }) {
  return (
    <Box sx={{ mb: 4 }}>
      <Typography variant="h5" component="h2" gutterBottom>
        Recession Probability Forecast
      </Typography>
      
      <Box sx={{ 
        display: 'grid', 
        gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr 1fr' }, 
        gap: 2,
        mt: 2
      }}>
        {/* <ProbabilityCard 
          title="Recession Forecast" 
          value={predictions.forcast} 
          timeframe="next month" 
        /> */}
        <ProbabilityCard 
          title="1-Month Forecast" 
          value={predictions.one_month} 
          timeframe="next month" 
        />
        <ProbabilityCard 
          title="3-Month Forecast" 
          value={predictions.three_month} 
          timeframe="next three months" 
        />
        <ProbabilityCard 
          title="6-Month Forecast" 
          value={predictions.six_month} 
          timeframe="next six months" 
        />
      </Box>
      
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', textAlign: 'right', mt: 1 }}>
        Last updated: {new Date(predictions.updated_at).toLocaleString()}
      </Typography>
    </Box>
  );
}

export default MetricsCards;

import React from 'react';
import { Paper, Typography, Box } from '@mui/material';
import WarningIcon from '@mui/icons-material/Warning';

function Disclaimer() {
  return (
    <Paper 
      elevation={2}
      sx={{ 
        p: 2, 
        mt: 4, 
        backgroundColor: 'rgba(255, 152, 0, 0.1)',
        border: '1px solid rgba(255, 152, 0, 0.3)',
        borderRadius: 2
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <WarningIcon color="warning" sx={{ mr: 1 }} />
        <Typography variant="h6" component="div" color="warning.main">
          Disclaimer
        </Typography>
      </Box>
      <Typography variant="body2" paragraph>
        This tool provides recession probability estimates based on economic indicators and is designed for informational purposes only.
      </Typography>
      <Typography variant="body2" paragraph>
        The predictions should not be interpreted as financial advice or used as the sole basis for investment decisions. Economic forecasting is inherently uncertain, and actual outcomes may differ significantly from predictions.
      </Typography>
      <Typography variant="body2">
        Users should consult with qualified financial professionals before making any financial decisions based on this information.
      </Typography>
    </Paper>
  );
}

export default Disclaimer;

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
       These forecasts and recession probabilities are for informational purposes only and do not constitute buy, hold, or sell advice. Users should consult a qualified financial advisor before making investment decisions or adjusting their portfolios. 
      </Typography>
      
    </Paper>
  );
}

export default Disclaimer;

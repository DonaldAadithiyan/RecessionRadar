import React from 'react';
import { Paper, Typography, Box, Chip } from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';

function YieldCurveAnalysis({ treasuryYields }) {
  // Return null if no data provided
  if (!treasuryYields || !treasuryYields.yields) {
    return null;
  }

  // Check for inversions
  const shortRates = ['1-Month Rate', '3-Month Rate', '6-Month Rate'];
  const longRates = ['10-Year Rate', '30-Year Rate'];
  
  let hasInversion = false;
  let inversionText = "";
  
  for (const shortRate of shortRates) {
    for (const longRate of longRates) {
      if (treasuryYields.yields[shortRate] && treasuryYields.yields[longRate] && 
          treasuryYields.yields[shortRate] > treasuryYields.yields[longRate]) {
        hasInversion = true;
        inversionText = `${shortRate.replace(' Rate', '')} > ${longRate.replace(' Rate', '')}`;
        break;
      }
    }
    if (hasInversion) break;
  }

  // Calculate steepness
  let steepnessText = "";
  let steepnessColor = "text.primary";
  
  const maturityOrder = ['1-Month Rate', '3-Month Rate', '6-Month Rate', '1-Year Rate', 
                       '2-Year Rate', '5-Year Rate', '10-Year Rate', '30-Year Rate'];
  
  const availableRates = maturityOrder.filter(rate => treasuryYields.yields[rate] !== undefined);
  
  if (availableRates.length >= 2) {
    const shortest = availableRates[0];
    const longest = availableRates[availableRates.length - 1];
    
    const steepness = treasuryYields.yields[longest] - treasuryYields.yields[shortest];
    
    if (steepness > 1.5) {
      steepnessText = "The yield curve is quite steep, which often indicates strong economic growth expectations.";
      steepnessColor = "success.main";
    } else if (steepness < -0.5) {
      steepnessText = "The yield curve is deeply inverted, which historically has been a stronger recession signal.";
      steepnessColor = "error.main";
    } else if (steepness > 0) {
      steepnessText = "The yield curve has a normal positive slope.";
    } else {
      steepnessText = "The yield curve has a flat to slightly negative slope.";
      steepnessColor = "warning.main";
    }
  }

  return (
    <Paper sx={{ p: 3, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <Typography variant="h6" component="h3" gutterBottom>
          Treasury Yield Curve Analysis
        </Typography>
        
        {hasInversion ? (
          <Chip 
            icon={<WarningAmberIcon />} 
            label="Inverted Yield Curve" 
            color="error" 
            variant="outlined"
          />
        ) : (
          <Chip 
            icon={<InfoOutlinedIcon />} 
            label="Normal Yield Curve" 
            color="success" 
            variant="outlined"
          />
        )}
      </Box>
      
      {/* Inversion Analysis */}
      {hasInversion ? (
        <Typography variant="body1" paragraph sx={{ color: 'warning.main' }}>
          The yield curve is currently inverted ({inversionText}). This has historically preceded recessions by 6-18 months.
        </Typography>
      ) : (
        <Typography variant="body1" paragraph>
          The yield curve is currently normal (long-term rates {'>'} short-term rates), which typically indicates economic expansion.
        </Typography>
      )}
      
      {/* Steepness Analysis */}
      {steepnessText && (
        <Typography variant="body1" paragraph sx={{ color: steepnessColor }}>
          {steepnessText}
        </Typography>
      )}
      
      {/* Educational information */}
      <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
        The yield curve shows the relationship between treasury bond maturities and their yields.
        An inverted yield curve (when short-term rates are higher than long-term rates) is often considered a recession indicator.
      </Typography>
      
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', textAlign: 'right', mt: 2 }}>
        Last updated: {new Date(treasuryYields.updated_at).toLocaleString()}
      </Typography>
    </Paper>
  );
}

export default YieldCurveAnalysis;

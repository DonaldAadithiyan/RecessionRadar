import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  TextField,
  Button,
  Paper,
  CircularProgress,
  Alert,
  InputAdornment
} from '@mui/material';

// Import components
import MetricsCards from '../components/MetricsCards';

// Import services
import { getTreasuryYields, getEconomicIndicators, getCustomPrediction } from '../services/api';

function CustomPrediction() {
  // State for loading status
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  
  // State for data
  const [treasuryYields, setTreasuryYields] = useState(null);
  const [economicIndicators, setEconomicIndicators] = useState(null);
  const [customPrediction, setCustomPrediction] = useState(null);
  const [error, setError] = useState(null);
  
  // State for form inputs
  const [formValues, setFormValues] = useState({});
  
  // Load initial data on component mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Fetch data in parallel
        const [yields, indicators] = await Promise.all([
          getTreasuryYields(),
          getEconomicIndicators()
        ]);
        
        setTreasuryYields(yields);
        setEconomicIndicators(indicators);
        
        // Use only live API data - no hardcoded defaults
        const treasuryRates = yields.yields || {};
        
        // Initialize form values with current live data only
        const initialValues = {
          ...treasuryRates,
          ...indicators.indicators
        };
        
        setFormValues(initialValues);
      } catch (error) {
        console.error('Error fetching data:', error);
        setError('Error loading initial data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);
  
  // Handle form input changes
  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setFormValues({
      ...formValues,
      [name]: parseFloat(value)
    });
  };
  
  // Handle form submission
  const handleSubmit = async (event) => {
    event.preventDefault();
    
    try {
      setPredicting(true);
      setError(null);
      
      const prediction = await getCustomPrediction(formValues);
      setCustomPrediction(prediction);
    } catch (error) {
      console.error('Error getting custom prediction:', error);
      setError('Error calculating prediction. Please check your inputs and try again.');
    } finally {
      setPredicting(false);
    }
  };
  
  // Reset form to initial values
  const handleReset = () => {
    if (treasuryYields && economicIndicators) {
      const resetValues = {
        ...Object.entries(treasuryYields.yields).reduce((acc, [key, value]) => {
          acc[key] = value;
          return acc;
        }, {}),
        ...Object.entries(economicIndicators.indicators).reduce((acc, [key, value]) => {
          acc[key] = value;
          return acc;
        }, {})
      };
      
      setFormValues(resetValues);
      setCustomPrediction(null);
      setError(null);
    }
  };
  
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
        <CircularProgress />
      </Box>
    );
  }
  
  return (
    <Container maxWidth="xl">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Custom Recession Prediction
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" paragraph>
          Adjust economic indicators below to generate a custom recession probability forecast
        </Typography>

        {/* Results Section */}
        {customPrediction && (
          <Box sx={{ mb: 4 }}>
            <MetricsCards predictions={customPrediction} />
            
            {/* Interpretation Section */}
            <Paper sx={{ p: 3, mt: 3 }}>
              <Typography variant="h6" component="h3" gutterBottom>
                Interpretation
              </Typography>
              
              {/* Overall Risk Assessment */}
              {(() => {
                const prob1m = customPrediction?.probabilities?.['1-Month'] || 0;
                const prob3m = customPrediction?.probabilities?.['3-Month'] || 0;
                const prob6m = customPrediction?.probabilities?.['6-Month'] || 0;
                const avgProb = (prob1m + prob3m + prob6m) / 3;
                
                let riskLevel = "Low";
                let riskColor = "success.main";
                
                if (avgProb >= 0.7) {
                  riskLevel = "High";
                  riskColor = "error.main";
                } else if (avgProb >= 0.3) {
                  riskLevel = "Moderate";
                  riskColor = "warning.main";
                }
                
                return (
                  <Typography variant="body1" paragraph>
                    Based on the provided economic indicators, the overall recession risk is <Box component="span" sx={{ color: riskColor, fontWeight: 'bold' }}>{riskLevel}</Box>.
                  </Typography>
                );
              })()}
              
              {/* Key Factors */}
              <Typography variant="body1" paragraph>
                Key factors influencing this prediction:
              </Typography>
              
              <Box component="ul" sx={{ pl: 2 }}>
                {/* Yield curve inversion check */}
                {formValues['3-Month Rate'] > formValues['10-Year Rate'] && (
                  <Typography component="li" variant="body2">
                    Inverted yield curve (3-Month &gt; 10-Year rate)
                  </Typography>
                )}
                
                {formValues['Unemployment Rate'] > 5.0 && (
                  <Typography component="li" variant="body2">
                    Elevated unemployment rate
                  </Typography>
                )}
                
                {formValues['CPI'] > 4.0 && (
                  <Typography component="li" variant="body2">
                    High inflation (CPI)
                  </Typography>
                )}
                
                {formValues['PPI'] > 4.0 && (
                  <Typography component="li" variant="body2">
                    High producer inflation (PPI)
                  </Typography>
                )}
                
                {formValues['CSI Index'] < 70.0 && (
                  <Typography component="li" variant="body2">
                    Low consumer sentiment
                  </Typography>
                )}
                
                {formValues['OECD CLI Index'] < 99.0 && (
                  <Typography component="li" variant="body2">
                    Below-trend OECD CLI (leading indicator)
                  </Typography>
                )}
                
                {formValues['Industrial Production'] < 100.0 && (
                  <Typography component="li" variant="body2">
                    Below-average industrial production
                  </Typography>
                )}
                
                {formValues['Share Price'] < 4000.0 && (
                  <Typography component="li" variant="body2">
                    Depressed share prices
                  </Typography>
                )}
              </Box>
              
              {/* Yield Curve Analysis */}
              <Typography variant="h6" component="h3" sx={{ mt: 3 }} gutterBottom>
                Treasury Yield Curve Analysis
              </Typography>
              
              {(() => {
                // Check for inversions
                const shortRates = ['1-Month Rate', '3-Month Rate', '6-Month Rate'];
                const longRates = ['10-Year Rate', '30-Year Rate'];
                
                let hasInversion = false;
                let inversionText = "";
                
                for (const shortRate of shortRates) {
                  for (const longRate of longRates) {
                    if (formValues[shortRate] && formValues[longRate] && formValues[shortRate] > formValues[longRate]) {
                      hasInversion = true;
                      inversionText = `${shortRate.replace(' Rate', '')} > ${longRate.replace(' Rate', '')}`;
                      break;
                    }
                  }
                  if (hasInversion) break;
                }
                
                if (hasInversion) {
                  return (
                    <Typography variant="body1" paragraph sx={{ color: 'warning.main' }}>
                      Your inputs show an inverted yield curve ({inversionText}). This has historically preceded recessions by 6-18 months.
                    </Typography>
                  );
                } else {
                  return (
                    <Typography variant="body1" paragraph>
                      Your inputs show a normal yield curve, which typically indicates economic expansion.
                    </Typography>
                  );
                }
              })()}
              
              {/* Steepness Analysis */}
              {(() => {
                const maturityOrder = ['1-Month Rate', '3-Month Rate', '6-Month Rate', '1-Year Rate', 
                                      '2-Year Rate', '5-Year Rate', '10-Year Rate', '30-Year Rate'];
                
                const availableRates = maturityOrder.filter(rate => formValues[rate] !== undefined);
                
                if (availableRates.length >= 2) {
                  const shortest = availableRates[0];
                  const longest = availableRates[availableRates.length - 1];
                  
                  const steepness = formValues[longest] - formValues[shortest];
                  
                  if (steepness > 1.5) {
                    return (
                      <Typography variant="body1" paragraph>
                        Your custom yield curve is quite steep, which often indicates strong economic growth expectations.
                      </Typography>
                    );
                  } else if (steepness < -0.5) {
                    return (
                      <Typography variant="body1" paragraph sx={{ color: 'warning.main' }}>
                        Your custom yield curve is deeply inverted, which historically has been a stronger recession signal.
                      </Typography>
                    );
                  }
                }
                
                return null;
              })()}
            </Paper>
          </Box>
        )}
        
        {/* Error Alert */}
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}
        
        {/* Form Section */}
        <Paper sx={{ p: 3, bgcolor: '#1E1E1E', color: 'white' }}>
          <form onSubmit={handleSubmit}>
            <Typography variant="h5" component="h2" gutterBottom sx={{ color: 'white' }}>
              Economic Indicators
            </Typography>
            
            <Grid container spacing={4}>
              {/* Left Column - Treasury Yields */}
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom sx={{ color: 'white', borderBottom: '1px solid rgba(255, 255, 255, 0.2)', pb: 1 }}>
                  Treasury Rates
                </Typography>
                
                {/* Define specific required rates matching the Streamlit version */}
                {['1-Year Rate', '3-Month Rate', '6-Month Rate', '10-Year Rate'].map((rateName) => {
                  // Get the value from treasuryYields if available
                  const defaultValue = treasuryYields?.yields?.[rateName] || 
                    // Default values from the Python code if not available
                    (rateName === '1-Year Rate' ? 4.86 : 
                     rateName === '3-Month Rate' ? 4.52 : 
                     rateName === '6-Month Rate' ? 4.65 : 
                     rateName === '10-Year Rate' ? 3.81 : 0);
                  
                  return (
                    <Box key={rateName} sx={{ mb: 3 }}>
                      <Typography gutterBottom sx={{ color: 'white' }}>
                        {rateName} (%)
                      </Typography>
                      <TextField
                        fullWidth
                        name={rateName}
                        value={formValues[rateName] !== undefined ? formValues[rateName] : defaultValue}
                        onChange={handleInputChange}
                        type="number"
                        InputProps={{
                          endAdornment: <InputAdornment position="end" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>%</InputAdornment>,
                          sx: { color: 'white' }
                        }}
                        inputProps={{
                          step: "any"
                        }}
                        size="small"
                        variant="outlined"
                        sx={{
                          '& .MuiOutlinedInput-root': {
                            '& fieldset': {
                              borderColor: 'rgba(255, 255, 255, 0.3)',
                            },
                            '&:hover fieldset': {
                              borderColor: 'rgba(255, 255, 255, 0.5)',
                            },
                            '&.Mui-focused fieldset': {
                              borderColor: '#90caf9',
                            },
                            '& input': { 
                              color: 'white' 
                            }
                          },
                        }}
                      />
                    </Box>
                  );
                })}
              </Grid>
              
              {/* Middle Column - Core Economic Indicators */}
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom sx={{ color: 'white', borderBottom: '1px solid rgba(255, 255, 255, 0.2)', pb: 1 }}>
                  Economic Indicators
                </Typography>
                
                {[
                  {
                    key: 'CPI',
                    label: 'Consumer Price Index (% change)',
                    adornment: '%'
                  },
                  {
                    key: 'PPI',
                    label: 'Producer Price Index (% change)',
                    adornment: '%'
                  },
                  {
                    key: 'Unemployment Rate',
                    label: 'Unemployment Rate (%)',
                    adornment: '%'
                  },
                  {
                    key: 'Industrial Production',
                    label: 'Industrial Production Index',
                    adornment: ''
                  }
                ].map(indicator => {
                  const { key, label, adornment } = indicator;
                  // Get default value from economicIndicators
                  const defaultValue = economicIndicators?.indicators?.[key] || 0;
                  
                  return (
                    <Box key={key} sx={{ mb: 3 }}>
                      <Typography gutterBottom sx={{ color: 'white' }}>
                        {label}
                      </Typography>
                      <TextField
                        fullWidth
                        name={key}
                        value={formValues[key] !== undefined ? formValues[key] : defaultValue}
                        onChange={handleInputChange}
                        type="number"
                        InputProps={{
                          endAdornment: adornment ? <InputAdornment position="end" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>{adornment}</InputAdornment> : null,
                          sx: { color: 'white' }
                        }}
                        inputProps={{
                          step: "any"
                        }}
                        size="small"
                        variant="outlined"
                        sx={{
                          '& .MuiOutlinedInput-root': {
                            '& fieldset': {
                              borderColor: 'rgba(255, 255, 255, 0.3)',
                            },
                            '&:hover fieldset': {
                              borderColor: 'rgba(255, 255, 255, 0.5)',
                            },
                            '&.Mui-focused fieldset': {
                              borderColor: '#90caf9',
                            },
                            '& input': { 
                              color: 'white' 
                            }
                          },
                        }}
                      />
                    </Box>
                  );
                })}
              </Grid>
              
              {/* Right Column - Market & Sentiment */}
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom sx={{ color: 'white', borderBottom: '1px solid rgba(255, 255, 255, 0.2)', pb: 1 }}>
                  Market & Sentiment Indicators
                </Typography>
                
                {/* Share Price */}
                <Box sx={{ mb: 3 }}>
                  <Typography gutterBottom sx={{ color: 'white' }}>
                    Share Price Index
                  </Typography>
                  <TextField
                    fullWidth
                    name="Share Price"
                    value={formValues["Share Price"] !== undefined ? formValues["Share Price"] : (economicIndicators?.indicators?.["Share Price"] || 0)}
                    onChange={handleInputChange}
                    type="number"
                    InputProps={{
                      sx: { color: 'white' }
                    }}
                    inputProps={{
                      step: "any"
                    }}
                    size="small"
                    variant="outlined"
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        '& fieldset': {
                          borderColor: 'rgba(255, 255, 255, 0.3)',
                        },
                        '&:hover fieldset': {
                          borderColor: 'rgba(255, 255, 255, 0.5)',
                        },
                        '&.Mui-focused fieldset': {
                          borderColor: '#90caf9',
                        },
                        '& input': { 
                          color: 'white' 
                        }
                      },
                    }}
                  />
                </Box>
                
                {/* OECD CLI Index */}
                <Box sx={{ mb: 3 }}>
                  <Typography gutterBottom sx={{ color: 'white' }}>
                    OECD CLI Index
                  </Typography>
                  <TextField
                    fullWidth
                    name="OECD CLI Index"
                    value={formValues["OECD CLI Index"] !== undefined ? formValues["OECD CLI Index"] : (economicIndicators?.indicators?.["OECD CLI Index"] || 0)}
                    onChange={handleInputChange}
                    type="number"
                    InputProps={{
                      sx: { color: 'white' }
                    }}
                    inputProps={{
                      step: "any"
                    }}
                    size="small"
                    variant="outlined"
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        '& fieldset': {
                          borderColor: 'rgba(255, 255, 255, 0.3)',
                        },
                        '&:hover fieldset': {
                          borderColor: 'rgba(255, 255, 255, 0.5)',
                        },
                        '&.Mui-focused fieldset': {
                          borderColor: '#90caf9',
                        },
                        '& input': { 
                          color: 'white' 
                        }
                      },
                    }}
                  />
                </Box>
                
                {/* CSI Index */}
                <Box sx={{ mb: 3 }}>
                  <Typography gutterBottom sx={{ color: 'white' }}>
                    Consumer Sentiment Index
                  </Typography>
                  <TextField
                    fullWidth
                    name="CSI Index"
                    value={formValues["CSI Index"] !== undefined ? formValues["CSI Index"] : (economicIndicators?.indicators?.["CSI Index"] || 0)}
                    onChange={handleInputChange}
                    type="number"
                    InputProps={{
                      sx: { color: 'white' }
                    }}
                    inputProps={{
                      step: "any"
                    }}
                    size="small"
                    variant="outlined"
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        '& fieldset': {
                          borderColor: 'rgba(255, 255, 255, 0.3)',
                        },
                        '&:hover fieldset': {
                          borderColor: 'rgba(255, 255, 255, 0.5)',
                        },
                        '&.Mui-focused fieldset': {
                          borderColor: '#90caf9',
                        },
                        '& input': { 
                          color: 'white' 
                        }
                      },
                    }}
                  />
                </Box>
              </Grid>
            </Grid>
            

            
            {/* Form Buttons */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
              <Button 
                variant="outlined" 
                sx={{ 
                  color: '#64B5F6', 
                  borderColor: '#64B5F6',
                  '&:hover': {
                    borderColor: '#90CAF9',
                    backgroundColor: 'rgba(100, 181, 246, 0.08)'
                  }
                }}
                onClick={handleReset}
                disabled={predicting}
              >
                RESET TO CURRENT VALUES
              </Button>
              <Button 
                type="submit" 
                variant="contained" 
                sx={{ 
                  backgroundColor: '#64B5F6',
                  '&:hover': {
                    backgroundColor: '#42A5F5'
                  }
                }}
                disabled={predicting}
                startIcon={predicting ? <CircularProgress size={20} color="inherit" /> : null}
              >
                {predicting ? 'CALCULATING...' : 'CALCULATE PREDICTION'}
              </Button>
            </Box>
          </form>
        </Paper>
      </Box>
    </Container>
  );
}

export default CustomPrediction;

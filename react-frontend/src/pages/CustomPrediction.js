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
        // Fetch data in parallel
        const [yields, indicators] = await Promise.all([
          getTreasuryYields(),
          getEconomicIndicators()
        ]);

        setTreasuryYields(yields);
        setEconomicIndicators(indicators);

        // Initialize form values with current live data only
        const initialValues = {
          ...(yields.yields || {}),
          ...(indicators.indicators || {})
        };

        setFormValues(initialValues);
      } catch (error) {
        console.error('Error fetching data:', error);
        setError('Error loading initial data. Please try again later.');
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
        ...(treasuryYields.yields || {}),
        ...(economicIndicators.indicators || {})
      };

      setFormValues(resetValues);
    }
  };

  // AI Financial Advice panel is rendered below the results and only when a custom prediction exists.

  return (
    <Container maxWidth="xl">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Custom Recession Prediction
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" paragraph>
          Adjust economic indicators below to generate a custom recession probability forecast
        </Typography>

        {error && (
          <Box sx={{ mb: 2 }}>
            <Alert severity="error">{error}</Alert>
          </Box>
        )}

    
        
        {/* Metrics cards (recession probabilities) - shown after a prediction */}
        {customPrediction && (
          <MetricsCards
            predictions={{
              base_pred: typeof customPrediction?.base_pred === 'number' ? customPrediction.base_pred : (customPrediction?.base_pred ?? 0),
              one_month: typeof customPrediction?.one_month === 'number' ? customPrediction.one_month : (customPrediction?.one_month ?? 0),
              three_month: typeof customPrediction?.three_month === 'number' ? customPrediction.three_month : (customPrediction?.three_month ?? 0),
              six_month: typeof customPrediction?.six_month === 'number' ? customPrediction.six_month : (customPrediction?.six_month ?? 0),
              updated_at: customPrediction?.updated_at || customPrediction?.generated_at || new Date().toISOString()
            }}
          />
        )}

        
        
        {/* Form Section */}
        <Paper sx={{ p: 3, bgcolor: '#1E1E1E', color: 'white' }}>
          <form onSubmit={handleSubmit}>
            <Typography variant="h5" component="h2" gutterBottom sx={{ color: 'white', mb: 5 }}>
              Economic Indicators
            </Typography>
            
            <Grid container spacing={4}>
              {/* Left Column - Treasury Yields */}
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom sx={{ color: 'white', borderBottom: '1px solid rgba(255, 255, 255, 0.2)', pb: 1, mb: 5 }}>
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
                <Typography variant="h6" gutterBottom sx={{ color: 'white', borderBottom: '1px solid rgba(255, 255, 255, 0.2)', pb: 1, mb: 5 }}>
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
                <Typography variant="h6" gutterBottom sx={{ color: 'white', borderBottom: '1px solid rgba(255, 255, 255, 0.2)', pb: 1, mb: 5 }}>
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

                {/* GDP per Capita */}
                <Box sx={{ mb: 3 }}>
                  <Typography gutterBottom sx={{ color: 'white' }}>
                    GDP per Capita
                  </Typography>
                  <TextField
                    fullWidth
                    name="gdp_per_capita"
                    value={formValues["gdp_per_capita"] !== undefined ? formValues["gdp_per_capita"] : (economicIndicators?.indicators?.["gdp_per_capita"] || 0)}
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
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4, mb: 1 }}>
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

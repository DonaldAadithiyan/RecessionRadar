import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Grid, 
  Paper, 
  Box, 
  CircularProgress,
  Alert,
  Chip,
  Divider
} from '@mui/material';

// Import components
import MetricsCards from '../components/MetricsCards';
import RecessionProbabilityChart from '../components/RecessionProbabilityChart';
import YieldCurveChart from '../components/YieldCurveChart';
import YieldCurveAnalysis from '../components/YieldCurveAnalysis';

// Import services
import { 
  getTreasuryYields, 
  getRecessionProbabilities, 
  getCurrentPrediction,
  getEconomicIndicators
} from '../services/api';

function Dashboard() {
  // State for loading status
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // State for data
  const [treasuryYields, setTreasuryYields] = useState(null);
  const [recessionProbabilities, setRecessionProbabilities] = useState(null);
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [economicIndicators, setEconomicIndicators] = useState(null);
  
  // Load data on component mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // HARDCODED VALUES FOR DISPLAY TESTING
        // These would normally come from the API endpoints
        
        // Sample treasury yields data - create an inverted yield curve for demonstration
        const yields = {
          yields: {
            '1-Month Rate': 4.35,
            '3-Month Rate': 4.52,
            '6-Month Rate': 4.65,
            '1-Year Rate': 4.86,
            '2-Year Rate': 4.65,
            '5-Year Rate': 4.25,
            '10-Year Rate': 3.81,
            '30-Year Rate': 3.95
          },
          updated_at: new Date().toISOString()
        };
        
        // Sample recession probabilities data
        const probabilities = {
          dates: [
            '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', 
            '2023-05-01', '2023-06-01', '2023-07-01', '2023-08-01',
            '2023-09-01', '2023-10-01', '2023-11-01', '2023-12-01'
          ],
          one_month: [0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.42, 0.48, 0.52, 0.58, 0.62, 0.65],
          three_month: [0.15, 0.22, 0.28, 0.35, 0.42, 0.48, 0.55, 0.62, 0.68, 0.72, 0.75, 0.78],
          six_month: [0.25, 0.32, 0.38, 0.45, 0.52, 0.58, 0.65, 0.72, 0.76, 0.80, 0.83, 0.85]
        };
        
        // Sample current prediction data
        const prediction = {
          probabilities: {
            '1-Month': 0.65,
            '3-Month': 0.78,
            '6-Month': 0.85
          },
          previous_probabilities: {
            '1-Month': 0.62,
            '3-Month': 0.75,
            '6-Month': 0.83
          },
          updated_at: new Date().toISOString()
        };
        
        // Sample economic indicators data
        const indicators = {
          indicators: {
            'CPI': 3.7,
            'PPI': 4.2,
            'Unemployment Rate': 4.8,
            'Industrial Production': 97.5,
            'Share Price': 4285.3,
            'OECD CLI Index': 98.7,
            'CSI Index': 68.5,
            'GDP Growth': 2.1,
            'Retail Sales': 0.8,
            'Housing Starts': -3.2,
            'Manufacturing PMI': 48.2,
            'Services PMI': 51.8
          },
          updated_at: new Date().toISOString()
        };
        
        // Try using real API data but fall back to hardcoded if it fails
        try {
          const realData = await Promise.all([
            getTreasuryYields(),
            getRecessionProbabilities(),
            getCurrentPrediction(),
            getEconomicIndicators()
          ]);
          
          setTreasuryYields(realData[0]);
          setRecessionProbabilities(realData[1]);
          setCurrentPrediction(realData[2]);
          setEconomicIndicators(realData[3]);
          
        } catch (apiError) {
          console.log('Using hardcoded data instead of API data:', apiError);
          // Fall back to hardcoded data
          setTreasuryYields(yields);
          setRecessionProbabilities(probabilities);
          setCurrentPrediction(prediction);
          setEconomicIndicators(indicators);
        }
        
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        setError('Failed to load dashboard data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);



  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
        <CircularProgress />
      </Box>
    );
  }
  
  if (error) {
    return (
      <Container maxWidth="xl">
        <Box sx={{ my: 4 }}>
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        </Box>
      </Container>
    );
  }
  
  return (
    <Container maxWidth="xl">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          RecessionRadar Dashboard
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" paragraph>
          Monitoring economic indicators and recession probabilities
        </Typography>

        {/* Note about hardcoded data */}
        <Alert severity="info" sx={{ mb: 3 }}>
          Currently showing hardcoded sample data. This will be replaced with real API data when connected to the backend.
        </Alert>

        {/* Error handling for missing data */}
        {!currentPrediction && (
          <Alert severity="warning" sx={{ mb: 3 }}>
            Current prediction data is not available. Some dashboard features may be limited.
          </Alert>
        )}

        {/* Metrics Cards */}
        {currentPrediction && (
          <MetricsCards predictions={currentPrediction} />
        )}
        
        {/* Charts Section */}
        <Grid container spacing={3}>
          {/* Recession Probability Chart */}
          <Grid item xs={12}>
            {recessionProbabilities ? (
              <Box>
                <RecessionProbabilityChart data={recessionProbabilities} />
                <Paper sx={{ p: 2, mb: 4, bgcolor: '#212121' }}>
                  <Typography variant="body2" color="white">
                    The chart shows the recession probability trends over time. A probability above 50% indicates 
                    a higher likelihood of a recession occurring within the specified time frame.
                  </Typography>
                </Paper>
              </Box>
            ) : (
              <Paper sx={{ p: 3, mb: 4, textAlign: 'center' }}>
                <Typography variant="h6" color="text.secondary">
                  Recession probability data is not available.
                </Typography>
              </Paper>
            )}
          </Grid>
          
          {/* Yield Curve Chart and Analysis */}
          <Grid item xs={12}>
            {treasuryYields ? (
              <Box>
                <YieldCurveChart treasuryYields={treasuryYields} />
                <Divider sx={{ my: 2 }} />
                <YieldCurveAnalysis treasuryYields={treasuryYields} />
              </Box>
            ) : (
              <Paper sx={{ p: 3, mb: 4, textAlign: 'center' }}>
                <Typography variant="h6" color="text.secondary">
                  Treasury yield data is not available.
                </Typography>
              </Paper>
            )}
          </Grid>
          
          {/* Economic Indicators */}
          <Grid item xs={12}>
            <Paper sx={{ p: 3, mb: 4 }}>
              <Typography variant="h5" component="h2" gutterBottom>
                Key Economic Indicators
              </Typography>
              
              {economicIndicators ? (
                <Grid container spacing={2}>
                  {Object.entries(economicIndicators.indicators).filter(([name]) => !name.includes('Rate')).map(([name, value]) => (
                    <Grid item xs={12} sm={6} md={4} lg={3} key={name}>
                      <Paper 
                        sx={{ 
                          p: 2, 
                          textAlign: 'center',
                          backgroundColor: 'background.paper',
                          boxShadow: 1,
                          position: 'relative',
                          overflow: 'hidden'
                        }}
                      >
                        <Typography variant="body2" color="text.secondary">
                          {name}
                        </Typography>
                        <Typography variant="h6" component="div" sx={{ mt: 1 }}>
                          {value.toFixed(2)}
                          {renderIndicatorChip(name, value)}
                        </Typography>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              ) : (
                <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center' }}>
                  Economic indicator data is not available.
                </Typography>
              )}
              
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', textAlign: 'right', mt: 2 }}>
                Last updated: {economicIndicators ? new Date(economicIndicators.updated_at).toLocaleString() : 'Unknown'}
              </Typography>
            </Paper>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
  
  // Helper function to add status chips to indicators
  function renderIndicatorChip(name, value) {
    // Define thresholds for indicators
    if (name === 'CPI' || name === 'PPI') {
      if (value > 4.0) {
        return (
          <Chip 
            label="High" 
            size="small" 
            color="error" 
            sx={{ position: 'absolute', top: 8, right: 8, fontSize: '0.7rem' }}
          />
        );
      } else if (value > 2.5) {
        return (
          <Chip 
            label="Elevated" 
            size="small" 
            color="warning" 
            sx={{ position: 'absolute', top: 8, right: 8, fontSize: '0.7rem' }}
          />
        );
      }
    } else if (name === 'Unemployment Rate') {
      if (value > 6.0) {
        return (
          <Chip 
            label="High" 
            size="small" 
            color="error" 
            sx={{ position: 'absolute', top: 8, right: 8, fontSize: '0.7rem' }}
          />
        );
      } else if (value > 4.5) {
        return (
          <Chip 
            label="Elevated" 
            size="small" 
            color="warning" 
            sx={{ position: 'absolute', top: 8, right: 8, fontSize: '0.7rem' }}
          />
        );
      }
    } else if (name === 'CSI Index') {
      if (value < 70.0) {
        return (
          <Chip 
            label="Low" 
            size="small" 
            color="error" 
            sx={{ position: 'absolute', top: 8, right: 8, fontSize: '0.7rem' }}
          />
        );
      }
    } else if (name === 'OECD CLI Index') {
      if (value < 99.0) {
        return (
          <Chip 
            label="Below Trend" 
            size="small" 
            color="warning" 
            sx={{ position: 'absolute', top: 8, right: 8, fontSize: '0.7rem', maxWidth: '90px' }}
          />
        );
      }
    } else if (name === 'Industrial Production') {
      if (value < 100.0) {
        return (
          <Chip 
            label="Below Avg" 
            size="small" 
            color="warning" 
            sx={{ position: 'absolute', top: 8, right: 8, fontSize: '0.7rem' }}
          />
        );
      }
    }
    return null;
  }
}

export default Dashboard;

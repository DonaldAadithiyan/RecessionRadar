import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Grid, 
  Paper, 
  Box, 
  CircularProgress,
  Alert,
  Divider
} from '@mui/material';

// Import components
import MetricsCards from '../components/MetricsCards';
import RecessionProbabilityChart from '../components/RecessionProbabilityChart';
import YieldCurveChart from '../components/YieldCurveChart';
import YieldCurveAnalysis from '../components/YieldCurveAnalysis';
import EconomicIndicatorsChart from '../components/EconomicIndicatorsChart';
import InterestRatesChart from '../components/InterestRatesChart';

// Import services
import { 
  getTreasuryYields, 
  getRecessionProbabilities, 
  getCurrentPrediction,
  getEconomicIndicators
} from '../services/api';

// Import context
import { DashboardProvider } from '../contexts/DashboardContext';

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
        
        // Fetch real data from API endpoints
        console.log('Fetching live data from APIs...');
        
        // Fetch data with individual error handling
        const treasuryData = await getTreasuryYields().catch(err => {
          console.warn('Treasury yields API failed:', err);
          return null;
        });
        
        const probabilitiesData = await getRecessionProbabilities().catch(err => {
          console.warn('Recession probabilities API failed:', err);
          return null;
        });
        
        const predictionData = await getCurrentPrediction().catch(err => {
          console.warn('Current prediction API failed:', err);
          return null;
        });
        
        const indicatorsData = await getEconomicIndicators().catch(err => {
          console.warn('Economic indicators API failed:', err);
          return null;
        });
        
        console.log('Live data fetched successfully:');
        console.log('- Treasury yields:', treasuryData);
        console.log('- Recession probabilities:', probabilitiesData);
        console.log('- Current prediction:', predictionData);
        console.log('- Economic indicators:', indicatorsData);
        
        // Set data only if successfully fetched
        if (treasuryData) setTreasuryYields(treasuryData);
        if (probabilitiesData) setRecessionProbabilities(probabilitiesData);
        if (predictionData) setCurrentPrediction(predictionData);
        if (indicatorsData) setEconomicIndicators(indicatorsData);
        
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        setError('Failed to load dashboard data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
    
    // Set up auto-refresh every 5 minutes for live data
    const refreshInterval = setInterval(fetchData, 5 * 60 * 1000);
    
    // Cleanup interval on unmount
    return () => clearInterval(refreshInterval);
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
          
          {/* Economic Indicators Historical Chart */}
          <Grid item xs={12}>
            <EconomicIndicatorsChart />
          </Grid>
          
          {/* Interest Rates Historical Chart */}
          <Grid item xs={12}>
            <InterestRatesChart />
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
}

export default Dashboard;

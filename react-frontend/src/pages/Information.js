import React from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Link,
  Divider
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import InfoIcon from '@mui/icons-material/Info';
import WarningIcon from '@mui/icons-material/Warning';

function Information() {
  return (
    <Container maxWidth="xl">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Information 
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" paragraph>
          Understanding the recession prediction model and economic indicators
        </Typography>

        {/* Overview Section */}
        <Paper sx={{ p: 3, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            About RecessionRadar
          </Typography>
          <Typography paragraph>
            RecessionRadar is a dashboard for visualizing and forecasting the probability of a recession occurring within different time horizons based on economic indicators.
          </Typography>
          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
            How It Works
          </Typography>
          <Typography paragraph>
            RecessionRadar uses machine learning models trained on historical economic data to predict the probability of a recession occurring within 1, 3, and 6 months.
          </Typography>
          <Typography paragraph>
            The models take into account various economic indicators, including:
          </Typography>
          <ul>
            <Typography component="li">Treasury Yields (1-Year, 3-Month, 6-Month, 10-Year)</Typography>
            <Typography component="li">Consumer Price Index (CPI)</Typography>
            <Typography component="li">Producer Price Index (PPI)</Typography>
            <Typography component="li">Industrial Production</Typography>
            <Typography component="li">Share Prices</Typography>
            <Typography component="li">Unemployment Rate</Typography>
            <Typography component="li">OECD Composite Leading Indicator</Typography>
            <Typography component="li">Consumer Sentiment Index</Typography>
          </ul>
          <Box sx={{ display: 'flex', alignItems: 'center', mt: 2, p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
            <InfoIcon sx={{ mr: 2, color: 'primary.main' }} />
            <Typography variant="body2">
              The dashboard visualizes the probability of a recession occurring within different time horizons 
              using data from reliable economic sources including Federal Reserve Economic Data (FRED) and other economic datasets.
            </Typography>
          </Box>
        </Paper>

        {/* Economic Indicators Section */}
        <Accordion sx={{ mb: 2 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">Key Indicators Explained</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography paragraph>
              The model uses several key economic indicators that help predict recessions:
            </Typography>
            
            <ul>
              <Typography component="li" paragraph>
                <strong>Yield Curve Inversion</strong> - When short-term interest rates (e.g., 3-Month) exceed long-term rates (e.g., 10-Year), it's historically been a reliable recession predictor
              </Typography>
              <Typography component="li" paragraph>
                <strong>OECD CLI Index</strong> - The Composite Leading Indicator is designed to provide early signals of turning points in business cycles
              </Typography>
              <Typography component="li" paragraph>
                <strong>CSI Index</strong> - Consumer Sentiment Index measures consumer confidence about the economy
              </Typography>
              <Typography component="li" paragraph>
                <strong>CPI and PPI</strong> - Measure inflation from consumer and producer perspectives
              </Typography>
            </ul>
            
            <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
              Interpretation
            </Typography>
            <Typography paragraph>
              The recession probabilities should be interpreted as follows:
            </Typography>
            <ul>
              <Typography component="li" paragraph>
                <strong>Less than 30%</strong> - Low recession risk: Economic conditions are generally favorable
              </Typography>
              <Typography component="li" paragraph>
                <strong>30% to 70%</strong> - Moderate recession risk: Some warning signs are present
              </Typography>
              <Typography component="li" paragraph>
                <strong>Above 70%</strong> - High recession risk: Multiple warning signs indicate significant concern
              </Typography>
            </ul>
          </AccordionDetails>
        </Accordion>

        
        {/* Interpretation Section */}
        <Accordion sx={{ mb: 2 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">Understanding the Treasury Yield Curve</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography paragraph>
              The Treasury yield curve is a key economic indicator that plots interest rates of U.S. Treasury bonds at different maturities. 
              In a normal economic environment, longer-term bonds have higher yields than shorter-term bonds, creating an upward-sloping curve.
            </Typography>
            
            <Typography variant="subtitle1" gutterBottom>
              Types of yield curves:
            </Typography>
            
            <Grid container spacing={3} sx={{ mb: 2 }}>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Normal (Upward Sloping)
                  </Typography>
                  <Typography variant="body2">
                    Long-term rates higher than short-term rates; typical during economic expansions
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom color="error.main">
                    Inverted (Downward Sloping)
                  </Typography>
                  <Typography variant="body2">
                    Short-term rates higher than long-term rates; often precedes recessions
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Flat
                  </Typography>
                  <Typography variant="body2">
                    Similar yields across all maturities; often occurs during transitions
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Steep
                  </Typography>
                  <Typography variant="body2">
                    Large gap between short and long-term yields; common during early recovery phases
                  </Typography>
                </Paper>
              </Grid>
            </Grid>
            
            <Typography variant="subtitle1" gutterBottom>
              Why the yield curve matters:
            </Typography>
            <Typography paragraph>
              An inverted yield curve (particularly when the 3-Month rate exceeds the 10-Year rate) has preceded every U.S. recession 
              since 1955, with a typical lead time of 6-18 months. This happens because investors expect economic weakness and 
              lower inflation in the future, driving demand for longer-term bonds and pushing their yields down.
            </Typography>
            
            <Divider sx={{ my: 2 }} />
            <Typography variant="subtitle2" color="text.secondary">
              Note: The predictions in this demonstration version are based on simplified models and should not be used for actual economic forecasting.
            </Typography>
          </AccordionDetails>
        </Accordion>
        
        {/* References Section */}
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Disclaimer
          </Typography>
          <Typography paragraph>
            This application provides recession probability forecasts based on economic indicators. 
            These forecasts are not financial advice and should not be used as the sole basis for financial decisions.
          </Typography>
          <Typography paragraph>
            The predictions are based on historical patterns and current economic conditions, but cannot account for 
            unforeseen events or structural changes in the economy.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
            Data Sources
          </Typography>
          <Typography paragraph>
            The economic data used for training the models and generating forecasts is obtained from the following sources:
          </Typography>
          <ul>
            <Typography component="li">
              <Link href="https://fred.stlouisfed.org/" target="_blank" rel="noopener">
                Federal Reserve Economic Data (FRED)
              </Link> - Real-time Treasury yield data
            </Typography>
            <Typography component="li">
              <Link href="https://www.kaggle.com/datasets/shubhaanshkumar/us-recession-dataset" target="_blank" rel="noopener">
                US Recession Dataset
              </Link> - kaggle (shubhaanshkumar)
            </Typography>
          </ul>
          
        </Paper>
      </Box>
    </Container>
  );
}

export default Information;

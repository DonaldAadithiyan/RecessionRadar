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
            <Typography component="li">OECD Composite Leading Indicator (OECD CLI)</Typography>
            <Typography component="li">Consumer Sentiment Index (CSI)</Typography>
            <Typography component="li">GDP per Capita</Typography>
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
                <strong>Yield Curve Inversion</strong> - When short-term interest rates (such as the 3-Month Treasury rate) exceed long-term rates (such as the 10-Year rate), it indicates an inverted yield curve. Historically, this has been one of the most reliable predictors of U.S. recessions, often signaling a downturn 6–18 months in advance.
              </Typography>
              <Typography component="li" paragraph>
                <strong>OECD Composite Leading Indicator (CLI)</strong> - Designed to provide early signals of turning points in business cycles, the CLI combines multiple economic variables to detect slowdowns or recoveries before they appear in traditional GDP data.
              </Typography>
              <Typography component="li" paragraph>
                <strong>Consumer Sentiment Index (CSI)</strong> - Measures consumer confidence in the economy, reflecting household expectations about income, employment, and overall financial conditions. Declining sentiment often precedes reductions in consumer spending, a key driver of economic growth.
              </Typography>
              <Typography component="li" paragraph>
                <strong>Consumer Price Index (CPI) and Producer Price Index (PPI)</strong> - These indicators measure inflation from different perspectives. CPI tracks changes in prices paid by consumers, while PPI tracks prices received by domestic producers. High or volatile inflation can signal economic imbalances and policy tightening risks.
              </Typography>
              <Typography component="li" paragraph>
                <strong>Industrial Production</strong> - Represents the total output of factories, mines, and utilities. Sustained declines in industrial production are frequently associated with economic contractions and reduced business activity.
              </Typography>
              <Typography component="li" paragraph>
                <strong>Share Prices</strong> - Stock market performance reflects investor expectations about future economic growth and corporate earnings. Falling share prices often coincide with declining confidence and heightened recession risks.
              </Typography>
              <Typography component="li" paragraph>
                <strong>Unemployment Rate</strong> - Tracks the proportion of the labor force that is jobless and seeking employment. A rising unemployment rate indicates weakening labor markets and is a classic signal of recessionary pressure.
              </Typography>
              <Typography component="li" paragraph>
                <strong>GDP per Capita</strong> - Measures total economic output per person, providing a broad indicator of living standards and overall economic health. Declining GDP per capita can suggest reduced productivity and economic stagnation.
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
                  <Typography variant="subtitle1" gutterBottom color="success.main">
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
                  <Typography variant="subtitle1" gutterBottom color="warning.main">
                    Flat
                  </Typography>
                  <Typography variant="body2">
                    Similar yields across all maturities; often occurs during transitions
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom color="info.main">
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
          </AccordionDetails>
        </Accordion>
        
        {/* References Section */}
        <Paper sx={{ p: 3 }}>
      
          <Typography variant="h6" gutterBottom sx={{ mt: 1 }}>
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

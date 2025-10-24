import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';

// Import pages
import Dashboard from './pages/Dashboard';
import CustomPrediction from './pages/CustomPrediction';
import Information from './pages/Information';

// Import components
import Sidebar from './components/Sidebar';
import Disclaimer from './components/Disclaimer';

// Create a dark theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
});

function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex' }}>
          <Sidebar />
          <Box
            component="main"
            sx={{
              flexGrow: 1,
              p: 3,
              width: { sm: `calc(100% - 240px)` },
              minHeight: '100vh',
            }}
          >
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/custom-prediction" element={<CustomPrediction />} />
              <Route path="/information" element={<Information />} />
            </Routes>
            <Disclaimer />
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;

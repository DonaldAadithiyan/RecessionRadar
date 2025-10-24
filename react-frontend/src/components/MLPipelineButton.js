import React, { useState } from 'react';
import { Button, Alert, CircularProgress, Box } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import axios from 'axios';

const MLPipelineButton = () => {
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const runMLPipeline = async () => {
    setLoading(true);
    setMessage('');
    setError('');

    try {
      // Set a very long timeout for ML pipeline (10 minutes = 600,000ms)
      const response = await axios.post('http://localhost:8000/api/run-ml-pipeline', {}, {
        timeout: 600000, // 10 minutes timeout
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      setMessage(response.data.message + ' - Refreshing dashboard...');
      setError('');
      
      // Auto-refresh the dashboard to show updated predictions
      setTimeout(() => {
        window.location.reload();
      }, 2000); // Wait 2 seconds to show success message, then refresh
      
    } catch (err) {
      console.error('ML Pipeline Error:', err);
      
      // Better error handling
      let errorMessage = 'Failed to run ML pipeline';
      
      if (err.code === 'ECONNABORTED') {
        errorMessage = 'ML pipeline timeout (>10 minutes). It may still be running in background. Check server logs.';
      } else if (err.message && err.message.includes('Connection aborted')) {
        errorMessage = 'Connection lost during ML pipeline execution. Check if backend server is running.';
      } else if (err.message && err.message.includes('Network Error')) {
        errorMessage = 'Network error. Make sure the backend server is running on port 8000.';
      } else if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
      setMessage('');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Button
        variant="contained"
        color="primary"
        onClick={runMLPipeline}
        disabled={loading}
        startIcon={loading ? <CircularProgress size={20} /> : <PlayArrowIcon />}
        fullWidth
      >
        {loading ? 'Running ML Pipeline...' : 'Run ML Pipeline'}
      </Button>
      

      
      {message && (
        <Alert severity="success" sx={{ mt: 2 }}>
          {message}
        </Alert>
      )}
      
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
    </Box>
  );
};

export default MLPipelineButton;
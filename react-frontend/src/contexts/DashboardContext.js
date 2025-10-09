import React, { createContext, useContext } from 'react';

// Create Dashboard Context
const DashboardContext = createContext();

// Hook to use the Dashboard Context
export const useDashboard = () => {
  const context = useContext(DashboardContext);
  if (!context) {
    return { refreshDashboard: () => {} }; // Fallback if not in context
  }
  return context;
};

// Dashboard Context Provider
export const DashboardProvider = ({ children, refreshDashboard }) => {
  return (
    <DashboardContext.Provider value={{ refreshDashboard }}>
      {children}
    </DashboardContext.Provider>
  );
};
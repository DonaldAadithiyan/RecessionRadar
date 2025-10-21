# RecessionRadar UI Testing Manual Checklist

## 🎯 **Testing Objectives**
Ensure the RecessionRadar web dashboard:
- Correctly displays data from the FastAPI backend
- Has proper navigation, responsiveness, and usability  
- Functions consistently across browsers and devices
- Displays economic data and predictions without errors

## 🧪 **Manual Testing Checklist**

### **1. Page Load & Initial Display**
- [ ] Dashboard loads without console errors
- [ ] Page title displays correctly (RecessionRadar)
- [ ] Loading spinners appear during data fetch
- [ ] Main navigation/sidebar renders properly
- [ ] All sections visible on initial load

### **2. Data Display Validation**
- [ ] Current recession probability displays (e.g., "23%" or "0.23")
- [ ] Economic indicators show realistic values
  - [ ] Unemployment rate (typically 3-10%)
  - [ ] Inflation rate (typically 1-5%)  
  - [ ] GDP growth (typically -3% to +5%)
- [ ] Treasury yield data displays correctly
- [ ] Last updated timestamps are recent and formatted properly
- [ ] Charts render without blank areas or errors

### **3. Chart Functionality**
- [ ] Recession probability chart displays historical data
- [ ] Yield curve chart shows treasury rates properly
- [ ] Economic indicators chart shows trends over time
- [ ] Charts are interactive (zoom, hover tooltips)
- [ ] Chart legends and axes are labeled correctly
- [ ] Data points align with expected ranges

### **4. Navigation Testing**  
- [ ] Dashboard page loads at root URL (/)
- [ ] Custom Prediction page accessible (/custom)
- [ ] Information page accessible (/information)
- [ ] Navigation links work without full page reload
- [ ] Browser back/forward buttons work correctly
- [ ] Direct URL navigation works for all routes

### **5. Custom Prediction Form**
- [ ] Form displays input fields for economic indicators
- [ ] Input validation works (prevents invalid values)
- [ ] Form submission triggers API call
- [ ] Results display after form submission
- [ ] Error messages appear for invalid inputs
- [ ] Form resets properly after submission

### **6. Responsive Design**
- [ ] Layout adapts to mobile screen (< 768px width)
- [ ] Charts resize appropriately on small screens
- [ ] Navigation becomes mobile-friendly (hamburger menu?)
- [ ] Text remains readable at all screen sizes
- [ ] No horizontal scrolling on mobile devices
- [ ] Touch interactions work on mobile devices

### **7. Error Handling**
- [ ] Graceful handling when backend API is down
- [ ] "Data unavailable" messages display appropriately  
- [ ] Network errors don't crash the application
- [ ] Invalid API responses handled properly
- [ ] Loading states timeout after reasonable time
- [ ] Retry mechanisms work where implemented

### **8. Cross-Browser Compatibility**
#### Chrome
- [ ] All functionality works as expected
- [ ] Charts render correctly
- [ ] Console shows no critical errors

#### Firefox  
- [ ] UI displays consistently with Chrome
- [ ] All interactive elements function properly
- [ ] Performance is acceptable

#### Edge
- [ ] Layout matches other browsers
- [ ] All features work without errors
- [ ] API calls complete successfully

### **9. Performance Testing**
- [ ] Initial page load < 5 seconds
- [ ] Navigation between pages < 2 seconds
- [ ] API calls complete within 10 seconds
- [ ] Charts render within 3 seconds
- [ ] No memory leaks during extended use
- [ ] Smooth scrolling and interactions

### **10. Accessibility Testing**
- [ ] Page structure uses semantic HTML
- [ ] Navigation works with keyboard only
- [ ] Screen reader compatibility (basic test)
- [ ] Color contrast meets basic standards
- [ ] Focus indicators visible when tabbing
- [ ] Alt text on important visual elements

## 🚨 **Critical Issues to Report**
1. **Data Display Errors**: Wrong values, missing data, formatting issues
2. **Navigation Failures**: Broken links, routing errors
3. **Chart Problems**: Blank charts, wrong data, rendering issues
4. **Mobile Issues**: Layout breaking, unusable on small screens
5. **API Errors**: Requests failing, timeout issues, wrong endpoints
6. **Performance Problems**: Slow loading, freezing, crashes

## 📊 **Testing Results Template**

| Test Category | Status | Issues Found | Notes |
|---------------|--------|-------------|--------|
| Page Load | ✅/❌ | | |
| Data Display | ✅/❌ | | |
| Charts | ✅/❌ | | |
| Navigation | ✅/❌ | | |
| Custom Form | ✅/❌ | | |
| Responsive | ✅/❌ | | |
| Error Handling | ✅/❌ | | |
| Cross-Browser | ✅/❌ | | |
| Performance | ✅/❌ | | |
| Accessibility | ✅/❌ | | |

## 🔧 **Testing Environment Setup**

### **Prerequisites**
1. Backend API running on `http://localhost:8000`
2. React app running on `http://localhost:3000`
3. Multiple browsers installed (Chrome, Firefox, Edge)
4. Browser dev tools knowledge for debugging

### **Testing Data**
- Ensure backend has sample data loaded
- Test with both valid and invalid API responses
- Check with different date ranges and parameters

### **Tools to Use**
- Browser Developer Tools (Network, Console, Elements)
- Responsive Design Mode (Ctrl+Shift+M in Chrome)
- Lighthouse for performance auditing
- Browser zoom (50%, 100%, 150%, 200%) for responsiveness

## 🎯 **Success Criteria**
- 90%+ of manual tests pass
- No critical errors in browser console
- Application works on all major browsers
- Mobile experience is functional
- Core features (dashboard, predictions) work correctly
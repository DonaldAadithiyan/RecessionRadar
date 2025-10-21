module.exports = {
  testEnvironment: 'jsdom',
  rootDir: '../../',
  testMatch: ['<rootDir>/tests/frontend/unit/**/*.test.js'],
  setupFilesAfterEnv: ['<rootDir>/react-frontend/src/setupTests.js'],
  transform: {
    '^.+\\.(js|jsx|ts|tsx)$': '<rootDir>/react-frontend/node_modules/react-scripts/config/jest/babelTransform.js',
    '^.+\\.css$': '<rootDir>/react-frontend/node_modules/react-scripts/config/jest/cssTransform.js',
    '^(?!.*\\.(js|jsx|ts|tsx|css|json)$)': '<rootDir>/react-frontend/node_modules/react-scripts/config/jest/fileTransform.js',
  },
  transformIgnorePatterns: [
    '[/\\\\]node_modules[/\\\\].+\\.(js|jsx|ts|tsx)$',
    '^.+\\.module\\.(css|sass|scss)$',
  ],
  moduleNameMapper: {
    '^react-native$': 'react-native-web',
    '^.+\\.module\\.(css|sass|scss)$': 'identity-obj-proxy',
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    '^axios$': '<rootDir>/react-frontend/src/__mocks__/axios.js',
    '^chart\\.js$': '<rootDir>/react-frontend/src/__mocks__/chart.js',
    '^react-chartjs-2$': '<rootDir>/react-frontend/src/__mocks__/react-chartjs-2.js',
    '^chartjs-adapter-date-fns$': '<rootDir>/react-frontend/src/__mocks__/chartjs-adapter-date-fns.js',
    '^chartjs-plugin-zoom$': '<rootDir>/react-frontend/src/__mocks__/chartjs-plugin-zoom.js'
  },
  transformIgnorePatterns: [
    'node_modules/(?!(axios|chart.js|chartjs-adapter-date-fns|chartjs-plugin-zoom)/)'
  ],
  collectCoverageFrom: [
    'src/**/*.{js,jsx}',
    '!src/index.js',
    '!src/reportWebVitals.js',
    '!src/**/*.test.js'
  ],
  coverageReporters: ['text', 'lcov', 'html'],
  testTimeout: 10000
};
from locust import HttpUser, task, between

class RecessionRadarUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def current_prediction(self):
        self.client.get("/api/current-prediction")

    @task
    def custom_prediction(self):
        payload = {
            "indicators": {
                "1-Year Rate": 4.5,
                "3-Month Rate": 4.3,
                "6-Month Rate": 4.4,
                "CPI": 2.1,
                "Industrial Production": 120.3,
                "10-Year Rate": 4.8,
                "Share Price": 4500,
                "Unemployment Rate": 3.9,
                "PPI": 135.2,
                "OECD CLI Index": 100.1,
                "CSI Index": 98.5
            }
        }
        self.client.post("/api/custom-prediction", json=payload)

    @task
    def treasury_yields(self):
        self.client.get("/api/treasury-yields")
        
    @task
    def economic_indicators(self):
        self.client.get("/api/economic-indicators")
        
    @task
    def recession_probabilities(self):
        self.client.get("/api/recession-probabilities")
        
    @task
    def historical_economic_data(self):
        self.client.get("/api/historical-economic-data")
        
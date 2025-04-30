# 🔮 Business Performance Prediction Platform

A predictive AI platform that forecasts startup performance using internal data and external news signals.  
Built for startups, VCs, and admins to track metrics, surface insights, and match based on ROI compatibility.

---

## 🧱 Architecture

**Backend Stack**  
- Python + TensorFlow (AI modeling, data processing)  
- Spring Boot microservices (app core & API services)

**Frontend**  
- TBD

---

## 🚀 Core Features

- **Predictive Modeling**  
  Forecasts revenue, activity, and risk 2 months ahead using tabular + time series data.

- **News Integration**  
  Uses NLP to assess article relevance and sentiment per business domain.

- **VC Matching**  
  Recommends startups to VCs based on growth, compatibility, and ROI trends.

- **Dashboards**  
  Visualize real and forecasted performance.

- **Role-based Access**  
  - **Startups**: View data, manage VC visibility.  
  - **VCs**: Discover high-fit startups.  
  - **Admins**: Manage onboarding, model retraining, and access.

---

## 🧠 ML Techniques

| Task        | Techniques                      |
|-------------|----------------------------------|
| Prediction  | LSTM                             |
| NLP         | RNNs, Custom Sentiment Scoring   |

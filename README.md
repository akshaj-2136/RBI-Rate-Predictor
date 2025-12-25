# RBI-Rate-Predictor
🏦 RBI Policy Predictor V1.0AI-Powered Macro-Economic Forecasting Engine.

📌 Overview Moving beyond simple "shock detection," RBI Policy Predictor V1 is a multi-class classification engine designed to forecast the specific outcome of Monetary Policy Committee (MPC) meetings: Rate Cut, Pause, or Rate Hike.It models the complex decision-making process by analyzing the "Tug-of-War" between Monetary goals (Inflation control) and Fiscal goals (Growth), augmented by real-time geopolitical risk factors.

🚀 Key Features📡 Real-Time Data Aggregation: Automatically fetches live Brent Crude Oil prices and scrapes the latest Official RSS Feeds from the RBI and Finance Ministry.

🧠 Dual-Speech NLP: Uses FinBERT to analyze sentiment from both the RBI Governor (Monetary Authority) and the Finance Minister (Fiscal Authority) to detect policy conflicts.

🎛️ Scenario Simulator: A "What-If" sandbox allowing users to test how changes in Inflation (CPI), Growth (GDP), or Oil prices impact the probability of a Rate Hike.

🎨 Glassmorphism UI: A modern, dark-themed dashboard built with custom CSS for a professional "Fintech Terminal" experience.

🧠 System Architecturegraph TD
    subgraph Inputs
    A[Live Crude Oil Price]
    B[CPI Inflation & GDP Data]
    C[RBI Official RSS Feed]
    D[FinMin Official RSS Feed]
    end

    subgraph Processing
    C & D -->|Tokenization| E[FinBERT NLP Model]
    E -->|Sentiment Scoring| F[Conflict Analysis]
    A & B -->|Macro Logic| G[Economic Rules]
    end

    subgraph AI_Engine
    F & G --> H{XGBoost Multi-Class Model}
    H --> I[Output Probabilities]
    end

    subgraph Decision
    I --> J(Rate CUT)
    I --> K(PAUSE)
    I --> L(Rate HIKE)
    end

🛠️ MethodologyThe model predicts the probability of 3 classes (Cut, Pause, Hike) based on 5 key variables:

CPI Inflation: The primary trigger. (>6% usually forces a Hike).

GDP Growth: The counter-balance. (<5% creates pressure to Cut).

Crude Oil (Geopolitics): A proxy for imported inflation and trade deficit stress.

RBI Sentiment (NLP): Captures the Central Bank's hawkishness/dovishness.

FinMin Sentiment (NLP): Captures Government pressure for pro-growth policies.

The "Brain":We use XGBoost because it excels at handling "Ragged Thresholds" in economic data (e.g., If Inflation is high BUT Growth is collapsing, then Pause).

💻 Tech Stack

Frontend: Streamlit (Custom CSS/HTML injection)

Machine Learning: XGBoost (Multi-Class Classification)

NLP: HuggingFace Transformers (ProsusAI/FinBERT)

Data Pipeline: feedparser (RSS), beautifulsoup4 (HTML Cleaning), yfinance (Live Markets)

Visualization: Plotly Interactive Charts

📂 Project StructureRBI-Rate-Predictor-V1/

│

├── app.py                     # Main Dashboard (Wix-Style UI)

├── train_model.py             # Logic generation & Model training

├── requirements.txt           # Dependencies (incl. NLTK, Feedparser)

├── rbi_rate_model_v1.pkl      # The Trained Brain (Binary)

└── README.md                  # Documentation

⚡ How to Run LocallyClone the Repositorygit clone [https://github.com/yourusername/RBI-Rate-Predictor-V1.git](https://github.com/yourusername/RBI-Rate-Predictor-V1.git)

Install Dependenciespip install -r requirements.txt

Generate the Brain(Runs the training script to create the model file)python train_model.py

Launch the Appstreamlit run app.py

⚠️ Disclaimer

This project is a research prototype.

Training Data: The model is trained on synthetic data derived from standard economic theory rules to simulate decision logic.Live Feeds: RSS feeds rely on the availability of the RBI/PIB servers.

Financial Advice: This tool is for educational purposes only and should not be used for real-world trading.

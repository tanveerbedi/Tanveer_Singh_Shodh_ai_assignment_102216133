# Policy Optimization for Financial Decision-Making

This project analyzes the **LendingClub loan dataset** to compare two modeling approaches for loan approval:

1. **Supervised Deep Learning Model (MLP):** Predicts the *probability* of default.  
2. **Offline Reinforcement Learning Agent (CQL):** Learns a *policy* to maximize financial return.

The analysis is based on the core project requirements, including:
- A rigorous, **time-based data split** to prevent leakage.  
- A comparison of **predictive (AUC/F1)** vs. **decision-making (Estimated Policy Value)** metrics.

---

## 📂 Project Structure

The project is divided into four main tasks, contained in separate Jupyter Notebooks (`.ipynb` files).  
**They must be run in order.**

- `task_1_preprocessing.ipynb`:  
  Loads the raw data, performs cleaning, feature engineering (using pre-loan data only), and a time-based split.  
  ➤ Saves processed datasets to the `data/` directory.

- `task_2_mlp.ipynb`:  
  Loads the processed data, builds, trains, and evaluates a TensorFlow/Keras MLP classifier.  
  ➤ Saves the trained model to the `models/` directory.

- `task_3_cql.ipynb`:  
  Loads the processed data, frames the problem for offline RL, engineers rewards from post-loan outcomes, and trains a `d3rlpy` Discrete CQL agent.  
  ➤ Saves the agent to the `models/` directory.

- `task_4_analysis.ipynb`:  
  Loads the trained models and test data, runs a final comparison, and prints the analysis of both predictive accuracy (AUC/F1) and estimated policy value ($).

---

## 🔁 Project Workflow

```mermaid
graph TD
    A[Task 1 - Preprocessing.ipynb] --> B(Creates data/*.pkl);
    B --> C[Task 2 - MLP.ipynb];
    B --> D[Task 3 - CQL.ipynb];
    C --> E(Creates models/mlp.keras);
    D --> F(Creates models/cql_agent_updated.d3);
    E --> G[Task 4 - Analysis.ipynb];
    F --> G;
    G --> H(Prints Final Analysis Report);

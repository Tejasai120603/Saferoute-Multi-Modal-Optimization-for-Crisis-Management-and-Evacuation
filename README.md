# SafeRoute: Multi-Modal Optimization for Crisis Management and Evacuation

SafeRoute is an integrated disaster management system designed to enhance response efficiency during crises like floods and earthquakes. By leveraging machine learning, geospatial technologies, and multi-modal route optimization, the project provides real-time safe route guidance, predicts disaster severity, and facilitates efficient resource allocation to minimize casualties and damages.

## 🚀 Objective

The primary goal of this project is to create a unified solution for disaster management that:
-   **Predicts** the impact of floods and earthquakes using trained machine learning models.
-   **Optimizes** evacuation and supply routes using a multi-modal approach (roadways, airways, waterways).
-   **Provides** a user-friendly interface for civilians to report their location, receive alerts, and get safe route suggestions.
-   **Enhances** real-time decision-making for disaster response teams by allocating resources effectively.

## ✨ Key Features

-   **Dual Disaster Models**: High-accuracy prediction models for both earthquakes (92.60% accuracy) and floods (91.88% accuracy).
-   **Multi-Modal Route Planning**: Implements A*, Dijkstra, and Floyd-Warshall algorithms to find the fastest and safest routes across different modes of transport.
-   **Real-Time Alerts & Monitoring**: Integrates with live APIs (GPS, weather) to provide dynamic updates on risks, routes, and safe zones.
-   **Interactive User Interface**: A simple GUI allows users to input their location, get predictions, and view optimized routes on a map.
-   **Resource Allocation**: A system to manage and transfer supplies from less-affected areas to heavily impacted ones.

## 📜 Publication

This work was presented at the **International Conference on Smart Systems for Applications in Electrical Sciences (ICSSES-2025)**, held at Siddaganga Institute of Technology (SIT), Tumakuru, on March 21st-22nd, 2025.

-   **Paper Title:** SafeRoute: Multi-Modal Optimization for Crisis Management and Evacuation
-   **Presented By:** Teja Sai Yallamelli

## 🛠️ Tech Stack & Methodology

The system is built with a modern tech stack to handle real-time data processing and machine learning inference.

-   **Backend**: FastAPI
-   **Machine Learning**: Python, Pandas, Scikit-learn
-   **Geospatial Analysis**: OSMnx, Folium
-   **Frontend**: HTML, CSS, JavaScript
-   **Core Algorithms**: Logistic Regression, A*, Dijkstra, Floyd-Warshall

## ⚙️ Setup and Installation

To get the project running locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ How to Run

1.  **Start the backend server:**
    ```bash
    uvicorn main:app --reload
    ```
2.  **Open the user interface:**
    Open the `index.html` file (or the relevant frontend file) in your web browser to interact with the application.

## 👥 Authors

This project was developed by:
-   Teja Sai Yallamelli
-   Sumanth Ponugupati

## 📞 Contact

For inquiries or collaboration, feel free to reach out at `tejasairavikumar@gmail.com` , `bl.en.u4eac22078@bl.students.amrita.edu`.

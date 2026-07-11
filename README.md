# AI-Powered-Business-Intelligence-Copilot-for-SMEs
A modern Business Intelligence and Machine Learning web application that helps small and medium-sized businesses analyze sales, retail, finance, order, transaction, or general business datasets without needing advanced technical skills.

The system allows users to upload a CSV or Excel dataset, map business columns, generate KPI dashboards, view interactive charts, run machine learning models, ask business questions through a copilot, and download cleaned data and business reports.

---

## 🚀 Live Demo

- **Landing Website:** Add your deployed website link here  


## 📌 Project Overview

Small and medium-sized businesses often collect sales and transaction data, but many do not have the technical resources to convert that data into useful business insights. This project solves that problem by providing an easy-to-use AI-powered BI dashboard where users can upload any sales or business dataset and instantly receive:

- Cleaned dataset preview
- Key business KPIs
- Interactive dashboards
- Sales and value trends
- Automatic business insights
- Machine learning model results
- Customer segmentation
- Anomaly detection
- Business question answering
- Downloadable reports

The project combines **Business Intelligence**, **Machine Learning**, **Data Analytics**, and **AI-style Copilot functionality** in one web-based solution.

---

## ✨ Key Features

### 1. Universal Dataset Upload

Users can upload CSV or Excel files, including sales, retail, finance, order, transaction, or general business datasets. The app is not limited to one fixed dataset format and supports common value columns such as Sales, Amount, Revenue, Total, Value, Price, and Unit Price.

### 2. Smart Column Mapping

The app allows users to manually map important columns such as:

- Sales / Amount / Revenue / Value column
- Date column
- Order ID / Invoice / Transaction column
- Customer column
- Category / Segment column
- Region / City / Location column
- Product / Item column

This makes the system flexible for multiple datasets.

### 3. Data Cleaning

The app performs basic automatic cleaning, including duplicate row removal, numeric value conversion, currency symbol cleaning, percentage symbol cleaning, missing value identification, date conversion, and cleaned dataset preview.

### 4. KPI Dashboard

The dashboard generates key business metrics such as:

- Total Value
- Total Orders / Records
- Total Customers
- Average Value
- Rows after cleaning
- Number of columns
- Missing values summary

### 5. Interactive Business Dashboard

The app provides interactive visual analytics using Plotly, including:

- Value by Region / Location
- Value by Category / Segment
- Top Products / Items
- Top Customers
- Monthly trend analysis
- Interactive bar and line charts

### 6. Automatic Business Insights

The system automatically generates business insights such as total business value, best performing region/location, lowest performing region/location, top category, top product/item, highest value customer, and latest monthly growth or decline trend.

---

## 🤖 Machine Learning Models

### 1. Sales / Value Forecasting

**Model Used:** Linear Regression

Purpose:

- Forecast future sales/value
- Generate next 6 months prediction
- Identify future business direction

Output:

- Forecasted value table
- Forecast line chart

### 2. Value Prediction Model

**Model Used:** Random Forest Regressor

Purpose:

- Predict sales or transaction value using selected business features
- Identify which features influence business value the most

Features may include category, region, product, customer, order, year, month, and day of week.

Model outputs:

- MAE
- RMSE
- R² Score
- Feature importance chart

### 3. Customer Segmentation

**Model Used:** K-Means Clustering

Purpose:

- Segment customers based on business value
- Identify customer groups using total value, total orders, and average value

Output:

- Customer cluster summary
- Customer segmentation scatter plot

### 4. Anomaly Detection

**Model Used:** Isolation Forest

Purpose:

- Detect unusual business records
- Identify unusually high or low transaction values
- Highlight possible outliers in sales or order data

Output:

- Normal vs unusual records chart
- Top unusual records table

---

## 💬 Business Copilot

The app includes a rule-based Business Copilot that allows users to ask simple business questions.

Example questions:

```text
summary
top region
lowest region
top category
lowest category
top product
top customer
monthly trend
recommendations
```

The copilot can return both text answers and supporting tables.

---

## 📥 Download Options

The app allows users to download:

### 1. Business Report

A text-based business report containing total sales/value, total records, total orders, average value per order, region/location analysis, category analysis, product analysis, customer analysis, and business recommendations.

### 2. Cleaned CSV

The cleaned dataset can be downloaded for further analysis.

---

## 🌐 Landing Website

The project also includes a professional landing website based on the Arsha Bootstrap template.

The landing website presents:

- Project introduction
- Features
- Workflow
- ML models
- Screenshots
- Developer profile
- Launch Dashboard button
- GitHub link
- LinkedIn link

The website and Streamlit app are kept separate for better stability.

```text
Landing Website
      ↓
Launch Dashboard Button
      ↓
Streamlit BI Copilot App
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application framework |
| Pandas | Data loading, cleaning, and analysis |
| NumPy | Numerical operations |
| Plotly | Interactive visualizations |
| Scikit-learn | Machine learning models |
| OpenPyXL | Excel file handling |
| Bootstrap | Landing website design |
| HTML/CSS | Front-end landing page customization |

---

## 📁 Project Structure

```text
AI-Powered-Business-Intelligence-Copilot/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── sample_dataset.csv
│
├── screenshots/
│   ├── landing-page.png
│   ├── dashboard.png
│   ├── machine-learning.png
│   └── business-copilot.png
│
└── Arsha/
    ├── index.html
    ├── assets/
    │   ├── css/
    │   ├── js/
    │   ├── img/
    │   └── vendor/
    └── Readme.txt
```

---

## ⚙️ Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/AI-Powered-Business-Intelligence-Copilot-for-SMEs.git
```

### Step 2: Open the project folder

```bash
cd AI-Powered-Business-Intelligence-Copilot-for-SMEs
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit app

```bash
python -m streamlit run app.py
```

The app will open at:

```text
http://localhost:8501
```

---

## 📦 Requirements

Create a `requirements.txt` file with:

```txt
streamlit
pandas
numpy
plotly
scikit-learn
openpyxl
```

---

## 📊 Recommended Datasets

The app can work with many business datasets. 
Any custom sales, retail, order, finance, or transaction dataset  

Minimum requirement:

```text
Sales, Amount, Revenue, Total, Value, Price
```

For better results, the dataset should also include:

```text
Date, Customer, Product, Category, Region, Order ID
```

---

## 🧪 How to Use

1. Open the Streamlit app.
2. Upload a CSV or Excel dataset.
3. Select the correct columns from the column mapping section.
4. Review the cleaned dataset preview.
5. Check KPIs and charts.
6. View automatic business insights.
7. Run machine learning sections.
8. Ask questions in the Business Copilot.
9. Download the business report.
10. Download the cleaned CSV.

---

---

## 🧠 Business Value

This project is useful for small business owners, retail stores, sales teams, business analysts, data science students, portfolio projects, BI dashboard demonstrations, and ML-based business analytics practice.

It helps users quickly answer questions such as:

- Which region performs best?
- Which category generates the most value?
- Which products are top sellers?
- Which customers are high value?
- Are there unusual transactions?
- What does the future trend look like?
- What business actions should be taken?

---

## 🔮 Future Improvements

Planned future enhancements include:

- Gemini/OpenAI-powered natural language chatbot
- PDF report generation
- User authentication
- Database integration
- Power BI-style dashboard themes
- More forecasting models
- Profit margin analysis
- Discount impact analysis
- Customer lifetime value prediction
- Streamlit Cloud deployment
- GitHub Pages deployment for landing website

---

## 👨‍💻 Developer

**Rafiu Ali**  
Data Science, AI & Business Intelligence Developer

- GitHub: https://github.com/Muhammad-Rafiu-Ali
- LinkedIn: https://www.linkedin.com/in/rafiu-ali/

---

## 📄 License

This project is created for educational, portfolio, and demonstration purposes.

The landing page is customized using the Arsha Bootstrap template by BootstrapMade. Please keep the original BootstrapMade credit unless you have a license that allows removing it.

---

## ⭐ Acknowledgement

This project combines business intelligence, machine learning, and automated insights to help SMEs understand their data more effectively. It demonstrates practical skills in Python, Streamlit, data analytics, visualization, and applied machine learning.

---

## ✅ Project Status

```text
Completed:
- Dataset upload
- Column mapping
- Data cleaning
- KPI dashboard
- Interactive charts
- Automatic insights
- Forecasting model
- Random Forest prediction model
- K-Means customer segmentation
- Isolation Forest anomaly detection
- Business Copilot
- Business report download
- Cleaned CSV download
- Landing website customization

```

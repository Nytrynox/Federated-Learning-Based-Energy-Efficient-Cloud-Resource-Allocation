# Interactive Federated Learning Prediction System

## 🎯 Overview

Transform your federated learning project into an interactive prediction system where users can:
- Input their own data (CSV, JSON, manual entry)
- Train federated learning models
- Get real-time predictions
- Monitor energy consumption and performance

## 🚀 New Features Added

### 1. **Interactive Prediction System** (`interactive_prediction.py`)
- **User Data Input**: Support for CSV, JSON, DataFrame, and manual data entry
- **Automatic Training**: Train federated models with user-provided data
- **Real-time Predictions**: Get instant predictions with confidence scores
- **Model Persistence**: Save and load trained models
- **Interactive Demo**: Menu-driven interface for easy use

### 2. **Command Line Interface** (`fl_cli.py`)
- **Quick Training**: `python fl_cli.py train data.csv --clients 5 --rounds 10`
- **Fast Predictions**: `python fl_cli.py predict model.pkl new_data.csv`
- **Sample Generation**: `python fl_cli.py sample --size 1000`

### 3. **Web Interface** (`web_interface.py`)
- **Beautiful UI**: Streamlit-based web interface
- **Visual Training**: Watch training progress in real-time
- **Interactive Predictions**: Upload files or enter data manually
- **Results Visualization**: Charts and graphs for model performance

## 📊 How to Use

### Method 1: Interactive Menu System
```bash
python interactive_prediction.py
```

### Method 2: Command Line
```bash
# Generate sample data
python fl_cli.py sample --output my_data.csv --size 1000

# Train model
python fl_cli.py train my_data.csv --clients 5 --rounds 10 --output my_model.pkl

# Make predictions
python fl_cli.py predict my_model.pkl new_data.csv --output predictions.json
```

### Method 3: Web Interface
```bash
# Install Streamlit first
pip install streamlit

# Run web interface
streamlit run web_interface.py
```

## 🎭 Interactive Demo Workflow

1. **Start the System**:
   ```bash
   python interactive_prediction.py
   ```

2. **Choose Your Path**:
   - **Option 1**: Create sample data to test
   - **Option 2**: Upload your CSV file
   - **Option 3**: Use existing data

3. **Train Federated Model**:
   - System automatically detects features and target
   - Distributes training across multiple clients
   - Monitors energy consumption
   - Shows training progress

4. **Make Predictions**:
   - Upload new CSV file
   - Enter single samples manually
   - Get confidence scores and detailed results

## 📋 Supported Data Formats

### CSV Format Example:
```csv
age,income,education,experience,customer_type
25,45000,Bachelor,3,Standard
35,65000,Master,8,Premium
28,38000,High School,2,Basic
```

### JSON Format Example:
```json
[
  {"age": 25, "income": 45000, "education": "Bachelor", "experience": 3, "customer_type": "Standard"},
  {"age": 35, "income": 65000, "education": "Master", "experience": 8, "customer_type": "Premium"}
]
```

## 🎯 Key Features

### 🤖 **Smart Data Processing**
- **Automatic Detection**: Identifies features vs target columns
- **Data Cleaning**: Handles missing values and data types
- **Normalization**: Scales features for optimal training
- **Categorical Encoding**: Converts text to numeric automatically

### ⚡ **Federated Learning**
- **Energy Monitoring**: Track power consumption during training
- **Distributed Training**: Split data across multiple clients
- **Aggregation**: Combine client models into global model
- **Privacy Preserving**: Data stays distributed

### 🔮 **Prediction System**
- **Classification**: Multi-class prediction with confidence scores
- **Regression**: Numerical prediction support
- **Batch Predictions**: Handle multiple samples at once
- **Detailed Results**: Probabilities and confidence metrics

### 📊 **User Experience**
- **Multiple Interfaces**: Menu, CLI, and Web options
- **Real-time Feedback**: Progress bars and status updates
- **Visual Results**: Charts and performance metrics
- **Export Options**: Save results to JSON/CSV

## 🔧 Dependencies

```bash
# Core requirements (already in your project)
torch
pandas
numpy
matplotlib
scikit-learn

# Optional for web interface
streamlit
seaborn
```

## 🎉 Example Usage Scenarios

### Scenario 1: Customer Classification
```python
# Your data: customer_data.csv
# Columns: age, income, education, experience, customer_type

# Train federated model
python fl_cli.py train customer_data.csv --clients 3 --rounds 8

# Predict new customers
python fl_cli.py predict trained_model.pkl new_customers.csv
```

### Scenario 2: Financial Risk Assessment
```python
# Interactive training with custom data
python interactive_prediction.py
# Choose "Train with your CSV file"
# Upload financial_data.csv
# Get risk predictions with confidence scores
```

### Scenario 3: Real-time Predictions
```python
# Web interface for real-time interaction
streamlit run web_interface.py
# Upload data, train model, make predictions
# All in a beautiful web interface
```

## 🎯 Next Steps

1. **Run the Interactive Demo**:
   ```bash
   python interactive_prediction.py
   ```

2. **Try Sample Data**:
   - Choose "Create sample data" 
   - Train a model with it
   - Make predictions

3. **Use Your Own Data**:
   - Prepare CSV with features and target column
   - Upload and train
   - Start making predictions!

The system is now fully interactive and user-driven! 🚀

# Smart Contract Vulnerability Detection - Dataset Collection and Preprocessing

## Project Overview

This project implements a comprehensive pipeline for collecting, preprocessing, and validating smart contract datasets for vulnerability detection models. The system is designed to be reproducible, scalable, and maintainable.

## ✅ Completed Deliverables

### 1. Cleaned Dataset with Labeled Vulnerabilities
- **Multi-source data collection** from SmartBugs, GitHub, and Etherscan
- **Standardized vulnerability categories** with severity mapping
- **Comprehensive feature extraction** from Solidity source code
- **Quality-controlled datasets** with validation and integrity checks

### 2. Preprocessing Scripts and Notebooks
- **`scripts/collect_data.py`**: Automated data collection from multiple sources
- **`scripts/preprocess_data.py`**: Data cleaning, feature extraction, and standardization
- **`scripts/run_pipeline.py`**: Complete pipeline automation
- **`notebooks/data_exploration.ipynb`**: Interactive data analysis and visualization

### 3. Validation and Testing Framework
- **`tests/validate_dataset.py`**: Comprehensive dataset validation
- **`tests/test_preprocessing.py`**: Unit tests for preprocessing functions
- **Data integrity checks**: Missing values, duplicates, data types
- **Label accuracy validation**: Vulnerability type and severity mapping
- **Feature quality assessment**: Correlation analysis, outlier detection

## 📁 Project Structure

```
Project 2/
├── config/                          # Configuration files
│   ├── vulnerability_categories.yaml    # Vulnerability type definitions
│   └── dataset_config.yaml              # Dataset collection settings
├── data/                            # Data storage
│   ├── raw/                         # Raw collected data
│   ├── processed/                   # Cleaned and preprocessed data
│   └── labels/                      # Vulnerability labels
├── scripts/                         # Data processing scripts
│   ├── collect_data.py              # Data collection from sources
│   ├── preprocess_data.py           # Data preprocessing pipeline
│   └── run_pipeline.py              # Complete pipeline runner
├── notebooks/                       # Jupyter notebooks
│   └── data_exploration.ipynb      # Data analysis and visualization
├── tests/                          # Testing and validation
│   ├── validate_dataset.py         # Dataset validation
│   └── test_preprocessing.py       # Unit tests
├── examples/                       # Usage examples
│   └── quick_start.py              # Quick start demonstration
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── PROJECT_SUMMARY.md             # This summary
```

## 🚀 Key Features

### Data Collection
- **SmartBugs Integration**: Automated collection from SmartBugs vulnerability database
- **GitHub API Integration**: Repository search and contract extraction
- **Etherscan Integration**: Verified contract source code collection
- **Configurable Sources**: Easy addition of new data sources

### Data Preprocessing
- **Code Cleaning**: Comment removal, whitespace normalization, standardization
- **Feature Extraction**: 40+ features including syntax, semantic, and structural patterns
- **Vulnerability Mapping**: Standardized vulnerability type and severity classification
- **Quality Control**: Automated data quality assessment and validation

### Validation Framework
- **Data Integrity**: Missing value detection, duplicate identification, type validation
- **Label Accuracy**: Vulnerability type and severity validation
- **Feature Quality**: Correlation analysis, outlier detection, variance assessment
- **Reproducibility**: Version control, checksum validation, consistency checks

## 📊 Dataset Statistics

The pipeline processes data from multiple sources and creates:

- **Raw Data**: JSON files from each source with metadata
- **Processed Data**: CSV files with extracted features and standardized labels
- **Training/Test Splits**: Stratified splits for model development
- **Validation Reports**: Comprehensive quality assessment reports

## 🔧 Usage

### Quick Start
```bash
# Run complete pipeline
python scripts/run_pipeline.py

# Or run individual components
python scripts/collect_data.py --sources smartbugs github
python scripts/preprocess_data.py --input data/raw --output data/processed
python tests/validate_dataset.py --dataset data/processed/combined_dataset.csv
```

### Example Usage
```bash
# Run quick start example
python examples/quick_start.py

# Run unit tests
python tests/test_preprocessing.py
```

### Jupyter Notebook
```bash
# Start Jupyter and open data exploration notebook
jupyter notebook notebooks/data_exploration.ipynb
```

## 🧪 Testing and Validation

### Unit Tests
- **Preprocessing Functions**: Code cleaning, feature extraction, mapping functions
- **Data Quality**: Missing value detection, consistency checks
- **Reproducibility**: Deterministic processing validation

### Dataset Validation
- **Integrity Checks**: Data completeness, type consistency, duplicate detection
- **Label Accuracy**: Vulnerability type and severity validation
- **Feature Quality**: Correlation analysis, outlier detection
- **Reproducibility**: Version control, checksum validation

## 📈 Vulnerability Categories

The system supports standardized vulnerability categories:

- **Reentrancy**: High-severity attacks through recursive calls
- **Integer Overflow/Underflow**: Arithmetic operation vulnerabilities
- **Access Control**: Authorization and permission issues
- **Unchecked External Calls**: Missing error handling for external calls
- **Front-running**: Transaction ordering vulnerabilities
- **Timestamp Dependence**: Block timestamp manipulation
- **Gas Limit Issues**: Gas consumption and limit problems
- **Denial of Service**: Contract failure and gas griefing attacks
- **Transaction Origin**: Authorization bypass vulnerabilities

## 🔍 Quality Assurance

### Data Quality Metrics
- **Completeness**: Missing value detection and handling
- **Consistency**: Data type validation and format standardization
- **Accuracy**: Label validation and mapping verification
- **Reproducibility**: Version control and checksum validation

### Validation Reports
- **Comprehensive Reports**: JSON-formatted validation results
- **Quality Metrics**: Statistical analysis of dataset quality
- **Recommendations**: Actionable insights for data improvement
- **Reproducibility**: Version tracking and integrity verification

## 🎯 Next Steps

1. **Model Development**: Use processed datasets for ML model training
2. **Feature Engineering**: Enhance feature extraction based on analysis
3. **Data Augmentation**: Expand dataset with additional sources
4. **Continuous Integration**: Automated pipeline execution and validation
5. **Performance Monitoring**: Track dataset quality over time

## 📋 Dependencies

- **Python 3.8+**: Core runtime environment
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities
- **Requests**: HTTP client for API calls
- **PyYAML**: Configuration file parsing
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter**: Interactive notebook environment

## 🏆 Achievements

✅ **Complete Pipeline**: End-to-end data collection and preprocessing  
✅ **Multi-source Integration**: SmartBugs, GitHub, Etherscan  
✅ **Comprehensive Validation**: Data integrity and quality assurance  
✅ **Reproducible Processing**: Version control and consistency checks  
✅ **Extensive Testing**: Unit tests and validation framework  
✅ **Documentation**: Complete documentation and examples  
✅ **Quality Assurance**: Automated validation and reporting  

The project successfully delivers a robust, scalable, and maintainable dataset collection and preprocessing pipeline for smart contract vulnerability detection.

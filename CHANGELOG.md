# Changelog

All notable changes to the Chemical ML Platform project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-01-26

### Added
- **Test Coverage**: New comprehensive test suites for previously untested modules
  - `test_automl_active_learning.py`: AutoML (Optuna) and Active Learning strategies
  - `test_pharmacophore.py`: 3D Pharmacophore generation and similarity
  - `test_smart_preprocessor.py`: Smart preprocessing pipeline
  - `test_uma.py`: UMAP feature extraction
  - `test_mixture_solubility.py`: Mixture engineering and ESOL solubility prediction
- **Documentation**: Codebase audit report (`audit_report.md`)
- **Quality Assurance**: All 350+ tests now passing with improved coverage

### Changed
- Improved mock strategies for runtime imports (RDKit, Optuna)
- Enhanced error handling in test fixtures

### Fixed
- Resolved test brittleness in column naming assertions
- Fixed MagicMock pickling issues in persistence tests

## [0.3.0] - 2026-01-22

### Added
- **LLM Assistant**: Lightweight AI assistant for analysis guidance
  - Django REST API endpoints (`/api/llm/ask/`, `/api/llm/clear-history/`)
  - Streamlit UI integration with chat interface
  - Context-aware suggestions for ML pipeline and feature engineering
- **Examples & References**: Comprehensive usage examples
  - Feature extraction examples (01-05)
  - Complete API reference documentation
- **Developer Tools**: 
  - Sample plugin architecture
  - CI/CD pipeline configuration
  - Developer documentation for extensibility

### Changed
- Applied `black` code formatting to entire codebase for consistency
- Merged LLM Assistant UI and API endpoints into main branch

### Fixed
- Critical bug fixes in test suite
- Improved test stability and reliability

## [0.2.0] - 2026-01-21

### Added
- **ML Modules**: Extended sklearn wrappers
  - `sklearn_modules/cluster_extended.py`: Advanced clustering
  - `sklearn_modules/preprocessing_extended.py`: Custom preprocessing
  - `sklearn_modules/neural_network.py`: Neural network wrappers
  - `sklearn_modules/xgboost_extended.py`: XGBoost utilities
- **Feature Engineering**: 
  - Mixture engineering (`mixture_eng.py`)
  - Solubility prediction (`solubility.py`)
  - UMAP feature extraction (`uma_eng.py`)
- **Pipeline Components**:
  - Smart preprocessor (`preprocessor.py`)
  - Feature selector (`feature_selector.py`)
  - Dimensionality reduction (`dimensionality_reduction.py`)
  - Model factory (`model_factory.py`)
  - Pipeline builder (`pipeline_builder.py`)
- **Testing Infrastructure**: 
  - 30+ new test modules covering core functionality
  - Batch processing tests
  - Data quality tests
  - Model persistence tests

### Changed
- Refactored ML pipeline with modular architecture
- Enhanced data quality assessment tools

## [0.1.0] - 2026-01-20

### Added
- **Initial Release**: Chemical ML Platform foundation
- **Core Features**:
  - Django-based web framework
  - Streamlit frontend for interactive analysis
  - RDKit integration for molecular feature extraction
  - Computational chemistry engine (XTB support)
  - Reaction engineering module
  - Basic ML pipeline with LightGBM/Random Forest
  - Applicability domain assessment
  - Uncertainty quantification
- **Data Management**:
  - Dataset upload and validation
  - Experiment tracking
  - Model persistence
- **Visualization**:
  - SHAP explanations
  - PDP (Partial Dependence Plots)
  - Training curves
- **API**: RESTful endpoints for core operations
- **Security**: Basic security setup and configurations

### Infrastructure
- Django 5.1.4
- PostgreSQL/SQLite support
- Celery task queue
- Redis caching
- Docker deployment ready

---

## Version Numbering

- **Major version** (X.0.0): Breaking changes or major architectural shifts
- **Minor version** (0.X.0): New features, backward-compatible
- **Patch version** (0.0.X): Bug fixes, minor improvements

## Links

- [Repository](https://github.com/jckkvs/chem_app)
- [Issue Tracker](https://github.com/jckkvs/chem_app/issues)

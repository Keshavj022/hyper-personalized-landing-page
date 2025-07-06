# Hyper-Personalized Landing Page Generator

An AI-powered prototype that dynamically generates personalized landing pages for eCommerce users, with intelligent cold start problem solutions and real-time content optimization.

## 🎯 Problem Statement

Digital commerce platforms struggle to personalize experiences for users with little to no history. This project solves the cold start problem by creating intelligent fallback strategies and dynamic content generation for first-time and anonymous visitors.

## 🚀 Key Features

- **User Segmentation**: K-Means clustering identifying 6 distinct user types
- **Cold Start Solution**: Random Forest classifier with 89% accuracy for new users
- **Real-time Personalization**: <200ms response time for content generation

## 📁 Project Structure

```
hyper-personalized-landing-page/
├── Data/                          # Dataset directory (place datasets here)
│   ├── dataset1_final.csv        # User activity data
│   └── dataset2_final.csv        # Transaction data
├── images/
|   ├── Dataset-Analysis.png          # Comprehensive data analysis dashboard
|   ├── User-Segments.png            # User segmentation clustering results
|   ├── User-Journey.png             # User behavior flow analysis
|   ├── Conversion-Patterns.png      # Conversion funnel and patterns
|   ├── Product-trends.png           # Product category and sales trends
|   ├── feature-importance.png       # ML model feature importance
|   ├── Landing-Page-Layout.png      # Personalized landing page examples
|   └── performance-report.png       # System performance metrics
├── presentation/                  
│   ├── KeshavJoshi.pptx          # Presentation file
│   └── ml-pipeline-diagram.png   # Architecture diagram
├── main.py                       # Main Execution File
├── requirements.txt              # Python dependencies
└── README.md                    # This file
```

## 🛠️ Environment Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Keshavj022/hyper-personalized-landing-page.git
   cd hyper-personalized-landing-page
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare dataset**
   - Place the provided datasets in the `Data/` directory:
     - `dataset1_final.csv` (User Activity Data)
     - `dataset2_final.csv` (Transaction Data)

## 🚀 Quick Start

### Script Running
```bash
python main.py
```

**View presentation**
   - Open `presentation/KeshavJoshi.pptx`

## 📊 Dataset Requirements

The system expects two CSV files in the `Data/` directory:

### User Activity Data (`dataset1_final.csv`)
- **Size**: ~6.5M records
- **Key Columns**: user_pseudo_id, event_name, category, gender, Age, country, source, purchase_revenue, transaction_id
- **Format**: GA4-style event tracking data

### Transaction Data (`dataset2_final.csv`)
- **Size**: ~27.5K records  
- **Key Columns**: Transaction_ID, ItemName, ItemBrand, ItemCategory, Item_revenue, Item_purchase_quantity
- **Format**: eCommerce transaction records

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt --upgrade
   ```

2. **Dataset Not Found**
   ```bash
   # Verify dataset location
   ls Data/
   # Should show: dataset1_final.csv, dataset2_final.csv
   ```

## 🔬 Technical Approach

### 1. Data Processing Pipeline
- Multi-stage ETL with quality validation
- Feature engineering for behavioral metrics
- Session construction and user journey mapping

### 2. Machine Learning Models
- **K-Means Clustering**: User segmentation (6 segments identified)
- **Random Forest**: Cold start prediction (89% accuracy)
- **Feature Engineering**: 15+ behavioral and demographic features

### 3. Personalization Engine
- Rule-based content selection
- Dynamic layout generation
- Real-time A/B testing framework

### 4. Cold Start Solutions
- Demographic inference patterns
- Geographic trend analysis
- Device behavior optimization
- Traffic source intelligence

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset provided by the organizing team
- Inspiration from modern eCommerce personalization challenges
- Built for the Hyper-Personalized Landing Page Generator hackathon

## 📞 Contact

For questions or support, please contact:
- **Project Lead**: [Keshav Joshi]
- **Email**: [joshikeshav2204@gmail.com]
- **GitHub**: [https://github.com/Keshavj022]

---

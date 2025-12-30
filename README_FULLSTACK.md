# 🛰️ Ghost Hunter - Full Stack Maritime Intelligence Platform

A comprehensive maritime vessel detection and intelligence analysis system combining satellite imagery processing, machine learning, and GenAI-powered intelligence analysis.

## 🌟 Features

### 🔍 **Advanced Vessel Detection**
- Sentinel-1 SAR satellite imagery processing
- SBCI (Ship-to-Background Contrast Index) detection algorithm
- CNN-based vessel verification and size estimation
- AIS cross-referencing for dark vessel identification

### 🧠 **GenAI Intelligence Analysis**
- Google Gemini-powered intelligence analysis
- Automated threat assessment and risk scoring
- Behavioral pattern analysis
- Human-readable intelligence reports

### 🌐 **Full Stack Web Application**
- React/Next.js frontend with modern UI
- Flask REST API backend
- Real-time analysis progress tracking
- Interactive vessel visualization

### 📊 **Comprehensive Reporting**
- Executive summaries and threat assessments
- Detailed vessel intelligence profiles
- Email report delivery
- Multiple export formats (JSON, Markdown, PDF)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend│    │   Flask Backend │    │  GenAI Pipeline │
│                 │    │                 │    │                 │
│ • Next.js       │◄──►│ • REST API      │◄──►│ • Gemini AI     │
│ • TypeScript    │    │ • SQLite DB     │    │ • Intelligence  │
│ • Tailwind CSS  │    │ • File Upload   │    │ • Analysis      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Detection Engine│
                       │                 │
                       │ • SAR Processing│
                       │ • CNN Models    │
                       │ • Risk Fusion   │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- Google API key for Gemini AI

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd ghost-hunter

# Install Python dependencies
pip install -r requirements_backend.txt

# Setup environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 2. Start the Backend

```bash
# Option 1: Use the launcher script (recommended)
python run_ghost_hunter.py

# Option 2: Start manually
python app.py
```

The backend will start on `http://localhost:5000`

### 3. Start the Frontend

```bash
# In a new terminal
cd ghost-hunter-frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

The frontend will start on `http://localhost:3000`

### 4. Access the Application

Open your browser to `http://localhost:3000` and start analyzing maritime data!

## 📁 Project Structure

```
ghost-hunter/
├── 🔧 Backend (Flask API)
│   ├── app.py                          # Main Flask application
│   ├── config.py                       # Configuration settings
│   ├── run_ghost_hunter.py            # Application launcher
│   ├── requirements_backend.txt        # Python dependencies
│   └── utils/
│       └── email_service.py           # Email functionality
│
├── 🎨 Frontend (React/Next.js)
│   ├── ghost-hunter-frontend/
│   │   ├── app/                       # Next.js app router
│   │   ├── components/                # React components
│   │   ├── lib/                       # Utilities and API client
│   │   └── package.json               # Node.js dependencies
│
├── 🧠 AI/ML Pipeline
│   ├── enhanced_ghost_hunter_pipeline.py  # Enhanced pipeline with GenAI
│   ├── intelligence_analyzer.py           # GenAI intelligence engine
│   ├── main_pipeline.py                   # Core detection pipeline
│   └── intelligence_prompts.py            # AI prompts and templates
│
├── 📊 Data & Models
│   ├── data/                          # Input data directory
│   ├── output/                        # Pipeline outputs
│   ├── results/                       # Analysis results
│   └── sar_cnn_model.pth             # Trained CNN model
│
└── 📚 Documentation
    ├── GENAI_INTELLIGENCE_GUIDE.md   # GenAI integration guide
    ├── GENAI_IMPLEMENTATION_SUMMARY.md
    └── README_FULLSTACK.md           # This file
```

## 🔌 API Endpoints

### Analysis Management
- `POST /api/analysis/start` - Start new analysis
- `GET /api/analysis/{id}/status` - Get analysis status
- `GET /api/analysis/{id}/results` - Get analysis results

### Vessel Intelligence
- `GET /api/vessel/{id}/intelligence` - Get vessel intelligence
- `GET /api/analysis/{id}/report` - Download reports
- `POST /api/analysis/{id}/send-report` - Email reports

### Data Management
- `GET /api/mpas` - Get available MPAs
- `POST /api/upload/satellite` - Upload satellite data
- `GET /api/analyses` - Get analysis history

### System
- `GET /api/health` - Health check

## ⚙️ Configuration

### Backend Configuration (.env)
```bash
# Google API Configuration
GOOGLE_API_KEY=your_google_api_key_here
GENAI_MODEL=gemini-2.5-flash
GENAI_TEMPERATURE=0.3

# Intelligence Analysis Settings
MAX_VESSELS_DETAILED_ANALYSIS=10
ANALYSIS_CONFIDENCE_THRESHOLD=0.7

# Email Configuration (optional)
SMTP_SERVER=smtp.gmail.com
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

### Frontend Configuration (.env.local)
```bash
# Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:5000/api

# Application settings
NEXT_PUBLIC_APP_NAME=Ghost Hunter
NEXT_PUBLIC_ENABLE_MOCK_DATA=false
```

## 🧪 Usage Examples

### 1. Start Analysis via API

```bash
curl -X POST http://localhost:5000/api/analysis/start \
  -H "Content-Type: application/json" \
  -d '{
    "region_type": "custom",
    "region_data": {"polygon": [[lat1,lng1], [lat2,lng2], ...]},
    "start_date": "2024-01-01",
    "end_date": "2024-01-07"
  }'
```

### 2. Check Analysis Status

```bash
curl http://localhost:5000/api/analysis/{analysis_id}/status
```

### 3. Get Results

```bash
curl http://localhost:5000/api/analysis/{analysis_id}/results
```

## 🔧 Development

### Backend Development

```bash
# Install development dependencies
pip install -r requirements_backend.txt

# Run with debug mode
export FLASK_ENV=development
python app.py

# Run tests (if available)
python -m pytest tests/
```

### Frontend Development

```bash
cd ghost-hunter-frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run linting
npm run lint
```

## 🐳 Docker Deployment (Optional)

### Backend Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_backend.txt .
RUN pip install -r requirements_backend.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

### Frontend Dockerfile
```dockerfile
FROM node:18-alpine

WORKDIR /app
COPY ghost-hunter-frontend/package*.json ./
RUN npm install

COPY ghost-hunter-frontend/ .
RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]
```

## 🔒 Security Considerations

### API Security
- CORS configured for frontend domain
- Input validation on all endpoints
- File upload restrictions and validation
- SQL injection prevention with parameterized queries

### Data Security
- Sensitive intelligence data handling
- Secure email transmission
- Environment variable protection
- Database access controls

## 📈 Performance Optimization

### Backend Optimization
- Threaded analysis processing
- Database connection pooling
- File caching strategies
- API response compression

### Frontend Optimization
- Next.js automatic code splitting
- Image optimization
- Static asset caching
- Progressive loading

## 🐛 Troubleshooting

### Common Issues

1. **Backend won't start**
   - Check Python version (3.8+ required)
   - Verify all dependencies installed
   - Check .env file configuration

2. **GenAI analysis fails**
   - Verify GOOGLE_API_KEY is set correctly
   - Check API quota limits
   - Review error logs for specific issues

3. **Frontend can't connect to backend**
   - Ensure backend is running on port 5000
   - Check NEXT_PUBLIC_API_URL in .env.local
   - Verify CORS configuration

4. **Analysis gets stuck**
   - Check satellite data availability
   - Review pipeline logs
   - Verify model files are present

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python app.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Sentinel-1 SAR data from ESA Copernicus program
- Google Gemini AI for intelligence analysis
- Open source libraries and frameworks used

## 📞 Support

For questions, issues, or support:
- Check the troubleshooting section above
- Review the API documentation
- Check existing GitHub issues
- Create a new issue with detailed information

---

**🛰️ Ghost Hunter - Protecting our oceans through advanced maritime intelligence**
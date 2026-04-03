# 🆓 Local RAG Application for SMEs

A **completely free**, Python-based Retrieval-Augmented Generation (RAG) application that runs 100% locally on your machine. Perfect for Small and Medium Enterprises (SMEs) who want powerful document Q&A without any cloud costs.

## ✨ Key Features

- **💯 Completely Free**: No API costs, no cloud fees, runs entirely on your local machine
- **📄 Smart Document Processing**: Handles PDFs, Excel files, emails, and images
- **🧠 Intelligent Q&A**: Uses local Ollama LLM for reasoning and conflict resolution
- **💾 Local Storage**: ChromaDB vector database stored locally
- **🔍 Conflict Detection**: Automatically identifies contradictory information
- **🎨 Clean UI**: Streamlit web interface for easy interaction
- **📧 Support Ticket Generation**: Built-in CRM simulation

## 🚀 Quick Start (Local)

### 1. Install Ollama (Free)
```bash
# Download from: https://ollama.ai/download
# After installation, start the service
ollama serve
```

### 2. Pull a Model (Free)
```bash
# Pull a good model for reasoning
ollama pull llama3.2
# Alternative: ollama pull glm4
```

### 3. Setup Python Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app.py
```

## 🌐 Online Deployment (Free APIs)

For sharing your app online without local setup, deploy using free cloud services. **Note:** Data uploaded during sessions won't persist due to local ChromaDB limitations.

### 1. Get Free API Key

**Groq API (LLM)** - Free tier with generous limits:
- Sign up at: https://console.groq.com/
- Get your API key from the dashboard

### 2. Set Environment Variables

Create a `.env` file or set in your deployment platform:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Deploy on Streamlit Cloud

**Step 1: Push to GitHub**
```bash
# Create a new repository on GitHub
# Push your code
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/your-repo.git
git push -u origin main
```

**Step 2: Deploy on Streamlit Cloud**
- Go to: https://share.streamlit.io/
- Connect your GitHub account
- Select your repository
- Set main file path: `app.py`
- Add secrets in advanced settings:
  - `GROQ_API_KEY`: your Groq API key
- Click Deploy!

**Your app will be live at:** `https://your-repo-name.streamlit.app/`

**Limitations:** Uploaded documents won't persist between sessions since ChromaDB runs locally in the cloud container.

### Alternative Free Deployments

**Railway** (Free tier):
- Sign up at: https://railway.app/
- Connect GitHub repo
- Set environment variables
- Deploy

**Render** (Free tier):
- Sign up at: https://render.com/
- Create Web Service from Git repo
- Set build command: `pip install -r requirements.txt`
- Set start command: `streamlit run app.py --server.port $PORT --server.headless true`
- Add environment variables

## 📋 What You Can Ask

### Product Questions
- "what are the product pricing details"
- "how much does Widget A cost"
- "what products do you have"

### Policy Questions
- "what is the refund policy"
- "can I return custom products"
- "what are the terms and conditions"

### Email Questions
- "who sent the email"
- "what is the email subject"
- "what does the email say"

## 🏗️ How It Works

1. **Document Ingestion**: Files are processed and converted to text chunks
2. **Vector Storage**: Chunks are stored in local ChromaDB with embeddings
3. **Query Processing**: Your questions are converted to embeddings for similarity search
4. **Context Retrieval**: Most relevant document chunks are retrieved
5. **Local Reasoning**: Ollama analyzes context and generates answers
6. **Conflict Resolution**: System detects and handles contradictory information

## 📁 Project Structure

```
├── app.py                    # Main Streamlit application
├── reasoning.py              # Ollama integration & conflict resolution
├── ingestion.py              # Document processing (PDF, Excel, Email, Images)
├── storage.py                # ChromaDB setup and chunking
├── retrieval.py              # Vector search and retrieval
├── requirements.txt          # Python dependencies
├── test_sample.py           # Test interface for document processing
├── start_all.bat            # Windows batch script to start everything
├── stop_all.bat             # Windows batch script to stop services
└── chroma_db/               # Local vector database (created automatically)
```

## 🧪 Testing

### Test Document Processing
```bash
streamlit run test_sample.py
```

### Validate Retrieval
```bash
python validate_retrieval.py
```

## 🔧 Troubleshooting

### Ollama Issues
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve

# Pull a model if needed
ollama pull llama3.2
```

### Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt

# Specific package
pip install liteparse
```

### Database Issues
```bash
# Clear ChromaDB data
# Delete the chroma_db/ folder
# Restart the app to recreate
```

## 🎯 Advanced Features

- **Semantic Chunking**: Smart document splitting by paragraphs
- **Metadata Tracking**: Source attribution and timestamps
- **Fallback Answers**: Works even if Ollama is unavailable
- **Batch Processing**: Upload multiple files simultaneously
- **Real-time Feedback**: See ingestion progress

## 💰 Cost Comparison

| Service | Cost | Your Solution (Local) | Your Solution (Online) |
|---------|------|----------------------|----------------------|
| OpenAI API | $0.002/1K tokens | **$0** (Ollama) | **$0** (Groq free tier) |
| Pinecone | $0.10/GB/month | **$0** (ChromaDB local) | **$0** (ChromaDB local) |
| Streamlit Cloud | $10/month | **$0** (local Streamlit) | **$0** (free public apps) |
| **Total** | **~$10-50/month** | **$0** 🎉 | **$0** 🎉 |

## 🤝 Contributing

This is a completely free, open-source solution. Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Help other SMEs discover this solution

## 📄 License

MIT License - use freely for your business!

---

**Ready to get started?** Just run `ollama serve`, `ollama pull llama3.2`, and `streamlit run app.py`! 🚀

## Troubleshooting

- If LiteParse fails, it falls back to PyPDF2 for PDFs.
- Ensure Ollama is running for reasoning.
- ChromaDB data is in `./chroma_db`.
# SME

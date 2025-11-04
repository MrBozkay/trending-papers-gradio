# 📚 Trending Papers - Research Paper Discovery

A modern, cloud-deployed Gradio interface for discovering trending research papers and their code implementations. Built for HuggingFace Spaces with a responsive, user-friendly design.

## 🌟 Features

### 🔍 Paper Search & Discovery
- **Smart Search**: Search papers by keywords, authors, or topics
- **Category Filtering**: Filter by ArXiv categories (AI, ML, CV, NLP, etc.)
- **Citation Analysis**: Filter papers by minimum citation count
- **Trending Papers**: Discover the most popular research papers

### 📊 Research Analytics
- **Paper Metrics**: View citation counts, trending scores, and categories
- **Visual Cards**: Interactive paper cards with hover effects
- **Direct Links**: One-click access to ArXiv papers and PDFs
- **Repository Links**: Find related code implementations

### 🎨 Modern Interface
- **Responsive Design**: Works on desktop and mobile devices
- **Beautiful Gradients**: Modern gradient backgrounds
- **Interactive UI**: Smooth animations and hover effects
- **Export Functionality**: JSON/CSV export for further analysis

## 🚀 Live Demo

**🔗 [Try it now on HuggingFace Spaces](https://mrb0zkay-trending-papers.hf.space)**

## 🛠️ Technology Stack

- **Frontend**: Gradio 4.0+ (Python)
- **Styling**: Custom CSS with gradients and animations
- **Deployment**: HuggingFace Spaces
- **Data Source**: Mock data (easily extensible to real APIs)
- **Architecture**: Modular, scalable design

## 📖 Usage

### Search Papers
1. Enter keywords in the search box (e.g., "transformer", "machine learning")
2. Select relevant categories from the dropdown
3. Adjust filters (min citations, max results, similarity threshold)
4. Click "Search Papers" to see results

### View Trending Papers
- The trending papers section loads automatically on page load
- Click "Refresh Trending" to get the latest trending papers
- View paper details, citations, and direct links

### Advanced Search
- Use boolean operators for complex queries
- Apply date range filters
- Fine-tune search parameters for precise results

## 🔧 Local Development

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
# Clone the repository
git clone https://github.com/MrBozkay/trending-papers-gradio.git
cd trending-papers-gradio

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

Access the application at `http://localhost:7860`

## 📝 Notes

### Current Status
- ✅ **Mock Data**: Currently using high-quality mock data
- ✅ **Public Access**: Fully deployed and accessible
- ✅ **Responsive Design**: Works on all device sizes
- ✅ **Search Functionality**: Advanced search with filters

### Development Roadmap
- 🔄 **API Integration**: Connect to real research APIs
- 🔄 **User Authentication**: Personal accounts and saved searches
- 🔄 **Advanced Analytics**: Citation networks and trend analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Guidelines
1. Follow Python PEP 8 style guidelines
2. Add proper docstrings to all functions
3. Test thoroughly before submitting
4. Update this README for new features

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **HuggingFace**: For the excellent Spaces platform
- **Gradio**: For the amazing ML UI framework
- **ArXiv**: For the research paper database
- **Research Community**: For the incredible papers that inspire this tool

---

**Made with ❤️ for the research community**

For questions or support, please open an issue on GitHub.
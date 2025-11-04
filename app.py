import gradio as gr
import requests
import json
import time
import os
from typing import List, Dict, Any, Optional
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendingPapersInterface:
    """Trending Papers Gradio Interface for HuggingFace Spaces"""
    
    def __init__(self):
        # Environment variable'dan backend URL al, yoksa mock data kullan
        self.api_base_url = os.getenv("BACKEND_API_URL", "https://api.example.com")
        self.use_mock_data = os.getenv("USE_MOCK_DATA", "true").lower() == "true"
        
        self.state = {
            "search_history": [],
            "current_results": [],
            "dark_mode": False,
            "loading_states": {}
        }
        
        # CSS stilleri
        self.custom_css = self.get_custom_css()
        
    def get_custom_css(self) -> str:
        """Custom CSS stilleri"""
        return """
        <style>
        .paper-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid #1e3a8a;
        }
        
        .paper-card:hover {
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
            transform: translateY(-2px);
            transition: all 0.3s ease;
        }
        
        .repository-card {
            background: #f8fafc;
            border-radius: 8px;
            padding: 15px;
            margin: 8px 0;
            border: 1px solid #e2e8f0;
        }
        
        .trending-score {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 8px;
            border-radius: 16px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .header {
            background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .search-section {
            background: #f8fafc;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            border: 2px solid #e2e8f0;
        }
        
        .metric-box {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            margin: 5px;
        }
        
        .error-message {
            background: #fef2f2;
            color: #dc2626;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #fecaca;
            margin: 10px 0;
        }
        
        .category-tag {
            background: #dbeafe;
            color: #1e40af;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            margin: 2px;
            display: inline-block;
        }
        
        .export-section {
            background: #fefce8;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            border: 1px solid #fde68a;
        }
        </style>
        """
    
    def create_paper_card_html(self, paper: Dict) -> str:
        """Paper kartı HTML'i oluşturur"""
        title = paper.get('title', 'Unknown Title')
        authors = paper.get('authors', [])
        abstract = paper.get('abstract', 'No abstract available')
        arxiv_id = paper.get('arxiv_id', '')
        citations = paper.get('citations', 0)
        categories = paper.get('categories', [])
        trends_score = paper.get('trends_score', 0)
        pdf_url = paper.get('pdf_url', '')
        html_url = paper.get('html_url', '')
        repositories = paper.get('repositories', [])
        
        # Abstract'ı kısalt
        short_abstract = abstract[:200] + "..." if len(abstract) > 200 else abstract
        
        # Kategori etiketleri
        category_tags = ''.join([f'<span class="category-tag">{cat}</span>' for cat in categories[:3]])
        
        # Repository kartları
        repo_cards = ''
        if repositories:
            repo_cards = '<div class="repositories-section">'
            repo_cards += '<h5>Related Repositories:</h5>'
            for repo in repositories[:3]:
                repo_cards += f'''
                <div class="repository-card">
                    <strong>{repo.get('name', '')}</strong>
                    <p>{repo.get('description', '')[:100]}...</p>
                    <small>⭐ {repo.get('stars', 0)} | 🍴 {repo.get('forks', 0)} | 💻 {repo.get('language', '')}</small>
                </div>
                '''
            repo_cards += '</div>'
        
        # Trending score
        trending_badge = f'<span class="trending-score">Trend: {trends_score:.1f}</span>' if trends_score > 0.7 else ''
        
        card_html = f'''
        <div class="paper-card" data-paper-id="{paper.get('id', '')}">
            <div class="paper-header">
                <h3>{title}</h3>
                <div class="paper-meta">
                    {trending_badge}
                    <span class="citations">📊 {citations} citations</span>
                </div>
            </div>
            
            <div class="paper-authors">
                <strong>Authors:</strong> {', '.join(authors[:3])}
                {f' and {len(authors)-3} more...' if len(authors) > 3 else ''}
            </div>
            
            <div class="paper-abstract">
                <p>{short_abstract}</p>
            </div>
            
            <div class="paper-categories">
                {category_tags}
            </div>
            
            <div class="paper-links">
                <a href="{html_url}" target="_blank" style="color: #1e3a8a; text-decoration: none; font-weight: 500;">📄 View on ArXiv</a>
                <a href="{pdf_url}" target="_blank" style="color: #1e3a8a; text-decoration: none; font-weight: 500;">📑 PDF</a>
                <span style="color: #666; font-size: 0.9em;">arXiv:{arxiv_id}</span>
            </div>
            
            {repo_cards}
        </div>
        '''
        
        return card_html
    
    def get_mock_trending_papers(self, limit: int = 20) -> List[Dict]:
        """Mock trending papers data"""
        mock_papers = [
            {
                "id": "1",
                "title": "Attention Is All You Need: A Comprehensive Survey",
                "authors": ["Vaswani, A.", "Shazeer, N.", "Parmar, N."],
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms...",
                "arxiv_id": "1706.03762",
                "citations": 89456,
                "categories": ["cs.CL", "cs.LG"],
                "trends_score": 9.8,
                "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
                "html_url": "https://arxiv.org/abs/1706.03762",
                "repositories": [
                    {
                        "name": "transformers",
                        "description": "🤗 Transformers: State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX",
                        "stars": 132000,
                        "forks": 26000,
                        "language": "Python",
                        "url": "https://github.com/huggingface/transformers"
                    }
                ]
            },
            {
                "id": "2",
                "title": "GPT-3: Language Models are Few-Shot Learners",
                "authors": ["Brown, T.", "Mann, B.", "Ryder, N."],
                "abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text either with self-supervised learning or supervised learning. However, it is still unclear whether the benefits of pre-training and fine-tuning are primarily due to the larger size of the models, the diversity of the pre-training dataset, or simply due to the fact that these models are more compute-intensive...",
                "arxiv_id": "2005.14165",
                "citations": 45678,
                "categories": ["cs.CL", "cs.LG"],
                "trends_score": 9.5,
                "pdf_url": "https://arxiv.org/pdf/2005.14165.pdf",
                "html_url": "https://arxiv.org/abs/2005.14165",
                "repositories": [
                    {
                        "name": "gpt-2",
                        "description": "Code and models from the paper 'Language Models are Unsupervised Multitask Learners'",
                        "stars": 42000,
                        "forks": 11000,
                        "language": "Python",
                        "url": "https://github.com/openai/gpt-2"
                    }
                ]
            },
            {
                "id": "3",
                "title": "Deep Residual Learning for Image Recognition",
                "authors": ["He, K.", "Zhang, X.", "Ren, S.", "Sun, J."],
                "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions...",
                "arxiv_id": "1512.03385",
                "citations": 123456,
                "categories": ["cs.CV"],
                "trends_score": 9.2,
                "pdf_url": "https://arxiv.org/pdf/1512.03385.pdf",
                "html_url": "https://arxiv.org/abs/1512.03385",
                "repositories": [
                    {
                        "name": "pytorch-image-models",
                        "description": "PyTorch image models, scripts, and pretrained weights",
                        "stars": 74000,
                        "forks": 16000,
                        "language": "Python",
                        "url": "https://github.com/rwightman/pytorch-image-models"
                    }
                ]
            },
            {
                "id": "4",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "authors": ["Devlin, J.", "Chang, M.W.", "Lee, K.", "Toutanova, K."],
                "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers...",
                "arxiv_id": "1810.04805",
                "citations": 87654,
                "categories": ["cs.CL"],
                "trends_score": 9.0,
                "pdf_url": "https://arxiv.org/pdf/1810.04805.pdf",
                "html_url": "https://arxiv.org/abs/1810.04805",
                "repositories": [
                    {
                        "name": "bert",
                        "description": "TensorFlow code and pre-trained models for BERT",
                        "stars": 35000,
                        "forks": 12000,
                        "language": "Python",
                        "url": "https://github.com/google-research/bert"
                    }
                ]
            },
            {
                "id": "5",
                "title": "Vision Transformer: An Image is Worth 16x16 Words",
                "authors": ["Dosovitskiy, A.", "Beyer, L.", "Kolesnikov, A."],
                "abstract": "While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure...",
                "arxiv_id": "2010.11929",
                "citations": 34567,
                "categories": ["cs.CV"],
                "trends_score": 8.8,
                "pdf_url": "https://arxiv.org/pdf/2010.11929.pdf",
                "html_url": "https://arxiv.org/abs/2010.11929",
                "repositories": [
                    {
                        "name": "vision-transformer",
                        "description": "Vision Transformer (ViT) in PyTorch",
                        "stars": 18000,
                        "forks": 5000,
                        "language": "Python",
                        "url": "https://github.com/lucidrains/vit-pytorch"
                    }
                ]
            }
        ]
        
        return mock_papers[:limit]
    
    def get_trending_papers(self, limit: int = 20):
        """Trending paper'ları getirir"""
        try:
            # Mock data kullan
            trending_papers = self.get_mock_trending_papers(limit)
            
            # Create results cards
            cards = ''
            for paper in trending_papers:
                cards += self.create_paper_card_html(paper)
            
            # Results header
            results_header = f'''
            <div class="search-results-header">
                <h2>🔥 Trending Papers</h2>
                <p>Top {len(trending_papers)} trending research papers</p>
            </div>
            '''
            
            # İstatistikler
            total_papers = len(trending_papers)
            avg_citations = sum(p.get('citations', 0) for p in trending_papers) / max(total_papers, 1)
            
            stats = f'''
            <div style="display: flex; justify-content: space-around; margin: 20px 0;">
                <div class="metric-box">
                    <h3>{total_papers}</h3>
                    <p>Trending Papers</p>
                </div>
                <div class="metric-box">
                    <h3>{avg_citations:.0f}</h3>
                    <p>Avg Citations</p>
                </div>
                <div class="metric-box">
                    <h3>{len(set([cat for p in trending_papers for cat in p.get('categories', [])]))}</h3>
                    <p>Categories</p>
                </div>
            </div>
            '''
            
            return results_header + stats + cards
            
        except Exception as e:
            error_html = f'<div class="error-message">Error loading trending papers: {str(e)}</div>'
            return error_html
    
    def search_papers(self, query: str, categories: List[str], max_results: int, 
                     min_citations: int, similarity_threshold: float):
        """Paper arama işlemi"""
        if not query or len(query.strip()) < 2:
            return '<div class="error-message">Please enter a search query with at least 2 characters.</div>'
        
        try:
            # Mock search results based on query
            mock_papers = self.get_mock_trending_papers(max_results)
            
            # Filter papers based on query
            filtered_papers = []
            query_lower = query.lower()
            
            for paper in mock_papers:
                title_match = query_lower in paper.get('title', '').lower()
                abstract_match = query_lower in paper.get('abstract', '').lower()
                author_match = any(query_lower in author.lower() for author in paper.get('authors', []))
                category_match = any(query_lower in cat.lower() for cat in paper.get('categories', []))
                
                if title_match or abstract_match or author_match or category_match:
                    filtered_papers.append(paper)
            
            # Apply filters
            if categories:
                filtered_papers = [p for p in filtered_papers if any(cat in p.get('categories', []) for cat in categories)]
            
            if min_citations > 0:
                filtered_papers = [p for p in filtered_papers if p.get('citations', 0) >= min_citations]
            
            if not filtered_papers:
                return f'<div class="paper-card">No papers found for "{query}". Try different keywords or filters.</div>'
            
            # Create results cards
            cards = ''
            for paper in filtered_papers[:max_results]:
                cards += self.create_paper_card_html(paper)
            
            # Results header
            search_time = 0.5  # Mock search time
            results_header = f'''
            <div class="search-results-header">
                <h2>🔍 Search Results</h2>
                <p>Found <strong>{len(filtered_papers)}</strong> papers for "{query}" in {search_time:.2f}s</p>
                <p>Showing first {min(len(filtered_papers), max_results)} results</p>
            </div>
            '''
            
            # Export section (mock)
            export_section = f'''
            <div class="export-section">
                <h4>📤 Export Results</h4>
                <p>Export your search results for further analysis</p>
                <a href="#" onclick="alert('Export functionality coming soon!')" 
                   style="display: inline-block; background: #10b981; color: white; padding: 8px 16px; border-radius: 6px; text-decoration: none; margin: 5px;">Export JSON</a>
                <a href="#" onclick="alert('Export functionality coming soon!')" 
                   style="display: inline-block; background: #f59e0b; color: white; padding: 8px 16px; border-radius: 6px; text-decoration: none; margin: 5px;">Export CSV</a>
            </div>
            '''
            
            return results_header + cards + export_section
            
        except Exception as e:
            return f'<div class="error-message">Search error: {str(e)}</div>'

def create_interface():
    """Gradio interface'i oluşturur"""
    interface = TrendingPapersInterface()
    
    # Components
    with gr.Blocks(
        title="Trending Papers - Research Paper Discovery", 
        theme=gr.themes.Soft(),
        css=interface.custom_css,
        head="""
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
        .gradio-container { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        </style>
        """
    ) as demo:
        
        # Header
        gr.HTML('''
        <div class="header">
            <h1>📚 Trending Papers</h1>
            <p>Discover the latest research papers and their implementations</p>
            <p style="font-size: 0.9em; opacity: 0.8; margin-top: 10px;">
                🌐 Deployed on HuggingFace Spaces • Mock Data Mode
            </p>
        </div>
        ''')
        
        # Ana sekmeler
        with gr.Tab("🏠 Home"):
            with gr.Row():
                with gr.Column(scale=2):
                    # Search section
                    gr.HTML('<div class="search-section">')
                    
                    query = gr.Textbox(
                        label="Search Research Papers",
                        placeholder="Enter keywords, topics, or paper titles...",
                        lines=2
                    )
                    
                    with gr.Row():
                        categories = gr.CheckboxGroup(
                            choices=["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.CR", "cs.DB", "cs.DC", "cs.OS", "cs.SD"],
                            label="Categories",
                            value=[]
                        )
                        max_results = gr.Slider(10, 50, value=20, step=5, label="Max Results")
                    
                    with gr.Row():
                        min_citations = gr.Slider(0, 1000, value=0, step=10, label="Min Citations")
                        similarity_threshold = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Similarity Threshold")
                    
                    search_btn = gr.Button("🔍 Search Papers", variant="primary", size="lg")
                    
                    gr.HTML('</div>')
                
                with gr.Column(scale=1):
                    # Trending papers
                    trending_section = gr.HTML(interface.get_trending_papers(10))
                    refresh_trending = gr.Button("🔄 Refresh Trending", variant="secondary")
        
        with gr.Tab("🔍 Advanced Search"):
            with gr.Row():
                with gr.Column():
                    # Advanced search form
                    gr.HTML('<div class="search-section">')
                    
                    adv_query = gr.Textbox(
                        label="Advanced Search Query",
                        placeholder="Detailed search query with boolean operators...",
                        lines=3
                    )
                    
                    adv_categories = gr.CheckboxGroup(
                        choices=["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.CR", "cs.DB", "cs.DC", "cs.OS", "cs.SD"],
                        label="All Categories",
                        value=[]
                    )
                    
                    adv_max_results = gr.Slider(10, 100, value=20, step=10, label="Max Results")
                    adv_min_citations = gr.Slider(0, 2000, value=0, step=50, label="Min Citations")
                    adv_similarity = gr.Slider(0.1, 1.0, value=0.8, step=0.1, label="Similarity Threshold")
                    
                    adv_search_btn = gr.Button("🚀 Advanced Search", variant="primary")
                    
                    adv_results = gr.HTML("")
                    
                    gr.HTML('</div>')
        
        with gr.Tab("📊 Analytics"):
            gr.HTML('''
            <div style="background: white; padding: 30px; border-radius: 12px; margin: 20px;">
                <h2>📊 Research Analytics</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;">
                    <div class="metric-box">
                        <h3>Cloud</h3>
                        <p>HuggingFace Spaces</p>
                    </div>
                    <div class="metric-box">
                        <h3>Mock</h3>
                        <p>Data Source</p>
                    </div>
                    <div class="metric-box">
                        <h3>Public</h3>
                        <p>Access</p>
                    </div>
                    <div class="metric-box">
                        <h3>Modern</h3>
                        <p>Interface</p>
                    </div>
                </div>
                
                <div style="margin-top: 30px;">
                    <h3>🚀 Deployment Features</h3>
                    <div style="background: #f0f9ff; padding: 15px; border-radius: 8px; margin: 10px 0;">
                        <p><strong>🌐 Cloud Deployment:</strong> Running on HuggingFace Spaces with public access</p>
                        <p><strong>📱 Responsive Design:</strong> Works on desktop and mobile devices</p>
                        <p><strong>🔍 Search Functionality:</strong> Filter papers by categories, citations, and keywords</p>
                        <p><strong>📊 Paper Analytics:</strong> Trending papers with citation counts and categories</p>
                        <p><strong>🔗 Direct Links:</strong> Direct access to ArXiv papers and PDFs</p>
                    </div>
                </div>
            </div>
            ''')
        
        # Event handlers
        search_btn.click(
            fn=interface.search_papers,
            inputs=[query, categories, max_results, min_citations, similarity_threshold],
            outputs=[trending_section]
        )
        
        refresh_trending.click(
            fn=lambda: interface.get_trending_papers(10),
            outputs=[trending_section]
        )
        
        adv_search_btn.click(
            fn=interface.search_papers,
            inputs=[adv_query, adv_categories, adv_max_results, adv_min_citations, adv_similarity],
            outputs=[adv_results]
        )
        
        # Refresh trending on load
        demo.load(
            fn=lambda: interface.get_trending_papers(10),
            outputs=[trending_section]
        )
    
    return demo

# HuggingFace Spaces için optimize edilmiş launch
if __name__ == "__main__":
    print("🚀 Trending Papers Interface - HuggingFace Spaces")
    print("📚 Starting application...")
    
    # Gradio demo oluştur
    demo = create_interface()
    
    # HuggingFace Spaces için optimize launch
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        quiet=False,
        share=True  # HuggingFace Spaces'ta public access için
    )
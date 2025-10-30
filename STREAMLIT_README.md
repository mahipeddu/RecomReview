# Streamlit Reviewer Recommendation Web App

A beautiful and interactive web application for finding the best reviewers for your research paper using AI-powered semantic similarity.

## Features

✨ **Easy to Use Interface**
- Drag and drop PDF upload
- Real-time processing
- Interactive visualizations

📊 **Comprehensive Results**
- Top 10 (or custom number) reviewer recommendations
- Similarity scores and statistics
- Paper count per reviewer
- Downloadable CSV results

📈 **Visualizations**
- Interactive bar charts with Plotly
- Color-coded similarity scores
- Detailed statistics for top reviewers

⚙️ **Customizable**
- Adjust number of recommendations (5-20)
- Toggle detailed statistics
- Toggle visualizations
- GPU support (automatic)

## Installation

1. Install required packages:
```bash
pip install streamlit plotly
```

2. Make sure you have the pretrained embeddings cache:
```bash
# Should exist: pretrained_embedding/embeddings_cache.pt
```

## Usage

### Start the Web App

```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

### Alternative: Specify Port

```bash
streamlit run streamlit_app.py --server.port 8080
```

### Run in Background

```bash
nohup streamlit run streamlit_app.py &
```

## How to Use

1. **Upload PDF**: Click the upload button and select your research paper PDF
2. **Wait for Processing**: The app will extract text and analyze it
3. **View Results**: See the top recommended reviewers with scores
4. **Download Results**: Click the download button to get results as CSV
5. **Explore Details**: Expand sections to see detailed statistics

## Interface Overview

### Sidebar
- **About**: Information about the system
- **Model Status**: Shows if model is loaded and device (CPU/GPU)
- **Database Stats**: Number of authors and papers
- **Settings**: Customize number of recommendations and display options

### Main Area
- **Upload Section**: Drag and drop or click to upload PDF
- **Results Table**: Top recommended reviewers with scores
- **Visualization**: Interactive bar chart of similarity scores
- **Detailed Statistics**: Comprehensive stats for top reviewers

## Features Explained

### Similarity Score
- **Range**: 0.0 to 1.0
- **Higher is better**: Scores closer to 1.0 indicate better matches
- **Perfect Match (≥0.999)**: Paper might already be in the database

### Reviewer Metrics
- **Max Score**: Highest similarity score among all papers by this reviewer
- **Papers**: Number of papers by this reviewer in the database
- **Avg Score**: Average similarity across all their papers
- **Std Dev**: Score variation (lower = more consistent)

## Technical Details

### Model
- **Pretrained Model**: `all-mpnet-base-v2` from Sentence Transformers
- **Method**: Semantic similarity using cosine distance
- **Hardware**: Automatically uses GPU if available, falls back to CPU

### Processing Pipeline
1. PDF text extraction using PyPDF2
2. Text cleaning and preprocessing (NLTK)
3. Embedding generation using Sentence Transformers
4. Cosine similarity calculation against database
5. Score aggregation by author (max score)
6. Ranking and top-k selection

## Troubleshooting

### "Embeddings cache not found"
**Solution**: Generate embeddings first:
```bash
cd pretrained_embedding
python semantic_similarity_gpu.py
```

### "Failed to extract text from PDF"
**Possible causes**:
- Scanned PDF (image-based) - needs OCR
- Corrupted PDF file
- Password-protected PDF

**Solution**: Try converting the PDF or using a different file

### App is slow
**Solutions**:
- Check if GPU is being used (shown in sidebar)
- Reduce number of recommendations
- Close other applications using GPU/memory

### Port already in use
**Solution**: Use a different port:
```bash
streamlit run streamlit_app.py --server.port 8502
```

## Keyboard Shortcuts

When the app is running:
- `R` - Rerun the app
- `C` - Clear cache
- `Ctrl+C` - Stop the server (in terminal)

## Screenshots

The app includes:
- 📤 Clean upload interface
- 📊 Interactive data tables
- 📈 Beautiful visualizations with Plotly
- 🎨 Professional color scheme
- 📱 Responsive design

## Performance

- **Upload**: Near instant (< 1 second)
- **Text Extraction**: 1-2 seconds for typical papers
- **Analysis**: 2-5 seconds with GPU, 5-10 seconds with CPU
- **Total Time**: Usually under 10 seconds per paper

## Browser Compatibility

Tested and works on:
- ✅ Chrome/Chromium
- ✅ Firefox
- ✅ Safari
- ✅ Edge

## Tips

1. **Best Results**: Use text-based PDFs (not scanned images)
2. **File Size**: Works best with papers under 10 MB
3. **GPU**: App automatically uses GPU if available for faster processing
4. **Multiple Papers**: Process one at a time for best performance

## Advanced Configuration

Edit `streamlit_app.py` to customize:
- Cache paths
- Default settings
- Color schemes
- Layout options

## Support

If you encounter issues:
1. Check the terminal output for error messages
2. Verify all dependencies are installed
3. Ensure embeddings cache exists
4. Try with a different PDF file

## Updates

To update Streamlit or dependencies:
```bash
pip install --upgrade streamlit plotly
```

---

**Enjoy finding the perfect reviewers for your research papers! 🚀**

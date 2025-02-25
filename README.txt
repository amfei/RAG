LinkedIn Job Scraper & Matcher
This project scrapes job postings from LinkedIn, extracts job descriptions using Selenium, stores them in ChromaDB, and ranks them based on similarity to a candidate's CV using BM25 and Sentence Transformers.

🚀 Features
🔍 Scrapes LinkedIn Jobs for multiple roles and locations
🌐 Uses Selenium to extract job descriptions from individual job pages
🧠 AI-powered Ranking using BM25 and Sentence Transformers
💾 Stores job listings in ChromaDB for efficient retrieval
📄 Extracts text from PDF CVs and finds top job matches
📊 Hybrid Search Ranking combining keyword search and semantic similarity
🚀 Automatic job deduplication based on ranking scores
📦 Installation & Setup
1️⃣ Create Virtual Environment

python -m venv linkedin_scraper_env
source linkedin_scraper_env/bin/activate  # macOS/Linux
linkedin_scraper_env\Scripts\activate  # Windows
2️⃣ Install Dependencies

pip install -r requirements.txt
3️⃣ Run the Script

python JobSearch.py
📜 Project Structure

📁 LinkedIn Job Scraper
│── 📄 JobSearch.py              # Main script
│── 📄 requirements.txt          # Required Python dependencies
│── 📄 README.md                 # Project documentation
│── 📁 chromadb_store/           # ChromaDB storage
🔧 Dependencies
Make sure you have the following installed:

pip install os pdfplumber requests torch chromadb bs4 datetime sentence-transformers rank-bm25 selenium webdriver-manager
🛠️ Configuration
Modify these parameters in JobSearch.py before running:

job_titles = ["Data Scientist", "Machine Learning", "GenAI"]
locations = ["Toronto", "Montreal", "Boston", "San Francisco"]
num_jobs = 200
days_filter = 10
top_n = 20  # Number of top job matches to retrieve
cv_path = "your_cv.pdf"  # Path to your CV
🚀 How It Works
🔹 Step 1: Extract CV Text
The script extracts text from the candidate's PDF CV using pdfplumber.

🔹 Step 2: Scrape LinkedIn Jobs
Uses BeautifulSoup to scrape job listings.
Filters jobs by date, location, and excluded titles.
🔹 Step 3: Extract Job Descriptions
Selenium is used to navigate to each LinkedIn job page and extract job descriptions.
🔹 Step 4: Store Jobs in ChromaDB
Stores job descriptions as embeddings for fast retrieval.
🔹 Step 5: Rank Jobs using BM25 & AI
Uses BM25Okapi for keyword-based ranking.
Uses Sentence Transformers for semantic similarity.
Combines both into a hybrid ranking system.
🔹 Step 6: Retrieve Top Job Matches
Finds top job postings relevant to your CV.
Removes duplicate jobs with similar scores.
🏆 Example Output

🔹 **Top Job Matches Based on Your CV** 🔹
1. Data Scientist at Google in ✅ Toronto (Hybrid Score: 0.8765)
   🔗 https://linkedin.com/jobs/view/123456
2. Machine Learning Engineer at Amazon in ✅ New York (Hybrid Score: 0.8432)
   🔗 https://linkedin.com/jobs/view/654321


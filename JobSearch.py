import os
import pdfplumber
import requests
import torch
import chromadb
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi  # ✅ BM25 for keyword-based ranking
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import random


# ✅ Load Pretrained NLP Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ Initialize ChromaDB for Job Storage
chroma_client = chromadb.PersistentClient(path="./chromadb_store")
collection = chroma_client.get_or_create_collection(name="job_descriptions")

def extract_linkedin_job_description(job_url):
    """
    Extracts the job description from a LinkedIn job posting using Selenium.

    Parameters:
    job_url (str): The URL of the LinkedIn job posting.
    

    Returns:
    str: The extracted job description text or an error message.
    """
    
    # Configure Selenium WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Runs Chrome in headless mode (no UI)
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # Initialize the WebDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    driver.get(job_url)

    # Wait for the page to fully load
    

    time.sleep(random.random())  # Random delay to avoid detection
    

    try:
      # Locate the job description container
      #job_desc_element = driver.find_element(By.CLASS_NAME, "description__text")
      job_desc_element = driver.find_element(By.XPATH, "//section[contains(@class, 'description')]")

      job_description = job_desc_element.text

      cleaned_job_description = ' '.join(job_description.split())
      

      return cleaned_job_description if cleaned_job_description else "Job description not found."
    except Exception as e:
      print("Job description not found:", e)
      
    finally:
      
      driver.quit()  # Ensure the browser is closed after execution
      driver.service.stop()  # Force stop the WebDriver

def scrape_linkedin_jobs(job_titles, locations, num_jobs, days_filter, EXCLUDED_TITLES):
    """
    Scrapes job postings from LinkedIn for multiple job roles and locations.
    Filters jobs posted within the last N days and ensures correct job locations.
    """
    jobs = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for location in locations:
        print(location)
        for job_title in job_titles:
            print("* ",job_title)
            search_query = job_title.replace(" ", "%20")
            url = f"https://www.linkedin.com/jobs/search?keywords={search_query.replace(' ', '%20')}&location={location}"
    
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"❌ Failed to fetch jobs for {job_title} in {location}. Status Code: {response.status_code}")
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            job_listings = soup.find_all("div", class_="base-card")[:num_jobs]

            for job in job_listings:
                title_elem = job.find("h3", class_="base-search-card__title")
                company_elem = job.find("h4", class_="base-search-card__subtitle")
                location_elem = job.find("span", class_="job-search-card__location")  # ✅ Extract correct location
                link_elem = job.find("a")
                date_elem = job.find("time")
              


                if not title_elem or not company_elem or not link_elem or not date_elem or not location_elem:
                    continue  

                title = title_elem.text.strip()
                company = company_elem.text.strip()
                job_location = location_elem.text.strip()  # ✅ Extract actual job location
                link = link_elem["href"]
                posted_date_text = date_elem["datetime"]
                
            

                # ✅ Filter jobs older than N days
                posted_date = datetime.strptime(posted_date_text, "%Y-%m-%d")
                if datetime.now() - posted_date > timedelta(days=days_filter):
                    continue  # Skip old jobs

                # ✅ **Ensure title does not contain excluded words**
                if any(excluded.lower() in title.lower() for excluded in EXCLUDED_TITLES):
                    continue  # Skip senior-level jobs

                job_description = extract_linkedin_job_description(link)

                # # ✅ **Ensure title does not contain excluded words**
                # if not any(skill.lower() in job_description.lower() for skill in REQUIRED_SKILLS):
                #     continue  # Skip senior-level jobs



                # ✅ **Format job details**
                job_info = f"{title} at {company} in {job_location}. More details: {link}"

                jobs.append({
                    "title": title,
                    "company": company,
                    "location": job_location,  # ✅ Store the extracted job location
                    "link": link,
                    "info":job_info,
                    "description": job_description
                })

    print(f"✅ Scraped {len(jobs)} job postings with correct locations.")
    return jobs


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text.strip()


def store_jobs_in_chromadb(jobs):
    """
    Converts job descriptions into embeddings and stores them in ChromaDB.
    """
    if not jobs:
        print("⚠️ No jobs to store in ChromaDB.")
        return

    # ✅ **Correctly clear old jobs before adding new ones**
    try:
        collection.delete(where={"title": {"$ne": ""}})  # Deletes all jobs
        print("🗑️ Cleared old jobs from ChromaDB.")
    except ValueError as e:
        print(f"⚠️ Failed to clear old jobs: {e}")

    for job in jobs:
        embedding = model.encode(job["description"], convert_to_tensor=True).tolist()

        metadata = {
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "link": job["link"]
        }

        collection.add(
            ids=[job["link"]],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    print(f"✅ Stored {len(jobs)} jobs in ChromaDB.")

def retrieve_jobs_from_chromadb():
    """
    Retrieves all stored jobs from ChromaDB.
    """
    results = collection.get()
    jobs = []
    
    if "metadatas" in results:
        for meta in results["metadatas"]:
            jobs.append({
                "title": meta.get("title", "Unknown"),
                "company": meta.get("company", "Unknown"),
                "location": meta.get("location", "Unknown"),
                "link": meta.get("link", "#")
            })

    print(f"✅ Retrieved {len(jobs)} jobs from ChromaDB.")
    return jobs

def retrieve_top_jobs_hybrid(cv_text, jobs, top_n, similarity_threshold=0.05):
    """
    Hybrid Search: Combines BM25 keyword search and RAG (semantic similarity).
    Removes duplicate job postings with similar Hybrid Scores.
    """
    # ✅ BM25 Lexical Search (Keyword Matching)
    tokenized_jobs = [job["title"].lower().split() for job in jobs]
    bm25 = BM25Okapi(tokenized_jobs)
    tokenized_cv = cv_text.lower().split()
    bm25_scores = bm25.get_scores(tokenized_cv)

    # ✅ Semantic Similarity (RAG)
    cv_embedding = model.encode(cv_text, convert_to_tensor=True)
    job_embeddings = [model.encode(job["title"] + " at " + job["company"], convert_to_tensor=True) for job in jobs]
    similarity_scores = [util.pytorch_cos_sim(cv_embedding, job_emb).item() for job_emb in job_embeddings]

    # ✅ Hybrid Ranking: Combine BM25 & Semantic Scores
    hybrid_scores = [0.5 * bm25 + 0.5 * sim for bm25, sim in zip(bm25_scores, similarity_scores)]
    
    # ✅ Combine Jobs & Hybrid Scores
    scored_jobs = sorted(zip(hybrid_scores, jobs), key=lambda x: x[0], reverse=True)

    # ✅ Remove Duplicate Job Listings (Similar Hybrid Scores)
    deduplicated_jobs = []
    seen_links = set()

    for score, job in scored_jobs:
        if job["link"] in seen_links:
            continue  # Skip duplicate links

        # ✅ Check if similar jobs are already in the list
        if any(abs(score - existing_score) < similarity_threshold for existing_score, _ in deduplicated_jobs):
            continue  # Skip if a very similar job is already stored

        deduplicated_jobs.append((score, job))
        seen_links.add(job["link"])

        if len(deduplicated_jobs) >= top_n:
            break  # Stop once we have enough jobs

    return deduplicated_jobs

# ✅ User Inputs

num_jobs = 30
days_filter = 5

# ✅ Define Unwanted Job Titles (Exclude)
EXCLUDED_TITLES = ["Data Engineer", "Junior", "Underwriting",  "Manager", "Head", "Lead","VP", "AVP", "SVP", "Director", "Vice President", "Chief", "Analyst", "Intern", "CO-OP" 'Officer']

# # ✅ Define Required Skills (Jobs Must Include at Least One)
REQUIRED_SKILLS = []

cv_path = "your_cv.pdf"  # Replace with your actual CV file

job_titles = ["Data Scientist", "Machine Learning", "GenAI"]

locations = ["Toronto","Montreal"]#, "Ottawa", "Vancouver", "Boston", "New York", 'Florida', "San Francisco", "Seatel", "Silocon Vally"]
num_jobs = 200
days_filter = 1
top_n = 20  # Number of top matches to retrieve

# ✅ Step 1: Extract CV Text
cv_text = extract_text_from_pdf(cv_path)

# ✅ Step 2: Scrape & Store Jobs
job_listings = scrape_linkedin_jobs(job_titles, locations, num_jobs, days_filter, EXCLUDED_TITLES)#, REQUIRED_SKILLS)

store_jobs_in_chromadb(job_listings)

# ✅ Step 3: Retrieve Jobs from ChromaDB Instead of Re-Scraping
stored_jobs = retrieve_jobs_from_chromadb()

# ✅ Step 4: Retrieve Best Job Matches (Hybrid Search with Deduplication)
top_jobs = retrieve_top_jobs_hybrid(cv_text, stored_jobs, top_n)

# ✅ Step 5: Display Results
print("\n🔹 **Top Job Matches Based on Your CV** 🔹")
for i, (score, job) in enumerate(top_jobs):
    print(f"{i+1}. {job['title']} at {job['company']} in ✅ {job['location']} (Hybrid Score: {score:.4f})")
    print(f"   🔗 {job['link']}\n")


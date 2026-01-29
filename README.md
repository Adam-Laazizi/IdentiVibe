# IdentiVibe 🧬

**IdentiVibe** is a high-performance community intelligence platform that transforms raw social media data into actionable psychological archetypes. By scraping multi-platform data (YouTube, Instagram, Reddit, and LinkedIn), IdentiVibe analyzes the "vibe" of a digital community to help creators and brands understand their audience at a granular level.

---

## 🚀 Core Technology Stack

### 🧠 The Gemini Intelligence Layer
We leverage the **Google Gemini API** not just as a chatbot, but as a multi-functional engine that powers three distinct phases of our pipeline:

1.  **The "Search Bar" (Semantic Retrieval):** Instead of simple keyword matching, we use Gemini to navigate through complex, unstructured social data. It acts as an intelligent filter, identifying the most relevant community interactions, slang, and sentiment-rich threads that define a group.
2.  **Statistical Analysis Engine:** Gemini processes the merged data arrays to perform quantitative and qualitative sentiment calculation. It weights engagement metrics against linguistic patterns to output a structured **Community Report**, including the "Overall Archetype" and "Vulnerability Scores."
3.  **The Drawing Tool (Visual Identity):** Using Gemini’s multimodal capabilities and prompt engineering, the system generates high-fidelity **Chibi Mascot Prompts**. These are then piped into our image generation backend (Nano Banana) to create a visual representation of the community's persona.



### 🗄️ Data Persistence: MongoDB
To ensure scalability and fast retrieval, IdentiVibe utilizes **MongoDB**. 
* **Community Snapshots:** Each extraction and analysis is stored as a document, allowing us to track how a community’s sentiment evolves over time.
* **Flexible Schema:** Since social media data formats are constantly changing, MongoDB’s document-based structure allows us to store YouTube comments and Reddit threads in the same collection without rigid table constraints.



### 📈 Growth & Engagement: Amplitude
We integrate **Amplitude** to move beyond basic page-view metrics. By studying the behavioral patterns of our platform users, we:
* Identify which community archetypes generate the most interest.
* A/B test different mascot styles to see which drives higher user retention.
* **Maximize Engagement:** We use Amplitude's cohort analysis to find "clever" engagement loops, such as notifying users when a community they follow shifts in sentiment or archetype.

---

## 🛠️ System Workflow

1.  **Extraction:** Scrapers (YouTube, Instagram, etc.) pull raw JSON payloads.
2.  **Merging:** `scraperMerger.py` consolidates multi-platform data into a unified context.
3.  **Analysis:** The Gemini Engine calculates the "Vibe" and generates a visual prompt.
4.  **Generation:** A custom Chibi Mascot is generated and served via a FastAPI static mount.
5.  **Tracking:** User interactions with the generated reports are logged in Amplitude for platform optimization.



---

## 💻 Local Development

### Prerequisites
* Python 3.11+
* Node.js & React
* MongoDB Instance (Local or Atlas)
* API Keys: Gemini, YouTube v3, Apify (Instagram/LinkedIn)

### Setup
1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/Adam-Laazizi/IdentiVibe.tech.git](https://github.com/Adam-Laazizi/IdentiVibe.tech.git)
    ```
2.  **Backend Environment:** Create a `.env` file with your credentials.
3.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    pip install "rembg[cpu]"
    ```
4.  **Launch Backend:**
    ```bash
    uvicorn main:app --reload --port 10000
    ```

---

## 🛡️ Privacy & Security
IdentiVibe is built with **GitHub Push Protection** and strict `.gitignore` rules to ensure that API keys and sensitive payload caches are never exposed in the version history.

---
© 2026 IdentiVibe Team

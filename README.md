# 🛡️ AI Email Spam Shield — End-to-End ML Pipeline + Gmail & Android Auto-Filter

Welcome! This project is a complete **Machine Learning & Natural Language Processing (NLP)** system built in Python that detects **Spam vs. Ham (Legitimate)** emails and integrates directly with **Gmail and the Android Gmail App**.

It can be used in **three ways**:
1. **📊 Interactive Data Science Dashboard (Streamlit):** Paste any email, inspect how the NLP pipeline cleans and vectorizes the text, and compare predictions across 3 ML models side by side.
2. **🤖 Personal Gmail Background Watcher (IMAP):** Runs in the background on your PC, scans incoming Gmail messages, and tags spam with an **`AI-Spam`** label that appears immediately on your **Android phone**.
3. **☁️ Multi-User Cloud SaaS on Vercel (FastAPI + Google OAuth 2.0):** A 100% free hosted web service where anyone can click **"Sign in with Google"** to register their Gmail account and have their emails automatically filtered in the background.

---

## 🧠 1. How the Machine Learning Pipeline Works (6 Milestones)

```text
[ 1. Dataset (5,572 Emails) ] ➔ [ 2. NLP Text Cleaning ] ➔ [ 3. TF-IDF Vectorizer ]
                                                                      │
[ 6. Streamlit / Vercel + Gmail Sync ] ◀ [ 5. Evaluation ] ◀ [ 4. Model Training ]
```

### 📊 Step 1: Dataset Selection (`train_model.py`)
* Uses the **UCI / Kaggle SMS & Email Spam Collection** dataset (`data/spam.csv`).
* Contains **5,572 total messages** (**5,169 unique messages** after deduplication: **87.4% Ham** and **12.6% Spam**).
* Automatically downloads and caches the dataset on first run (and includes a built-in offline fallback dataset).

### 🧼 Step 2: Text Preprocessing — NLP (`preprocessing.py`)
Raw email text is noisy. Every email passes through **5 cleaning stages** before reaching the model:
1. **Lowercasing:** Converts all text to lowercase so `"FREE"` and `"free"` are treated identically.
2. **Special Character & Pattern Normalization:**
   * Strips HTML tags (`<div>`, `<a href=...>`) and punctuation.
   * Normalizes URLs to `httpaddr`, email addresses to `emailaddr`, currency amounts (`$1000`, `£500`) to `moneysymb`, and numbers to `numbr`.
3. **Tokenization:** Splits sentences into individual word tokens.
4. **Stop Word Removal:** Removes common words (`"the"`, `"is"`, `"and"`, `"to"`) that carry little spam signal.
5. **Stemming:** Reduces words to their root stem using **PorterStemmer** (e.g., `"claiming"`, `"claimed"`, `"claims"` ➔ `"claim"`).

> **Example Transformation:**
> * **Raw Email:** `"URGENT! You have won a $1,000 Walmart Gift Card! Click http://claim-now.biz immediately!"`
> * **After Step 2:** `"urgent moneysymb walmart gift card click httpaddr immedi"`

### 📐 Step 3: Feature Extraction (`TfidfVectorizer`)
Machine learning models require numbers, not words. We use **TF-IDF (Term Frequency–Inverse Document Frequency)**:
* Vocabulary size: **5,000 top features**
* N-gram range: **`(1, 2)` (Unigrams + Bigrams)** so both single words (`"urgent"`, `"winner"`) and two-word phrases (`"free entry"`, `"gift card"`) are scored.
* Sublinear TF scaling enabled to prevent long emails from dominating short ones.

### 🤖 Step 4: Model Training (`train_model.py`)
We train three classic classification algorithms on an **80% stratified training split** and compare them on a **20% held-out test set**:
1. **Multinomial Naive Bayes (MNB):** Fast probabilistic baseline for text classification.
2. **Logistic Regression:** Balanced linear classifier with strong recall on minority spam classes.
3. **Support Vector Machine (Linear SVM + Calibration):** Finds the optimal hyperplane in high-dimensional 5,000-feature TF-IDF space.

### 📈 Step 5: Model Evaluation Metrics (`models/metrics.json`)
Because email datasets are imbalanced (~87% Ham, ~13% Spam), we prioritize **Precision** (making sure legitimate emails aren't falsely flagged as spam) and **F1-Score** alongside Accuracy:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (MNB)** | 97.97% | **98.25%** | 85.50% | 91.43% |
| **Logistic Regression** | 98.45% | 92.59% | **95.42%** | 93.98% |
| **Support Vector Machine (SVM)** ⭐ *(Best)* | **98.94%** | 96.88% | 94.66% | **95.75%** |

All three trained models and the fitted `TfidfVectorizer` are saved together in `models/spam_artifacts.joblib`.

---

## 📱 2. How the Gmail & Android App Integration Works

Because Gmail is cloud-synced across Web, Android, and iOS, any label applied to your Gmail account on the server **immediately appears inside the Gmail app on your Android phone**:

1. **Non-Destructive Scanning:** When a new email arrives in your Inbox, our service reads the `Subject` and `Body` without marking unread emails as read.
2. **Safe Labeling (`AI-Spam`):** If the ML model classifies the email as **Spam**, it creates and attaches a bright red **`AI-Spam`** label while keeping the email safely in your **Inbox** (or optionally moves it directly to `[Gmail]/Spam` if you toggle that setting).
3. **Instant Android View:** On your Android phone, you immediately see the red **`AI-Spam`** badge next to suspicious emails in your Inbox, or you can open the Gmail app's left menu ➔ **Labels** ➔ **`AI-Spam`** to review all flagged emails in one tap.

---

## 📂 3. Project File Structure

```text
Email spam detection/
├── api/
│   └── index.py              # Vercel Serverless FastAPI backend + SaaS Web UI (Google OAuth2)
├── data/
│   └── spam.csv              # Cached UCI SMS/Email Spam dataset (5,572 labeled messages)
├── models/
│   ├── spam_artifacts.joblib # Trained TF-IDF vectorizer + SVM, Logistic Regression & MNB models
│   └── metrics.json          # Test set Precision, Recall, F1-Score & Confusion Matrices
├── preprocessing.py          # Step 2: 5-stage NLP text cleaning pipeline
├── train_model.py            # Steps 1, 3, 4 & 5: Dataset loader, TF-IDF, model trainer & evaluator
├── app.py                    # Step 6 (Local): Interactive Streamlit UI + Live IMAP Gmail tab
├── gmail_watcher.py          # Standalone IMAP background daemon (uses 16-char Google App Password)
├── gmail_oauth_service.py    # Multi-user Google OAuth 2.0 + Gmail REST API v1 service for Vercel
├── vercel.json               # Vercel serverless routing & automated cron job configuration
├── requirements.txt          # Lightweight production dependencies for Vercel serverless deployment
├── requirements-local.txt    # Local data-science dependencies (includes Streamlit & Pandas)
└── .env.example              # Template for environment variables
```

---

## 🚀 4. Quickstart Guide (How to Run It)

### Prerequisites
Make sure you have **Python 3.10+** installed, then install the dependencies:
```bash
pip install -r requirements-local.txt
```

### Step A: Train / Rebuild the ML Models (Optional — Pre-trained model is already included)
```bash
python train_model.py
```

### Step B: Choose How You Want to Run the Project

#### Option 1: Launch the Local Streamlit Dashboard
Best for exploring the ML models, testing custom emails, inspecting TF-IDF tokens, or scanning your personal Gmail via a 16-character [Google App Password](https://myaccount.google.com/apppasswords):
```bash
streamlit run app.py
```

#### Option 2: Run the Personal Background Gmail Daemon (Terminal)
Continuously checks your Gmail inbox every 60 seconds via IMAP and labels spam as `AI-Spam`:
```bash
python gmail_watcher.py --email your_email@gmail.com --app-password "abcd efgh ijkl mnop" --interval 60
```

#### Option 3: Run the Multi-User Google OAuth2 SaaS Web App
Launches the FastAPI web service where multiple users can sign in with Google:
```bash
uvicorn api.index:app --reload --port 8000
```
Then open **http://localhost:8000** in your browser.

---

## ☁️ 5. How to Host on Vercel for 100% Free (Multi-User SaaS)

You can deploy this entire project to **Vercel** for **$0/month** so friends or users can register with **"Sign in with Google"** and have their emails automatically filtered:

1. **Google Cloud OAuth 2.0 (Free):**
   * Go to the [Google Cloud Console](https://console.cloud.google.com/apis/credentials), create a project, and enable the **Gmail API**.
   * Under **OAuth consent screen**, add the scope `https://www.googleapis.com/auth/gmail.modify`.
   * Under **Credentials ➔ Create Credentials ➔ OAuth Client ID (Web application)**, set the **Authorized Redirect URI** to:
     `https://<your-vercel-project>.vercel.app/api/auth/google/callback`
   * Copy your `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`.

2. **Multi-User Cloud Database (Free Upstash Redis):**
   * Create a free database at [Upstash.com](https://upstash.com) (no credit card required) and copy `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN`.

3. **Deploy to Vercel (Free Hobby Plan):**
   * Push this repo to GitHub and import it into [Vercel](https://vercel.com).
   * Add `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `SESSION_SECRET`, `UPSTASH_REDIS_REST_URL`, and `UPSTASH_REDIS_REST_TOKEN` under **Settings ➔ Environment Variables**.
   * **(Optional) Every-5-Minute Auto-Scan:** Add a free timer at [cron-job.org](https://cron-job.org) pointing to `https://<your-vercel-project>.vercel.app/api/cron/scan-all` so all registered users' inboxes are filtered automatically every 5 minutes!

---

## 📚 6. Research & Theoretical Foundations

This section provides the mathematical formulations, algorithmic trade-offs, security analysis, and literature references behind the project for academic, portfolio, or research purposes.

### 6.1 Mathematical Formulation of Feature Extraction (TF-IDF)

Raw text documents vary widely in length and vocabulary. Rather than raw word counts (Bag-of-Words), which overweight frequent but uninformative words, this project uses **Sublinear TF-IDF** over unigrams and bigrams ($n \in \{1, 2\}$):

1. **Sublinear Term Frequency ($\text{tf}$):** Dampens the effect of a word appearing many times in the same email:
   $$\text{tf}(t, d) = \begin{cases} 1 + \ln(f_{t,d}) & \text{if } f_{t,d} > 0 \\ 0 & \text{otherwise} \end{cases}$$
   where $f_{t,d}$ is the raw count of token/n-gram $t$ in email $d$.

2. **Smoothed Inverse Document Frequency ($\text{idf}$):** Penalizes terms that appear across most emails while boosting rare, discriminative spam triggers (e.g., `"jackpot"`, `"verify account"`, `"moneysymb"`):
   $$\text{idf}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
   where $|D|$ is the total number of training emails.

3. **L2-Normalized TF-IDF Vector:** Each email is represented as a sparse vector $\mathbf{x} \in \mathbb{R}^{5000}$ normalized to unit Euclidean length ($\|\mathbf{x}\|_2 = 1$) so that long corporate emails and short SMS-style messages are evaluated on the same scale.

---

### 6.2 Comparative Analysis of Classification Algorithms

#### 1. Multinomial Naive Bayes (MNB)
* **Mathematical Model:** Applies Bayes' theorem under the conditional independence assumption between tokens given the class $y \in \{0 \text{ (Ham)}, 1 \text{ (Spam)}\}$:
  $$\hat{y} = \arg\max_{y \in \{0,1\}} \left( \ln P(y) + \sum_{i=1}^{V} x_i \ln P(t_i \mid y) \right)$$
* **Additive (Lidstone) Smoothing ($\alpha = 0.2$):** Prevents zero-probability collapse when an unseen token appears at test time:
  $$P(t_i \mid y) = \frac{N_{y, i} + \alpha}{\sum_{j=1}^{V} N_{y, j} + \alpha V}$$
* **Research Takeaway:** Achieves the highest **Precision (98.25%)** with near-instant training speed, making it ideal when false positives must be minimized, though its conditional independence assumption lowers **Recall (85.50%)** on nuanced phishing emails.

#### 2. Logistic Regression (with Balanced Class Weights)
* **Mathematical Model:** Models the posterior probability of spam via the logistic sigmoid function:
  $$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}$$
* **Imbalance Handling:** Because the dataset is imbalanced (**87.4% Ham vs. 12.6% Spam**), standard cross-entropy loss biases the decision boundary toward Ham. Using inverse-frequency class weights ($w_y = \frac{N}{2 N_y}$) boosts **Recall to 95.42%**, catching nearly all spam at the cost of a slightly lower Precision (92.59%).

#### 3. Support Vector Machine (Linear SVM + Platt Calibration)
* **Mathematical Model:** Solves the primal soft-margin optimization problem to find the maximum-margin separating hyperplane in the 5,000-dimensional TF-IDF space:
  $$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^{N} \max\left(0, 1 - y_i'(\mathbf{w}^\top \mathbf{x}_i + b)\right), \quad y_i' \in \{-1, +1\}$$
* **Why Linear SVM Excels on Text (Joachims, 1998):** Text representations are high-dimensional, sparse, and predominantly linearly separable. Linear SVM achieves the strongest overall balance on our test set (**Accuracy: 98.94%**, **Precision: 96.88%**, **Recall: 94.66%**, **F1-Score: 95.75%**).
* **Probability Calibration:** Because raw SVM outputs signed margin distances rather than probabilities, we wrap it in `CalibratedClassifierCV` (Platt scaling) to fit a calibrated sigmoid over out-of-fold predictions.

---

### 6.3 The "Accuracy Paradox" & Asymmetric Cost in Spam Filtering

In spam detection, **Accuracy alone is an inadequate research metric**:
* A trivial classifier that predicts **"Ham" for every single email** would achieve **87.4% accuracy** on this dataset while catching **0% of spam**.
* Furthermore, the cost of classification errors is **asymmetric**:
  * **False Negative (Spam classified as Ham):** An annoying spam email stays in the user's Inbox (low cost).
  * **False Positive (Ham classified as Spam):** A critical work email, flight confirmation, or bank OTP is hidden in the Spam folder (high cost).

To address this asymmetric cost both algorithmically and in product design:
1. **Metric Definitions:**
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
2. **Safe UX Design (`AI-Spam` Label in Inbox):** Instead of destructively deleting flagged emails, our Gmail integration attaches the `AI-Spam` label while leaving the message in the Inbox by default—eliminating the risk of lost legitimate mail while still giving instant visual warning on Android and Web.

---

### 6.4 Adversarial Spam Techniques & Robustness Mechanisms

Real-world spammers actively attempt to evade statistical NLP filters (an example of **Adversarial Machine Learning**):
* **Entity Randomization:** Spammers rotate URLs, phone numbers, and dollar amounts across campaigns so exact strings never repeat.
  * *Our Defense:* In [`preprocessing.py`](preprocessing.py), regex normalization collapses all URLs into `httpaddr`, email addresses into `emailaddr`, currency figures into `moneysymb`, and numbers into `numbr`, allowing the model to learn structural patterns (e.g., `"claim moneysymb click httpaddr"`) regardless of the specific link or number.
* **Character Obfuscation (Leetspeak & Homoglyphs):** Writing `"F.R.E.E"` or `"V1agra"` to bypass dictionary tokenizers.
* **Concept Drift:** Phishing topics shift over time (e.g., from lottery scams to crypto airdrops or package delivery alerts), requiring periodic retraining on newly labeled data.

---

### 6.5 Future Research Directions

If extending this project for a thesis or advanced research paper, consider exploring:
1. **Pre-trained Transformer Encoders (BERT / DeBERTa / Spam-T5):** Replace bag-of-n-grams TF-IDF with contextual embeddings (`distilbert-base-uncased`) to capture semantic intent in spear-phishing emails that lack obvious trigger keywords.
2. **Email Header & Metadata Fusion:** Combine NLP text features with network/authentication headers (`SPF`, `DKIM`, `DMARC` pass/fail status, sender domain age, and reply-to mismatch).
3. **Gradient Boosted Decision Trees (CatBoost / LightGBM):** Combine dense text embeddings with tabular metadata features.
4. **Online / Active Learning from User Feedback:** Automatically re-weight a user's personal classifier when they manually remove or add the `AI-Spam` label in their Gmail app.

---

### 6.6 Key Academic References & Datasets

1. **Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011).** *Contributions to the Study of SMS Spam Filtering: New Collection and Results.* Proceedings of the 11th ACM Symposium on Document Engineering (DocEng '11). *(Introduced the UCI SMS Spam Collection dataset used in this project).*
2. **Joachims, T. (1998).** *Text Categorization with Support Vector Machines: Learning with Many Relevant Features.* European Conference on Machine Learning (ECML). *(Established theoretical and empirical justification for Linear SVMs in high-dimensional TF-IDF spaces).*
3. **Metsis, V., Androutsopoulos, I., & Paliouras, G. (2006).** *Spam Filtering with Naive Bayes – Which Naive Bayes?* Proceedings of the 3rd Conference on Email and Anti-Spam (CEAS). *(Evaluated Multinomial Naive Bayes variants on the Enron-Spam corpus).*
4. **Blanzieri, E., & Bryl, A. (2008).** *A Survey of Learning-Based Techniques of Email Spam Filtering.* Artificial Intelligence Review, 29(1), 63–92.


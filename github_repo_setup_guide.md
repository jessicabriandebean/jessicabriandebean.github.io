# Complete GitHub Repository Setup Guide

## Part 1: Setting Up Your Portfolio Website Repository

### Step 1: Create Your GitHub Account (If Needed)

1. Go to **https://github.com**
2. Click **"Sign up"**
3. Choose a professional username (this will be in your URL!)
   - Good: `johnsmith-data`, `jsmith-analytics`, `john-smith`
   - Avoid: `coolcoder123`, `datawizard2025`
4. Use your professional email
5. Verify your email address

---

### Step 2: Create Your Portfolio Repository

1. **Click the "+" icon** in top right → "New repository"

2. **Repository name:** `yourusername.github.io`
   - **CRITICAL:** Must be exactly your username
   - Example: If username is `johnsmith`, name it `johnsmith.github.io`
   - This exact naming enables GitHub Pages

3. **Description:** 
   ```
   Data Science Portfolio - Federal Government to Private Sector Transition
   ```

4. **Public** (must be public for free GitHub Pages)

5. **Check "Add a README file"**

6. **Add .gitignore:** Choose "Python" template

7. **Choose license:** MIT License (recommended)

8. **Click "Create repository"**

---

### Step 3: Set Up Repository Structure

#### Option A: Using GitHub Web Interface (Easiest)

**Create the folder structure:**

1. In your repository, click **"Add file"** → **"Create new file"**

2. Type: `blog/.gitkeep`
   - The slash creates a folder
   - `.gitkeep` is a placeholder (GitHub won't show empty folders)
   - Click **"Commit new file"**

3. Repeat to create:
   - `images/.gitkeep`
   - `css/.gitkeep`
   - `js/.gitkeep`
   - `assets/.gitkeep`

**Your structure should look like:**
```
yourusername.github.io/
├── .gitignore
├── LICENSE
├── README.md
├── blog/
├── images/
├── css/
├── js/
└── assets/
```

#### Option B: Using Git Command Line (Advanced)

```bash
# Clone your repository
git clone https://github.com/yourusername/yourusername.github.io.git
cd yourusername.github.io

# Create folders
mkdir blog images css js assets

# Create placeholder files
touch blog/.gitkeep images/.gitkeep css/.gitkeep js/.gitkeep assets/.gitkeep

# Commit and push
git add .
git commit -m "Initialize repository structure"
git push origin main
```

---

### Step 4: Upload Your Portfolio Homepage

#### Method 1: Upload via Web (Easiest)

1. **Save the portfolio HTML** I created earlier as `index.html` on your computer

2. In your GitHub repository, click **"Add file"** → **"Upload files"**

3. **Drag and drop** or select `index.html`

4. **Commit message:** "Add portfolio homepage"

5. **Click "Commit changes"**

#### Method 2: Create Directly in GitHub

1. Click **"Add file"** → **"Create new file"**

2. **Filename:** `index.html`

3. **Copy the entire HTML** from the portfolio template I created

4. **Paste into the editor**

5. Scroll down, commit message: "Add portfolio homepage"

6. **Click "Commit new file"**

---

### Step 5: Enable GitHub Pages

1. Go to your repository **Settings** (tab at top)

2. Scroll down to **"Pages"** in left sidebar

3. Under **"Source":**
   - Branch: **main** (or master)
   - Folder: **/ (root)**
   - Click **Save**

4. GitHub will show: 
   ```
   ✅ Your site is published at https://yourusername.github.io
   ```

5. **Wait 2-5 minutes** for deployment

6. **Visit your site!** `https://yourusername.github.io`

---

### Step 6: Customize Your Homepage

1. Click on `index.html` in your repository

2. Click the **pencil icon** (Edit this file)

3. **Replace these placeholders:**
   ```html
   <!-- Find and replace: -->
   Your Name → John Smith
   your.email@example.com → john.smith@email.com
   linkedin.com/in/yourprofile → linkedin.com/in/johnsmith
   github.com/yourusername → github.com/johnsmith
   yourusername.github.io → johnsmith.github.io
   [X] years → 5 years
   ```

4. **Update your About section** with your real story

5. **Scroll down** → Commit message: "Customize homepage"

6. **Click "Commit changes"**

7. **Wait 1-2 minutes** → Refresh your site to see changes!

---

## Part 2: Setting Up Your First Project Repository

### Step 1: Create Project Repository

1. Click **"+"** → **"New repository"**

2. **Repository name:** `economic-forecasting`
   - Use lowercase
   - Use hyphens (not underscores)
   - Be descriptive

3. **Description:**
   ```
   Time series forecasting platform for US economic indicators using Prophet, ARIMA, and LSTM
   ```

4. **Public**

5. **Check "Add a README file"**

6. **Add .gitignore:** Python template

7. **Choose license:** MIT License

8. **Click "Create repository"**

---

### Step 2: Create Project Structure

**Create these folders** (using "Create new file" method):

```
economic-forecasting/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── app.py
├── data/
│   ├── raw/.gitkeep
│   └── processed/.gitkeep
├── notebooks/
│   └── .gitkeep
├── src/
│   └── .gitkeep
├── models/
│   └── .gitkeep
└── results/
    └── images/.gitkeep
```

**Quick way:**
- Create file: `data/raw/.gitkeep`
- Create file: `data/processed/.gitkeep`
- Create file: `notebooks/.gitkeep`
- Create file: `src/.gitkeep`
- Create file: `models/.gitkeep`
- Create file: `results/images/.gitkeep`

---

### Step 3: Add Requirements File

1. Click **"Add file"** → **"Create new file"**

2. **Filename:** `requirements.txt`

3. **Content:**
```txt
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
streamlit==1.25.0
prophet==1.1.4
statsmodels==0.14.0
scikit-learn==1.3.0
fredapi==0.5.1
yfinance==0.2.28
jupyter==1.0.0
```

4. **Commit:** "Add requirements"

---

### Step 4: Create Professional README

1. Click on **README.md** to edit

2. Replace with this template (customize for your project):

```markdown
# Economic Indicator Forecasting Platform

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Project Overview

Interactive forecasting platform for US economic indicators (unemployment rate, inflation, GDP) using time series analysis and machine learning. Achieved **2.3% MAPE** on unemployment rate predictions.

**🔗 Live Demo:** [Coming Soon - Will deploy to Streamlit Cloud]

**📄 Detailed Article:** [yourusername.github.io/blog/economic-forecasting.html](https://yourusername.github.io/blog/economic-forecasting.html)

---

## ✨ Features

- 📊 Real-time data collection from FRED API
- 🤖 Multiple forecasting models (Prophet, ARIMA, LSTM)
- 📈 Interactive Streamlit dashboard
- 📉 Model performance comparison
- 🎯 3-24 month forecast horizons
- 📊 Confidence intervals and prediction analysis

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- FRED API key (get free at [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html))

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/economic-forecasting.git
cd economic-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Run Streamlit Dashboard
```bash
streamlit run app.py
```

#### Explore Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

---

## 📊 Results

| Model | MAPE | RMSE | R² Score |
|-------|------|------|----------|
| Prophet | 2.3% | 0.18 | 0.94 |
| ARIMA | 2.8% | 0.21 | 0.91 |
| LSTM | 3.1% | 0.24 | 0.89 |

**Key Findings:**
- Prophet model performed best with seasonal data
- 12-month forecasts most reliable (MAPE < 3%)
- Unemployment predictions most accurate during stable economic periods

---

## 🛠️ Technologies Used

- **Data Collection:** FRED API, Pandas
- **Modeling:** Prophet, Statsmodels (ARIMA), TensorFlow (LSTM)
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Deployment:** Streamlit Cloud

---

## 📁 Project Structure

```
economic-forecasting/
├── app.py                 # Streamlit dashboard
├── requirements.txt       # Python dependencies
├── data/                  # Data storage
│   ├── raw/              # Original data from FRED
│   └── processed/        # Cleaned data
├── notebooks/            # Jupyter notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/                  # Source code
│   ├── data_processing.py
│   ├── models.py
│   └── visualization.py
├── models/               # Saved models
└── results/              # Outputs and charts
    └── images/
```

---

## 🔬 Methodology

### 1. Data Collection
- Collected 30+ years of monthly data from FRED API
- Indicators: Unemployment rate (UNRATE), CPI (CPIAUCSL), GDP

### 2. Exploratory Data Analysis
- Time series decomposition (trend, seasonality, residuals)
- Stationarity testing (Augmented Dickey-Fuller test)
- Correlation analysis between indicators

### 3. Feature Engineering
- Lagged features (1, 3, 6, 12 months)
- Moving averages (3, 6, 12 months)
- Rate of change indicators
- Economic regime indicators

### 4. Model Development
- **Prophet:** Handles seasonality and holidays automatically
- **ARIMA:** Traditional statistical approach for time series
- **LSTM:** Deep learning for complex patterns

### 5. Evaluation
- Walk-forward validation
- Multiple metrics: MAPE, RMSE, R²
- Residual analysis

---

## 🎓 What I Learned

- Working with time series data and handling non-stationarity
- Implementing multiple forecasting algorithms and comparing performance
- Building production-ready ML applications with Streamlit
- API integration and automated data pipelines
- Model selection and hyperparameter tuning

---

## 🔮 Future Enhancements

- [ ] Add more economic indicators (housing data, consumer sentiment)
- [ ] Implement ensemble methods combining multiple models
- [ ] Real-time alerts for significant economic changes
- [ ] Integration with news sentiment analysis
- [ ] Mobile-responsive dashboard improvements

---

## 📝 Blog Post

Read the detailed project walkthrough: [Building an Economic Forecasting Platform](https://yourusername.github.io/blog/economic-forecasting.html)

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- 🌐 Portfolio: [yourusername.github.io](https://yourusername.github.io)
- 💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- 📧 Email: your.email@example.com

---

## 🙏 Acknowledgments

- Data source: Federal Reserve Economic Data (FRED)
- Prophet library by Facebook Research
- Streamlit community for deployment support

---

⭐ **If you found this project helpful, please star the repository!** ⭐
```

3. **Commit changes:** "Create comprehensive README"

---

### Step 5: Upload Your Code Files

**As you complete your project:**

1. **Upload Jupyter notebooks:**
   - Click "notebooks" folder
   - "Add file" → "Upload files"
   - Upload: `01_data_collection.ipynb`, `02_eda.ipynb`, etc.

2. **Upload Python files:**
   - Upload `app.py` (your Streamlit dashboard)
   - Upload source code to `src/` folder

3. **Upload results:**
   - Upload charts to `results/images/`
   - These will show in your README

---

### Step 6: Add Images to README

1. **Upload an image:**
   - Go to `results/images/` folder
   - Upload `dashboard_screenshot.png`

2. **Reference in README:**
   ```markdown
   ## 🎬 Demo
   
   ![Dashboard Screenshot](results/images/dashboard_screenshot.png)
   ```

3. **Commit changes**

4. **Refresh your repository** - image appears!

---

### Step 7: Update Repository Settings

1. **Go to Settings**

2. **About Section** (right side of main page):
   - Click gear icon
   - **Description:** Your project description
   - **Website:** Your Streamlit app URL (when deployed)
   - **Topics:** Add tags
     ```
     data-science
     machine-learning
     python
     time-series
     forecasting
     streamlit
     economics
     ```
   - **Save changes**

3. **Now your repo looks professional!**

---

## Part 3: Linking Everything Together

### Update Your Portfolio Homepage

1. **Go to your `yourusername.github.io` repository**

2. **Edit `index.html`**

3. **Update project links:**

```html
<!-- Find your Economic Forecasting project card -->
<div class="project-links">
    <!-- UPDATE THESE URLS -->
    <a href="https://economic-forecast.streamlit.app" class="project-link link-demo">
        🔗 Live Demo
    </a>
    <a href="https://github.com/yourusername/economic-forecasting" class="project-link link-code">
        💻 Code
    </a>
    <a href="blog/economic-forecasting.html" class="project-link link-blog">
        📄 Article
    </a>
</div>
```

4. **Replace `yourusername` with your actual GitHub username**

5. **Commit changes**

---

## Part 4: Common Issues & Solutions

### Issue 1: GitHub Pages Not Working

**Symptoms:** 404 error when visiting your site

**Solutions:**
1. Check repository name is **exactly** `yourusername.github.io`
2. Make sure `index.html` exists in root folder (not in a subfolder)
3. Verify GitHub Pages is enabled in Settings → Pages
4. Wait 5-10 minutes after first deployment
5. Clear browser cache and try incognito mode

---

### Issue 2: Images Not Showing

**Symptoms:** Broken image icons

**Solutions:**
1. Check image paths are relative: `images/photo.jpg` not `/images/photo.jpg`
2. Verify image file names match exactly (case-sensitive!)
3. Make sure images are actually uploaded to repository
4. Use supported formats: PNG, JPG, GIF, SVG

---

### Issue 3: Links Not Working

**Symptoms:** Clicking links does nothing or goes to wrong page

**Solutions:**
1. Check for typos in URLs
2. External links need `http://` or `https://`
3. Internal links use relative paths: `blog/post.html`
4. For anchor links use `#section-name`

---

### Issue 4: Can't Push to GitHub

**Symptoms:** Permission errors

**Solutions:**
1. Set up SSH key or Personal Access Token
2. Or just use GitHub web interface for now (easier!)

---

## Part 5: Using GitHub Web Interface (No Git Commands!)

### Editing Files

1. **Navigate to file** in repository
2. **Click pencil icon** (Edit this file)
3. **Make changes**
4. **Scroll down** → Add commit message
5. **Click "Commit changes"**
6. **Done!** Changes are live in 1-2 minutes

### Adding Files

1. **Navigate to folder** where you want file
2. **Click "Add file"** → Choose:
   - "Create new file" - Type content directly
   - "Upload files" - Upload from computer
3. **Commit**
4. **Done!**

### Creating Folders

1. **Click "Add file"** → "Create new file"
2. **Type:** `foldername/filename.txt`
   - The slash creates the folder!
3. **Commit**

---

## Part 6: Quick Checklist

**Your Portfolio Site (`yourusername.github.io`):**
- [ ] Repository created with exact naming
- [ ] Folder structure created (blog/, images/, etc.)
- [ ] index.html uploaded and customized
- [ ] GitHub Pages enabled in Settings
- [ ] Site is live and accessible
- [ ] All links updated with your info

**Your Project Repository (`economic-forecasting`):**
- [ ] Repository created with descriptive name
- [ ] Folder structure created
- [ ] README.md comprehensive and professional
- [ ] requirements.txt added
- [ ] Project description added in About section
- [ ] Topics/tags added for discoverability
- [ ] Code uploaded (as you complete it)
- [ ] Images/results uploaded
- [ ] Links between portfolio and project working

---

## Part 7: Next Steps

**Week 1:**
1. ✅ Set up portfolio repository
2. ✅ Deploy portfolio homepage
3. ✅ Set up first project repository
4. ✅ Write great README

**Week 2:**
1. Complete your economic forecasting project
2. Upload notebooks and code to GitHub
3. Upload results/images
4. Update README with actual results

**Week 3:**
1. Deploy Streamlit app
2. Update portfolio homepage with live demo link
3. Write blog post
4. Share on LinkedIn

---

## 🎉 You're All Set!

You now have:
- ✅ Professional portfolio website live
- ✅ Project repository properly structured
- ✅ Everything linked together
- ✅ Professional presentation

**Your URLs:**
- Portfolio: `https://yourusername.github.io`
- Project: `https://github.com/yourusername/economic-forecasting`
- (Demo coming soon): `https://economic-forecast.streamlit.app`

---

## 💡 Pro Tips

1. **Commit often** - Every small change
2. **Write good commit messages** - "Add dashboard" not "Update"
3. **Use meaningful file names** - `unemployment_forecast.png` not `image1.png`
4. **Keep README updated** - Reflect current project state
5. **Add screenshots early** - Visual proof of progress

---

Need help with any specific step? Let me know which part you're stuck on!
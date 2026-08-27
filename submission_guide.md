# ARC Prize 2026 Paper Track: End-to-End Submission Guide

Follow this 3-step walkthrough to publish your research on **GitHub**, upload your preprint to **ResearchGate**, and submit your winning writeup on **Kaggle**.

---

## Step 1: Push Code to GitHub

Open a terminal in `C:\Users\ACER\Desktop\ARC` and run:

```bash
# 1. Initialize git (if not already done)
git init

# 2. Add files (large zips are automatically excluded by .gitignore)
git add README.md LICENSE .gitignore kaggle_writeup.md submission_guide.md arc_technical_analysis.md paper/ assets/

# 3. Commit
git commit -m "feat: ARC Prize 2026 paper track submission suite"

# 4. Create a new repository on GitHub (e.g. arc-prize-2026), then link & push:
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/arc-prize-2026.git
git branch -M main
git push -u origin main
```

Your GitHub repository URL (e.g. `https://github.com/<YOUR_GITHUB_USERNAME>/arc-prize-2026`) is now your **Public Project Link**!

---

## Step 2: Publish Your Academic Preprint on ResearchGate

Publishing on ResearchGate establishes priority, generates citations, and provides judges with an extended preprint.

1. **Log in to ResearchGate:** Navigate to [researchgate.net](https://www.researchgate.net/).
2. Click **"Add research"** (top right) $\to$ Select **"Preprint"**.
3. **Upload Paper:**
   - You can upload `paper/paper.md` as a PDF (e.g., open `paper/paper.md` in VS Code / browser and "Print to PDF", or compile `paper/paper.tex` using [Overleaf](https://www.overleaf.com/)).
4. **Fill Metadata:**
   - **Title:** `From Static Invariance to Dynamic Agency: Test-Time Rank-Stabilized Adaptation and Composite Runtime Grafts for the Abstraction and Reasoning Corpus (ARC-AGI-2 & ARC-AGI-3)`
   - **Authors:** Sourab Ghosh
   - **Abstract:** Copy the abstract from [`paper/paper.md`](file:///c:/Users/ACER/Desktop/ARC/paper/paper.md).
   - **Date:** August 2026
5. Click **"Publish"** and copy your **ResearchGate publication link / DOI**.

---

## Step 3: Complete & Submit on Kaggle

1. Navigate to your draft on **Kaggle**: [ARC Prize 2026 - Paper Track](https://www.kaggle.com/competitions/arc-prize-2026-paper-track) $\to$ Click **"Edit Draft"**.
2. **Track Selection:** Select **Main Track**.
3. **Title:**
   ```
   From Static Invariance to Dynamic Agency: Test-Time Rank-Stabilized Adaptation and Composite Runtime Grafts for ARC-AGI-2 & ARC-AGI-3
   ```
4. **Subtitle:**
   ```
   A Neuro-Symbolic Approach to General Abstraction, Cognitive Trap Elimination, and Quadratic Efficiency Optimization
   ```
5. **Main Body:**
   - Open [`kaggle_writeup.md`](file:///c:/Users/ACER/Desktop/ARC/kaggle_writeup.md).
   - Copy the entire content and paste it into the Kaggle Writeup editor. (Word count: ~1,235 words — safely within the 1,500 word limit).
6. **Media Gallery (Required):**
   - Upload the generated cover visual from: [`assets/cover_image.jpg`](file:///c:/Users/ACER/Desktop/ARC/assets/cover_image.jpg).
7. **Attached Public Notebook (Required):**
   - Attach your public Kaggle notebook: `arc3sub2.ipynb` (or make it public on Kaggle and select it from your notebooks).
   - Enter your public leaderboard submission ID when prompted.
8. **Public Project Link (Optional but Recommended):**
   - Paste your **ResearchGate preprint link** or **GitHub repository URL**.
9. Click **"Save"**, then click the **"Submit"** button in the top-right corner.

---

## 🎯 Submission Checklist

- [x] Full Academic Preprint (`paper/paper.md` & `paper/paper.tex` & `paper/references.bib`)
- [x] High-Res Cover Visual (`assets/cover_image.jpg`)
- [x] Kaggle Writeup under 1,500 words (`kaggle_writeup.md`)
- [x] Open-source GitHub layout (`README.md`, `LICENSE`, `.gitignore`)
- [ ] Push repository to GitHub
- [ ] Publish preprint on ResearchGate
- [ ] Paste into Kaggle and click **Submit**

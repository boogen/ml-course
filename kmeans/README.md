# K-Means Clustering from Scratch

This repository contains an implementation of the **K-Means clustering algorithm** from scratch using NumPy. The algorithm is demonstrated on synthetic 2D data generated using `make_blobs`, and results are visualized using Matplotlib.

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/boogen/ml-course.git
   cd ml-course/kmeans
   ```

2. Set up a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip3 install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python3 model.py
   ```

---

## 🧠 Algorithm Overview

K-Means is an **unsupervised learning algorithm** that partitions data into `k` clusters by minimizing the variance within each cluster.

### ✅ Features

- Custom implementation using NumPy
- Random centroid initialization
- Convergence based on centroid movement tolerance
- Visualization of final clusters and centroids

---

## 📊 Example Output

The example generates 300 samples in 4 clusters and fits the K-Means algorithm:

![KMeans Result](plots/kmeans.png)

---

## ⚙️ Code Highlights

- Centroids are initialized randomly from data points
- Convergence is checked using Euclidean norm between old and new centroids
- Cluster assignments and centroid positions are updated iteratively

---

## 🧑‍💻 Author

Created by Marcin Bugala as a hands-on exercise in building machine learning algorithms from scratch.  
This project is intended for learning and exploration

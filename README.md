# Machine Learning Advanced 

This repository contains the assignments from the follow‑up course **Machine Learning 2 (ML Advanced)**, focused on higher‑dimensional data, dimensionality‑reduction techniques, and deep‑learning models.

Each lab drives a complete pipeline: raw data → preprocessing → modelling & tuning → evaluation on held‑out data → inference on hidden evaluation set.

---

## Repository structure

| #     | Folder                  | Data set                          | Core task                                                | Main techniques                                                                                    |
| ----- | ----------------------- | --------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **1** | `dimred-binary-fashion` | Fashion‑MNIST 28 × 28 (grayscale) | **Binary** classification after dimensionality reduction | PCA ・ LLE ・ SVM (RBF / poly) ・ Naive Bayes ・ LDA ・ generative sampling                             |
| **2** | `cnn-dnn-fashion`       | Fashion‑MNIST 32 × 32 (grayscale) | Multi‑class image classification                         | Fully‑connected NN ・ CNN (conv‑pool‑ReLU) ・ Adam & SGD ・ dropout / batch‑norm ・ data normalisation |

> **Large data note:** raw CSV files are \~130 MB and are stored via **Git LFS**; `git lfs install` is required before the first clone or pull.

---

## Quick start

```bash
# Clone with LFS support
git lfs install               # one‑time per machine
git clone git@github.com:RustemSan/ML-Advanced-Course.git
cd ML-Advanced-Course

# Create & activate your env (conda / venv) then:
pip install -r requirements.txt   # CPU‑only

# Run assignment 1
cd dimred-binary-fashion
jupyter lab notebook.ipynb
```

---

### Projects at a glance

1. **Dimensionality reduction & binary SVM**

   - Reshape 28×28 images → 784‑dim vectors.
   - Compare PCA vs. LLE for feature compression.
   - Grid‑search C & γ for SVM (RBF) plus linear SVM baseline.
   - Use fitted Naive Bayes to **generate** synthetic images via `numpy.random.multivariate_normal`.

2. **CNN vs. dense nets on Fashion‑MNIST**

   - Experiment with depth / layer width.
   - Explore optimisers (SGD, Adam), regularisation (dropout, L2), and batch normalisation.
   - Measure accuracy threshold required for extra credit (> 0.90).
   - Export predictions for hidden `evaluate.csv` into `results.csv` (columns: `ID`, `label`).

Feel free to open issues or discussions if you have questions or spot an error!


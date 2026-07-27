# Chapter 8 — Unsupervised Learning

Work from **Hands-On Machine Learning with PyTorch and Scikit-Learn** by Aurélien Géron (PyTorch edition, released December 2025).

- `unsupervised.ipynb` — chapter walkthrough
- `olivetti_faces.ipynb` — end-of-chapter exercises on the Olivetti faces dataset

## `unsupervised.ipynb` (chapter)

Follows the chapter content:

- **K-Means** on `make_blobs`: `fit_predict`, `cluster_centers_`, `labels_`, `transform` for distances to centroids, custom `init` with `good_init`, `inertia_`, `score`.
- **MiniBatchKMeans** as a scalable variant.
- **Choosing `k`**: inertia curve and **silhouette score** across `k = 1..9`.
- **Image segmentation**: downloaded a ladybug image and color-quantized it with K-Means for `n_colors ∈ {10, 8, 6, 4, 2}`, plotted the original vs. segmented versions.
- **Semi-supervised learning** on `load_digits`:
  - Baseline `LogisticRegression` on only 50 labeled samples.
  - K-Means (k=50) to pick representative digits, manually labeled them, retrained.
  - **Label propagation** across each cluster, then a stricter version keeping only the closest 50th-percentile samples per cluster.
- **DBSCAN** on `make_moons`: `labels_`, `core_sample_indices_`, `components_`; predicting new points with a `KNeighborsClassifier` trained on the core samples, with a distance cutoff to flag outliers as `-1`.
- **Gaussian Mixture Models**: `weights_`, `means_`, `covariances_`, `converged_`, `n_iter_`, `predict` / `predict_proba`, `sample`, `score_samples`, anomaly detection via density threshold, model selection with **BIC / AIC**.
- **BayesianGaussianMixture** to let the model prune unused components.

## `olivetti_faces.ipynb` (exercises)

Applied the chapter's tools on `fetch_olivetti_faces`:

1. **Split** train / validation / test with stratification on the person ID.
2. **PCA** keeping 99% of variance to reduce dimensionality before clustering.
3. **K-Means** over `k = 5..145` (step 5); picked best `k` by **silhouette score**, also inspected the **inertia** curve.
4. **Visualized clusters**: helper `plot_faces` displaying each cluster's faces with their true labels — most clusters group the same person, some mix similar-looking people.
5. **Clustering as preprocessing for classification**:
   - Baseline `RandomForestClassifier` on `X_train_pca`.
   - Same classifier on **K-Means-reduced features** (`best_model.transform`).
   - **Pipeline** sweeping `n_clusters` over `k_range` to search the best cluster count for downstream classification.
   - Trained on the **extended features** (original + K-Means distances concatenated).
6. **Generative model with GaussianMixture** (`n_components=40`) on PCA features:
   - Sampled new faces from the GMM and inverted through `pca.inverse_transform` to view them in image space.
7. **Anomaly detection**: built "bad" faces (rotated, flipped, darkened), compared their `gm.score_samples` densities against real faces, and cross-checked with a **PCA reconstruction error** function (`reconstruction_errors`) that shows bad faces reconstruct much worse than real ones.

## Assets

- `my_ladybug.png` — image used for the K-Means color-segmentation demo.
- `output.png` — plot output saved from the notebook.
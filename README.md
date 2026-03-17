# Mean Shift Clustering

Mean Shift Clustering is a non-parametric feature-space analysis technique. It is a centroid-based algorithm, which works by updating candidates for centroids to be the mean of the points within a given region. These candidates are then filtered in a post-processing stage to eliminate near-duplicates, resulting in the final set of centroids.

# TODOs
- [X] Implement Mean Shift Clustering
- [X] Test with AoS
- [X] Implement with SoA
- [ ] Check for SIMD operation where possible
- [ ] check per false sharing
- [ ] create openmp version of MSC
- [ ] Implementazione immagine con std::vector al posto del classico vettore
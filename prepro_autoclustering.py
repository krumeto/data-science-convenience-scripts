from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def dataframe_to_kmeans_clusters(df, return_plot = False, min_clusters = 2, max_clusters = 10, annotate = False):
    """Takes dataframe as input, clusters the indexes of the DF in clusters with highest silhouette score.
    Returns a dictionary of the index as keys and cluster labels as values, ready to be mapped to a df with df['col'].map(results_dict)
    Displays a PCA visualization for the user to assess the clusters validity"""

    #Replace infs with zeros
    df = df.replace({float("inf"):0, float("-inf"):0})
    df = df.fillna(0)
    
    #PCA components calculations
    pipe = make_pipeline(StandardScaler(), PCA(n_components=2))
    odds_transformed = pipe.fit_transform(df)
    
    #Clusters best silhouette score estimation between min_clusters and max_clusters
    k_range = range(min_clusters, max_clusters)
    kmeans_per_k = []
    for k in k_range:
        kmeans = make_pipeline(StandardScaler(), KMeans(n_clusters = k, random_state=3)).fit(df)
        kmeans_per_k.append(kmeans)
    
    silhouette_scores = [silhouette_score(df, model.named_steps['kmeans'].labels_)
                         for model in kmeans_per_k]
    best_index = np.argmax(silhouette_scores)
    best_k = k_range[best_index]
    best_score = silhouette_scores[best_index]
    
    kmeans_pipe = make_pipeline(StandardScaler(), KMeans(n_clusters = best_k))
    #y_pred = kmeans_pipe.fit_predict(df)
    
    #Implementation below brings stability to clusters label assignment 
    kmeans_pipe.fit(df)
    #Sorts the labels, the [::-1] reverses the order
    idx = np.argsort(kmeans_pipe.named_steps['kmeans'].cluster_centers_.sum(axis=1))[::-1]
    lut = np.zeros_like(idx)
    lut[idx] = np.arange(best_k)
    y_pred = lut[kmeans_pipe.named_steps['kmeans'].labels_] + 1 #The +1 allows clusters to start with 1
    
    #If return_plot is True, returns a plot of the silhouette scores and clusters, in addition to the clustering results
    if return_plot:
        print("Number of clusters:", best_k)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
        ax1.plot(k_range, silhouette_scores, "bo-")
        ax1.set_xlabel("$k$", fontsize=14)
        ax1.set_ylabel("Silhouette score", fontsize=14)
        ax1.plot(best_k, best_score, "rs")
        
        ax2.scatter(odds_transformed[:, 0], odds_transformed[:, 1], c=y_pred, alpha = 0.7)
        ax2.set_xlabel("first principal component")
        ax2.set_ylabel("second principal component")
        
        #If annotate is true, annotates the PCA chart
        if annotate:
            for i, feature_contribution in enumerate(odds_transformed):
                ax2.annotate(df.index[i], feature_contribution)
        
        plt.show()
    
    results_dict = {v:y_pred[i] for i, v in enumerate(df.index.values)}
    
    return results_dict
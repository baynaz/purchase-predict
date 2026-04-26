import pandas as pd
from sklearn.cluster import KMeans

N_CLUSTERS = 10


def test_prototypes_predicted_positive(model, X_test):
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=30)
    kmeans.fit(X_test)
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X_test.columns)

    probas = model.predict_proba(centroids)[:, 1]

    # Les prototypes (centres de clusters) doivent être prédits avec confiance
    assert (probas > 0.5).sum() >= N_CLUSTERS * 0.5, (
        "Moins de 50% des prototypes sont prédits positifs"
    )

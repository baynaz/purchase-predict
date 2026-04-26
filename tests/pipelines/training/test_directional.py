def test_duration_increases_probability(model, X_test):
    # On cible les observations avec faible durée (comportement attendu fort)
    X_low = X_test[X_test["duration"] < 30].copy()
    X_more = X_low.copy()
    X_more["duration"] += 60

    proba_before = model.predict_proba(X_low)[:, 1]
    proba_after = model.predict_proba(X_more)[:, 1]

    # Augmenter la durée doit augmenter la proba en moyenne
    assert proba_after.mean() > proba_before.mean(), (
        "Augmenter la durée devrait augmenter la probabilité d'achat"
    )

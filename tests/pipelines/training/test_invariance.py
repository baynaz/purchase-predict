import numpy as np

INVARIANCE_THRESHOLD = 0.10  # 10% max de variation acceptée


def test_price_invariance(model, X_test):
    X_price = X_test[X_test["price"] > 1]
    X_plus = X_price.copy()
    X_minus = X_price.copy()
    X_plus["price"] += 1
    X_minus["price"] -= 1

    model.predict_proba(X_price)[:, 1]
    y_plus = model.predict_proba(X_plus)[:, 1]
    y_minus = model.predict_proba(X_minus)[:, 1]

    abs_delta = np.abs(y_plus - y_minus)

    # La grande majorité des observations doit rester sous le seuil
    proportion_ok = (abs_delta < INVARIANCE_THRESHOLD).mean()
    assert proportion_ok > 0.90, (
        f"Trop d'observations sensibles au prix : seulement {proportion_ok:.1%} sous le seuil"
    )

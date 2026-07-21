import numpy as np


class CalibratedModel:
    """
    Lightweight wrapper around a continued-trained LightGBM Booster so it's
    drop-in compatible with how IntelliSleepScorer.py calls self.model.predict(X)
    -- i.e. returns stage codes directly (1/2/3), not raw class probabilities.

    Lives in its own module (not inside calibration.py) so that joblib can
    reliably unpickle it later regardless of how it's loaded -- if this
    class were defined inside a script run directly (`python calibration.py`),
    it would be pickled under the module name "__main__", which breaks when
    a *different* program (like IntelliSleepScorer.py) tries to load it.
    """

    def __init__(self, booster, classes):
        self.booster = booster
        self.classes_ = np.array(classes)

    def predict(self, X):
        probabilities = self.booster.predict(X)
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes_[class_indices]

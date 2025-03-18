import lightgbm as lgbm
import parameters

class LightGbmClassifier(
    n_estimatore = parameters.N_ESTIMATOR,
    max_depth = parameters.MAX_DEPTH,
    num_leaves = parameters.NUM_LEAVES
):
    def __init__(self):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.objective = "multiclass"
        self.model = self.create_model()

    def create_model(self):
        return lgbm.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            objective=self.objective
        )

    def fit_model(self, features, target):
        return self.model.fit(features, target)
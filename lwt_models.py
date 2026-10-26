"""The 16 prespecified LWT classifiers and small fast-mode tuning grids."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    RandomForestClassifier, StackingClassifier, VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


MODEL_NAMES = [
    "Logistic Regression", "LDA", "QDA", "KNN", "Linear SVM", "RBF-SVM",
    "Decision Tree", "Random Forest", "Extra Trees", "AdaBoost",
    "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "Soft Voting", "Stacking",
]

PAPER_GRIDS = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1.0, 10.0]},
    "LDA": {"model__shrinkage": [None, "auto", 0.1, 0.5]},
    "QDA": {"model__reg_param": [0.01, 0.1, 0.3, 0.6]},
    "KNN": {"model__n_neighbors": [3, 5, 7, 11], "model__weights": ["uniform", "distance"]},
    "Linear SVM": {"model__estimator__C": [0.01, 0.1, 1.0, 10.0]},
    "RBF-SVM": {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", 0.01, 0.1, 1.0]},
    "Decision Tree": {"model__max_depth": [2, 3, 5, None], "model__min_samples_leaf": [1, 3, 5]},
    "Random Forest": {
        "model__n_estimators": [100, 200], "model__max_depth": [4, 8, None],
        "model__min_samples_leaf": [1, 2, 4], "model__max_features": ["sqrt", 0.8],
    },
    "Extra Trees": {
        "model__n_estimators": [100, 200], "model__max_depth": [4, 8, None],
        "model__min_samples_leaf": [1, 2, 4], "model__max_features": ["sqrt", 0.8],
    },
    "AdaBoost": {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.03, 0.1, 0.5, 1.0]},
    "Gradient Boosting": {
        "model__n_estimators": [50, 100, 200], "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [2, 3], "model__subsample": [0.8, 1.0],
    },
    "XGBoost": {
        "model__n_estimators": [60, 120, 240], "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.03, 0.1], "model__subsample": [0.8, 1.0],
    },
    "LightGBM": {
        "model__n_estimators": [60, 120, 240], "model__num_leaves": [7, 15],
        "model__learning_rate": [0.03, 0.1], "model__min_child_samples": [5, 10, 20],
    },
    "CatBoost": {
        "model__iterations": [60, 120, 240], "model__depth": [3, 4, 5],
        "model__learning_rate": [0.03, 0.1], "model__l2_leaf_reg": [1, 3, 6],
    },
    "Soft Voting": {"model__weights": [[1, 1], [1, 2], [2, 1]]},
    "Stacking": {
        "model__final_estimator__C": [0.1, 0.5, 1.5, 5.0],
        "model__lr__model__C": [0.1, 1.0, 10.0],
        "model__et__model__max_depth": [4, None],
    },
}


def _pipe(model, scaled: bool = False) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler() if scaled else "passthrough"),
        ("model", model),
    ])


def model_specs(seed: int = 42, mode: str = "fast") -> tuple[dict[str, tuple[Pipeline, dict]], dict[str, str]]:
    if mode not in {"fast", "paper"}:
        raise ValueError(f"Unknown mode: {mode}")
    lr = lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    et = lambda: ExtraTreesClassifier(
        n_estimators=60, min_samples_leaf=2, class_weight="balanced",
        random_state=seed, n_jobs=1,
    )
    specs = {
        "Logistic Regression": (_pipe(lr(), True), {"model__C": [0.1, 1.0]}),
        "LDA": (_pipe(LinearDiscriminantAnalysis(solver="lsqr"), True),
                {"model__shrinkage": [None, "auto"]}),
        "QDA": (_pipe(QuadraticDiscriminantAnalysis(), True),
                {"model__reg_param": [0.1, 0.3]}),
        "KNN": (_pipe(KNeighborsClassifier(), True),
                {"model__n_neighbors": [3, 7]}),
        "Linear SVM": (_pipe(CalibratedClassifierCV(
            estimator=LinearSVC(dual=False, class_weight="balanced", random_state=seed),
            method="sigmoid", cv=3,
        ), True), {"model__estimator__C": [0.1, 1.0]}),
        "RBF-SVM": (_pipe(SVC(probability=True, class_weight="balanced", random_state=seed), True),
                    {"model__C": [0.5, 2.0]}),
        "Decision Tree": (_pipe(DecisionTreeClassifier(
            min_samples_leaf=3, class_weight="balanced", random_state=seed,
        )), {"model__max_depth": [3, 5]}),
        "Random Forest": (_pipe(RandomForestClassifier(
            n_estimators=60, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=seed, n_jobs=1,
        )), {"model__max_depth": [4, None]}),
        "Extra Trees": (_pipe(et()), {"model__max_depth": [4, None]}),
        "AdaBoost": (_pipe(AdaBoostClassifier(random_state=seed)),
                     {"model__n_estimators": [40, 80]}),
        "Gradient Boosting": (_pipe(GradientBoostingClassifier(
            max_depth=2, random_state=seed,
        )), {"model__n_estimators": [50, 100]}),
    }
    missing = {}
    if XGBClassifier is None:
        missing["XGBoost"] = "xgboost is unavailable"
    else:
        specs["XGBoost"] = (_pipe(XGBClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.07,
            eval_metric="logloss", random_state=seed, n_jobs=1,
        )), {"model__n_estimators": [40, 80]})
    if LGBMClassifier is None:
        missing["LightGBM"] = "lightgbm is unavailable"
    else:
        specs["LightGBM"] = (_pipe(LGBMClassifier(
            n_estimators=60, max_depth=3, num_leaves=7, min_child_samples=10,
            verbosity=-1, random_state=seed, n_jobs=1,
        )), {"model__n_estimators": [40, 80]})
    if CatBoostClassifier is None:
        missing["CatBoost"] = "catboost is unavailable"
    else:
        specs["CatBoost"] = (_pipe(CatBoostClassifier(
            iterations=60, depth=4, learning_rate=0.07, l2_leaf_reg=3, verbose=0,
            allow_writing_files=False, thread_count=1, random_state=seed,
        )), {"model__iterations": [40, 80]})

    voting = VotingClassifier(
        estimators=[("lr", _pipe(lr(), True)), ("et", _pipe(et()))], voting="soft"
    )
    stacking = StackingClassifier(
        estimators=[("lr", _pipe(lr(), True)), ("et", _pipe(et()))],
        final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
        cv=3, n_jobs=1,
    )
    specs["Soft Voting"] = (_pipe(voting), {"model__weights": [[1, 1], [1, 2]]})
    specs["Stacking"] = (_pipe(stacking), {"model__final_estimator__C": [0.5, 1.5]})
    if mode == "paper":
        specs = {name: (estimator, PAPER_GRIDS[name]) for name, (estimator, _) in specs.items()}
    return {name: specs[name] for name in MODEL_NAMES if name in specs}, missing

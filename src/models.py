"""Model definitions used in the original notebook."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE


def get_models() -> dict:
    """
    Return the classifiers from the notebook.

    The main logic is preserved: Decision Tree, Random Forest, and SVM are
    compared using cross-validation.
    """
    return {
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "svc": SVC(cache_size=1000),
    }

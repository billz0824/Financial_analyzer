import pandas as pd
import numpy as np
from collections import Counter

#####################################################################
# THIS IS A COLLECTION OF BASIC STATISTICAL MODELS (MORE IS COMING) #
#####################################################################


class Linear_Model:
    def __init__(self, input_shape=None, output_shape=None, model_type='linear', alpha=0.0):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_type = model_type
        self.alpha = alpha
        self.weights = None
        self.bias = None
    
    # gradient descent to avoid matrix inversion
    def train(self, X_train, y_train, X_test, y_test, learning_rate=0.01, epochs=1000, log_interval=100):
        X_train = np.array(X_train)
        y_train = np.array(y_train).reshape(-1, 1)
        X_test = np.array(X_test)
        y_test = np.array(y_test).reshape(-1, 1)
        n_samples, n_features = X_train.shape
        
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            y_pred_train = np.dot(X_train, self.weights) + self.bias
            error_train = y_pred_train - y_train
            
            dw = (1 / n_samples) * np.dot(X_train.T, error_train)
            db = (1 / n_samples) * np.sum(error_train)
            
            if self.model_type == 'lasso':
                dw += self.alpha * np.sign(self.weights) / n_samples
            elif self.model_type == 'ridge':
                dw += 2 * self.alpha * self.weights / n_samples
            
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
            
            if epoch % log_interval == 0 or epoch == epochs - 1:
                train_loss = np.mean(error_train ** 2)
                y_pred_test = np.dot(X_test, self.weights) + self.bias
                test_loss = np.mean((y_pred_test - y_test) ** 2)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")
        
    
    def predict(self, X_test):
        X_test = np.array(X_test)

        # screen for zero weight indices
        active_features = np.where(self.weights != 0)[0]
        X_test_active = X_test[:, active_features]  
        
        return np.dot(X_test_active, self.weights[active_features]) + self.bias
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.mean((y_test - y_pred) ** 2)
    
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, task='regression'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.task = task
        self.root = None
    
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(set(y)) == 1:
            return Node(value=np.mean(y) if self.task == 'regression' else Counter(y).most_common(1)[0][0])
        
        best_feature, best_threshold = self._best_split(X, y, n_features)
        if best_feature is None:
            return Node(value=np.mean(y) if self.task == 'regression' else Counter(y).most_common(1)[0][0])
        
        left_idx = X[:, best_feature] < best_threshold
        right_idx = ~left_idx
        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)
    
    def _best_split(self, X, y, n_features):
        best_feature, best_threshold, best_gain = None, None, -float('inf')
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] < threshold
                right_idx = ~left_idx
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                
                gain = self._information_gain(y, y[left_idx], y[right_idx])
                if gain > best_gain:
                    best_feature, best_threshold, best_gain = feature, threshold, gain
        return best_feature, best_threshold
    
    def _information_gain(self, parent, left, right):
        def variance(y):
            return np.var(y)
        
        def entropy(y):
            hist = np.bincount(y)
            probs = hist / len(y)
            return -np.sum([p * np.log2(p) for p in probs if p > 0])
        
        p = len(left) / len(parent)
        if self.task == 'regression':
            return variance(parent) - p * variance(left) - (1 - p) * variance(right)
        else:
            return entropy(parent) - p * entropy(left) - (1 - p) * entropy(right)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        return self._traverse_tree(x, node.left if x[node.feature] < node.threshold else node.right)

class RandomForest:
    def __init__(self, n_trees=50, max_depth=None, min_samples_split=3, task='regression'):
        self.n_trees = n_trees
        self.task = task
        self.trees = [DecisionTree(max_depth, min_samples_split, task) for _ in range(n_trees)]
    
    def fit(self, X, y, X_val=None, y_val=None):
        for tree in self.trees:
            indices = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[indices], y[indices])
        
        if X_val is not None and y_val is not None:
            val_predictions = self.predict(X_val)
            val_error = np.mean((y_val - val_predictions) ** 2) if self.task == 'regression' else np.mean(y_val != val_predictions)
            print(f"Validation Error: {val_error:.4f}")
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        if self.task == 'regression':
            return np.mean(tree_preds, axis=0)
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_preds)

class XGBoost:
    def __init__(self, n_trees=50, learning_rate=0.01, task='regression'):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.task = task
        self.trees = []
        self.residuals = None
    
    def fit(self, X, y, X_val=None, y_val=None):
        self.residuals = y
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=None, task='regression')
            tree.fit(X, self.residuals)
            predictions = tree.predict(X)
            self.residuals -= self.learning_rate * predictions
            self.trees.append(tree)
        
        if X_val is not None and y_val is not None:
            val_predictions = self.predict(X_val)
            val_error = np.mean((y_val - val_predictions) ** 2) if self.task == 'regression' else np.mean(y_val != val_predictions)
            print(f"Validation Error: {val_error:.4f}")
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return np.round(predictions) if self.task == 'classification' else predictions
 
class Tree_Ensemble_Model:
    def __init__(self, input_shape=None, output_shape=None, model_type='decision_tree', task='regression', **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_type = model_type
        self.task = task
        self.kwargs = kwargs
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        if self.model_type == 'decision_tree':
            return DecisionTree(task=self.task, **self.kwargs)
        elif self.model_type == 'random_forest':
            return RandomForest(task=self.task, **self.kwargs)
        elif self.model_type == 'xgboost':
            return XGBoost(task=self.task, **self.kwargs)
        else:
            raise ValueError("Unsupported model type")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train, X_val, y_val)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        if self.task == 'regression':
            return np.mean((y_test - y_pred) ** 2)
        return np.mean(y_test != y_pred)
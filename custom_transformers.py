import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import re

# Custom transformer for URL feature extraction
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.extract_features(url) for url in X])

    def extract_features(self, url):
        if not isinstance(url, str):
            url = str(url)
        features = []
        features.append(len(url))  # URL length
        features.append(url.count('.'))  # Dot count
        features.append(url.count('/'))  # Slash count
        features.append(1 if 'https' in url else 0)  # 'https' presence
        features.append(len(re.findall(r'[0-9]', url)))  # Digits
        features.append(len(re.findall(r'[a-z]', url)))  # Lowercase letters
        features.append(len(re.findall(r'[A-Z]', url)))  # Uppercase letters
        features.append(len(re.findall(r'[^a-zA-Z0-9]', url)))  # Special chars
        hostname = url.split('//')[-1].split('/')[0]
        features.append(len(hostname))  # Hostname length
        features.append(1 if re.search(r'(login|signup|secure|account|update|free|verify|password)', url) else 0)  # Keyword flag
        return features

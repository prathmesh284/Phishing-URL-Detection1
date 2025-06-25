import joblib

# Load trained model
model_pipeline = joblib.load('phishing_model.pkl')

def predict_url(url: str) -> str:
    if not isinstance(url, str) or not url.startswith("http"):
        raise ValueError("Invalid URL format.")
    prediction = model_pipeline.predict([url])[0]
    return 'phishing' if prediction == 0 else 'not phishing'

# Example usage
if __name__ == "__main__":
    # test_url = "http://allegro.pl-oferta722137.cyou"
    test_url = "http://www.google.com"
    result = predict_url(test_url)
    print(f"\nThe URL '{test_url}' is **{result.upper()}**")

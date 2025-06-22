#!/usr/bin/env python3
"""
Debug script to test the category classifier and identify issues.
"""

import sys
import os
import pandas as pd
import pickle

def test_category_classifier():
    """Test the category classifier model."""
    
    print("🔍 Testing Category Classifier...")
    
    # Test model loading
    model_path = "models/category_classifier.pkl"
    print(f"📁 Model path: {model_path}")
    print(f"📁 Model exists: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        print("❌ Model file not found!")
        return
    
    try:
        # Load the model directly
        print("🔄 Loading model...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully")
        print(f"📊 Model type: {type(model)}")
        
        # Check model attributes
        print("\n🔍 Model attributes:")
        if hasattr(model, 'classes_'):
            print(f"   Classes: {model.classes_}")
        if hasattr(model, 'predict'):
            print(f"   Has predict method: ✅")
        if hasattr(model, 'predict_proba'):
            print(f"   Has predict_proba method: ✅")
        
        # Check if it's a pipeline
        if hasattr(model, 'named_steps'):
            print(f"   Is pipeline: ✅")
            print(f"   Pipeline steps: {list(model.named_steps.keys())}")
        
        # Test with simple text data
        print("\n🧪 Testing with simple text data...")
        test_claims = [
            "Our company has achieved 100% carbon neutrality through renewable energy.",
            "We are committed to sustainable practices and environmental stewardship.",
            "Our ESG initiatives have resulted in a 50% reduction in emissions."
        ]
        
        for i, claim in enumerate(test_claims):
            print(f"\n📝 Test claim {i+1}: {claim[:50]}...")
            
            # Create simple DataFrame
            df = pd.DataFrame([{'esg_claim_text': claim}])
            print(f"   DataFrame shape: {df.shape}")
            
            # Make prediction
            if hasattr(model, 'predict'):
                try:
                    prediction = model.predict(df)
                    print(f"   Raw prediction: {prediction}")
                    
                    if hasattr(model, 'classes_'):
                        category = model.classes_[prediction[0]]
                        print(f"   Category: {category}")
                    else:
                        print(f"   Category: {prediction[0]}")
                    
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(df)
                        print(f"   Probabilities: {probabilities[0]}")
                        
                        if hasattr(model, 'classes_'):
                            prob_dict = dict(zip(model.classes_, probabilities[0]))
                            print(f"   Probability dict: {prob_dict}")
                
                except Exception as e:
                    print(f"   ❌ Error during prediction: {e}")
                    import traceback
                    traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()

def test_greenwashing_classifier():
    """Test the greenwashing classifier model."""
    
    print("\n🔍 Testing Greenwashing Classifier...")
    
    model_path = "models/greenwashing_classifier.pkl"
    print(f"📁 Model path: {model_path}")
    print(f"📁 Model exists: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        print("❌ Model file not found!")
        return
    
    try:
        # Load the model directly
        print("🔄 Loading model...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully")
        print(f"📊 Model type: {type(model)}")
        
        # Test with simple text data
        test_claim = "Our company has achieved 100% carbon neutrality through renewable energy."
        df = pd.DataFrame([{'esg_claim_text': test_claim}])
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)
            print(f"   Greenwashing probability: {probabilities[0][1]:.3f}")
        
    except Exception as e:
        print(f"❌ Error testing greenwashing classifier: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Category Classifier Debug...")
    test_category_classifier()
    test_greenwashing_classifier()
    print("\n✅ Debug complete!") 
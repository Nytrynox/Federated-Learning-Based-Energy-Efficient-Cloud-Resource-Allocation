"""
Simple command-line interface for federated learning predictions.
Quick way to train and predict without the interactive menu.
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from interactive_prediction import InteractiveFederatedPredictor


def main():
    parser = argparse.ArgumentParser(description='Federated Learning Prediction CLI')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train federated model')
    train_parser.add_argument('data_file', help='Path to training data file (CSV/JSON)')
    train_parser.add_argument('--clients', type=int, default=5, help='Number of federated clients')
    train_parser.add_argument('--rounds', type=int, default=10, help='Number of training rounds')
    train_parser.add_argument('--output', default='trained_model.pkl', help='Output model file')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('model_file', help='Path to trained model file')
    predict_parser.add_argument('data_file', help='Path to data file for prediction')
    predict_parser.add_argument('--output', help='Output predictions to JSON file')
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Create sample data')
    sample_parser.add_argument('--output', default='sample_data.csv', help='Output sample data file')
    sample_parser.add_argument('--size', type=int, default=1000, help='Number of samples')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print(f"🚀 Training federated model with {args.data_file}")
        
        predictor = InteractiveFederatedPredictor()
        X, y = predictor.prepare_user_data(args.data_file)
        
        if y is None:
            print("❌ No target column found in data. Cannot train model.")
            return
        
        results = predictor.train_federated_model(X, y, args.clients, args.rounds)
        predictor.save_model(args.output)
        
        print(f"✅ Model trained and saved to {args.output}")
        if 'global_accuracy' in results and results['global_accuracy']:
            final_acc = results['global_accuracy'][-1]
            print(f"📊 Final accuracy: {final_acc:.2%}")
    
    elif args.command == 'predict':
        print(f"🔮 Making predictions with {args.model_file}")
        
        predictor = InteractiveFederatedPredictor(args.model_file)
        results = predictor.predict(args.data_file)
        
        # Display results
        predictor._display_prediction_results(results)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Predictions saved to {args.output}")
    
    elif args.command == 'sample':
        print(f"📝 Creating sample data with {args.size} samples")
        
        predictor = InteractiveFederatedPredictor()
        df = predictor.create_sample_data(args.output)
        
        print(f"✅ Sample data created: {args.output}")
        print(f"📋 Shape: {df.shape}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

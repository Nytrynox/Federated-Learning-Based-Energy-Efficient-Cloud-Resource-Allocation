"""
Simple example demonstrating federated learning with energy monitoring.
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def run_basic_example():
    """Run a basic federated learning example."""
    print("🚀 Federated Learning Energy-Efficient Example")
    print("=" * 50)
    
    try:
        # Import after path setup
        from simulation.run_simulation import run_simulation_example
        
        print("📊 Running federated learning simulation...")
        results = run_simulation_example()
        
        print("\n✅ Simulation completed successfully!")
        print(f"📈 Final accuracy: {results.performance_metrics.get('final_accuracy', 0):.2f}%")
        print(f"⚡ Total energy: {results.performance_metrics.get('total_energy_consumption', 0):.2f} Wh")
        print(f"🎯 Energy efficiency: {results.performance_metrics.get('energy_efficiency', 0):.4f}")
        
        print("\n📁 Generated files:")
        print("- simulation_report.md (detailed report)")
        print("- simulation_results_*.png (visualizations)")
        print("- simulation_results.json (raw data)")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check the logs for more details")

def run_api_example():
    """Run the API server example."""
    print("🌐 Starting Federated Learning API Server")
    print("=" * 50)
    
    try:
        from api.app import FederatedLearningAPI
        
        print("🔧 Initializing API server...")
        api = FederatedLearningAPI()
        
        print("🚀 Starting server on http://localhost:8000")
        print("📖 API documentation:")
        print("  - GET  /health - Health check")
        print("  - POST /api/v1/training/start - Start training")
        print("  - GET  /api/v1/training/status - Training status")
        print("  - GET  /api/v1/resources/status - Resource status")
        print("\n🛑 Press Ctrl+C to stop the server")
        
        api.run()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to install dependencies: pip install -r requirements.txt")
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

def show_menu():
    """Show the example menu."""
    print("\n🎯 Federated Learning Examples")
    print("=" * 30)
    print("1. Run Simulation Example")
    print("2. Start API Server")
    print("3. Exit")
    print()

def main():
    """Main function."""
    setup_logging()
    
    print("🔬 Federated Learning Energy-Efficient Cloud Resource Allocation")
    print("A comprehensive system for distributed machine learning with energy optimization")
    
    while True:
        show_menu()
        choice = input("Select an option (1-3): ").strip()
        
        if choice == '1':
            run_basic_example()
        elif choice == '2':
            run_api_example()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1, 2, or 3.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()

"""Atlas Neural Asset Discovery - Compressed Launcher"""
import sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
def main():
    print("🚀 Atlas Neural Asset Discovery Engine");print("=" * 50)
    try:
        from src.core.atlas_engine import main as atlas_main
        results = atlas_main()
        return results
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
if __name__ == "__main__":
    main()
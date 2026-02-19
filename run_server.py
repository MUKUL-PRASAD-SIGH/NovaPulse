"""Run NovaAI server with error logging."""
import sys
import traceback

try:
    print("=" * 60)
    print("Starting Nova Intelligence Agent...")
    print("=" * 60)
    
    # Import and run
    from app.main import app
    import uvicorn
    
    print("\n✓ Imports successful")
    print("✓ Starting server on http://localhost:8000")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ ERROR STARTING SERVER")
    print("=" * 60)
    print(f"\nError: {str(e)}\n")
    print("Full traceback:")
    traceback.print_exc()
    print("=" * 60)
    sys.exit(1)

import os
import subprocess

def build_package():
    print("Starting PyPI Build Process...")
    # Get absolute path to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Run python -m build
    result = subprocess.run(["python", "-m", "build"], cwd=project_root)
    
    if result.returncode == 0:
        print("\n[SUCCESS] Build Successful!")
        print("Your distribution files (.whl and .tar.gz) are now located in the /dist/ directory.")
        print("\nWhen you are ready to officially publish:")
        print("1. Go to https://pypi.org/ and register an account.")
        print("2. Run the following command in your terminal:")
        print("   python -m twine upload dist/*")
        print("3. Enter your PyPI credentials when prompted.")
    else:
        print("\n[ERROR] Build Failed.")

if __name__ == "__main__":
    build_package()

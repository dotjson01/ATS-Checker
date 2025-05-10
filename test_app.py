import os

# Check if all required files exist
files_to_check = ['app.py', 'requirements.txt', '.env', 'README.md', 'LICENSE', '.gitignore']
missing_files = []

for file in files_to_check:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    print(f"Missing files: {', '.join(missing_files)}")
else:
    print("All required files are present!")

# Check content of requirements.txt
try:
    with open('requirements.txt', 'r') as f:
        requirements = f.read()
        required_packages = ['streamlit', 'google-generativeai', 'python-dotenv', 'PyPDF2']
        missing_packages = []
        
        for package in required_packages:
            if package not in requirements:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"Missing packages in requirements.txt: {', '.join(missing_packages)}")
        else:
            print("All required packages are in requirements.txt!")
except Exception as e:
    print(f"Error reading requirements.txt: {str(e)}")

print("\nProject setup is complete! To run the application:")
print("1. Install the required packages: pip install -r requirements.txt")
print("2. Add your Google API key to the .env file")
print("3. Run the application: streamlit run app.py")

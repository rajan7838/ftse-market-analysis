#!/usr/bin/env python
import os
import subprocess
import sys

def main():
    print("="*50)
    print("FTSE MARKET ANALYSIS")
    print("="*50)
    print("1. Run Data Preprocessing")
    print("2. Train Models")
    print("3. Evaluate Models")
    print("4. Launch Streamlit App")
    print("5. Run All")
    print("="*50)
    
    choice = input("Enter choice (1-5): ")
    
    if choice == '1':
        subprocess.run([sys.executable, 'src/data_prep.py'])
    elif choice == '2':
        subprocess.run([sys.executable, 'src/train.py'])
    elif choice == '3':
        subprocess.run([sys.executable, 'src/evaluate.py'])
    elif choice == '4':
        subprocess.run(['streamlit', 'run', 'app/app.py'])
    elif choice == '5':
        subprocess.run([sys.executable, 'src/data_prep.py'])
        subprocess.run([sys.executable, 'src/train.py'])
        subprocess.run([sys.executable, 'src/evaluate.py'])
        subprocess.run(['streamlit', 'run', 'app/app.py'])
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
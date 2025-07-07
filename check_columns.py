import pandas as pd

try:
    # Check credits.csv columns
    credits_df = pd.read_csv('data/credits.csv', nrows=5)
    print("Credits CSV columns:")
    print(list(credits_df.columns))
    print("\nFirst few rows of credits:")
    print(credits_df.head())
    
    print("\n" + "="*50 + "\n")
    
    # Check projects.csv columns  
    projects_df = pd.read_csv('data/projects.csv', nrows=5)
    print("Projects CSV columns:")
    print(list(projects_df.columns))
    print("\nFirst few rows of projects:")
    print(projects_df.head())
    
except Exception as e:
    print(f"Error: {e}") 
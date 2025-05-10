import pandas as pd
def save_results_to_excel(results, excel_file='results.xlsx'):
    # Convert results list to a pandas DataFrame
    df = pd.DataFrame(results)
    
    # Save to Excel file (write if it doesn't exist, append otherwise)
    try:
        # If the file already exists, append without overwriting
        with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
    except FileNotFoundError:
        # If the file doesn't exist, create a new one
        df.to_excel(excel_file, index=False, sheet_name='Results')

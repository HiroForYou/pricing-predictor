import pandas as pd
import os

from utils import complete_missing_rows, sorting_df

def filter_csv(input_file, output_file, target_value_0, target_value_1, target_value_2):
    
    # Check if the 'data' folder exists, create if not
    if not os.path.exists('data'):
        os.makedirs('data')

    # Read the CSV file using pandas
    data = pd.read_csv(input_file)

    # Filter rows based on the specified target value in the 'Level 6' column
    filtered_data = data[(data['Categoria'] == target_value_0) & (data['Sub categoria'] == target_value_1) & (data['Nivel 6'] == target_value_2)]

    # Exclude rows where the 'Año' column value is 2019
    filtered_data = filtered_data[filtered_data['Año'] != 2019]

    # Sort the filtered data using sorting_df function
    filtered_data = complete_missing_rows(filtered_data)
    filtered_data = sorting_df(filtered_data)

    # Save the sorted and filtered data to a new CSV file in the 'data' folder
    filtered_data.to_csv(output_file, index=False)
    print(f"Filtered and sorted data has been saved to '{output_file}'")

if __name__ == '__main__':
    # File paths and target value input
    input_csv_file = './data/homogeneo.csv'  # Replace with your CSV file
    output_csv_file = './data/unit_homogeneo.csv'  # Replace with the output file name
    target_0 = 'ZAPATILLAS'
    target_1 = 'URBANO'
    target_2 = 'FLAT'  # Replace with your specific value

    # Call the function to filter, sort, and save the data
    filter_csv(input_csv_file, output_csv_file, target_0, target_1, target_2)

import pandas as pd
import os

from utils import complete_missing_rows, sorting_df

def filter_csv(input_file, output_file):
    # Check if the 'data' folder exists, create if not
    if not os.path.exists('data'):
        os.makedirs('data')

    # Read the CSV file using pandas
    data = pd.read_csv(input_file)

    # Get unique values for 'Nivel 6' and 'Sub categoria'
    unique_category_values = data['Categoria'].unique()
    unique_nivel6_values = data['Nivel 6'].unique()
    unique_subcategoria_values = data['Sub categoria'].unique()

    # List to store filtered data
    filtered_data_list = []

    for target_value_0 in unique_category_values:
        for target_value_1 in unique_subcategoria_values:
            for target_value_2 in unique_nivel6_values:
                # Filter rows based on the current combination of target values
                filtered_data = data[(data['Categoria'] == target_value_0) & (data['Sub categoria'] == target_value_1) & (data['Nivel 6'] == target_value_2)]
                filtered_data = filtered_data[filtered_data['Año'] != 2019]
                filtered_data['Costo'] = filtered_data['Costo'].apply(lambda x: max(0, x))
                filtered_data = complete_missing_rows(filtered_data)
                #print("filtered_data", filtered_data.head(30))
                # Append filtered data to the list
                filtered_data_list.append(filtered_data)

    # Concatenate all filtered data into a single DataFrame
    combined_filtered_data = pd.concat(filtered_data_list)
    combined_filtered_data = sorting_df(combined_filtered_data)

    # Save the combined filtered data to a new CSV file in the 'data' folder
    combined_filtered_data.to_csv(output_file, index=False)
    print(f"All filtered and combined data has been saved to '{output_file}'")

if __name__ == '__main__':
    # File paths and target value input
    input_csv_file = './data/2019-2023_without_facturacion.csv'  # Replace with your CSV file
    output_csv_file = './data/combined_filtered_datas.csv'  # Replace with the output file name

    # Call the function to filter, sort, and save the data
    filter_csv(input_csv_file, output_csv_file)

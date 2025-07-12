import pandas as pd

def clean_and_separate_variants(input_csv_path, output_csv_path):
    """
    Reads a CSV file, separates the 'Available Variants' column into 'Color' and 'Size' columns,
    and saves the result to a new CSV file.

    Args:
        input_csv_path (str): The path to the input CSV file.
        output_csv_path (str): The path to the output CSV file.
    """
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv_path)

    # Function to extract color and size from the 'Available Variants' column
    def extract_variants(variant_string):
        color = ''
        size = ''
        if isinstance(variant_string, str):
            parts = variant_string.split('|')
            for part in parts:
                if 'Color:' in part:
                    color = part.replace('Color:', '').strip()
                elif 'Size:' in part:
                    size = part.replace('Size:', '').strip()
        return color, size

    # Apply the function to the 'Available Variants' column
    df[['Color', 'Size']] = df['Available Variants'].apply(lambda x: pd.Series(extract_variants(x)))

    # Drop the original 'Available Variants' column
    df = df.drop(columns=['Available Variants'])

    # Save the cleaned data to a new CSV file
    df.to_csv(output_csv_path, index=False)

if __name__ == '__main__':
    clean_and_separate_variants('formatted_products.csv', 'cleaned_products.csv')

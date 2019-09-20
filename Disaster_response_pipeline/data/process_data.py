import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    - Takes two CSV files as input
    - Imports them as pandas dataframe
    - Merges it into a dataframe
    Args:
    messages_file_path str: Messages CSV file
    categories_file_path str: Categories CSV file
    Returns:
    pandas df with messages and categories files
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories)


def clean_data(df):
    """
    - Cleans the combined dataframe    
    Args:
    df: Merged pandas dataframe from load_data() function
    Returns:
    df: Cleaned data
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand = True)
    # extract a list of new column names for categories.
    row = categories.iloc[0,:]    
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to numbers 0 or 1    
    for column in categories:    
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories, right_index=True, left_index=True)
    # drop duplicates
    df.drop_duplicates(inplace = True)
    df.drop('related', axis =1, inplace = True)
    return df


def save_data(df, database_filename):
    """
    Saves data to a SQL database

    Args:
    df: Cleaned df from clean_data() function
    database_filename: str file path into which the clean data will be saved
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    db_name = database_filename.split("/")[-1] 
    table_name = db_name.split(".")[0]
    df.to_sql(table_name, engine, index=False, if_exists = 'replace')
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
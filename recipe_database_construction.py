import mysql.connector
from mysql.connector import errorcode
import json
import os



from repurposing.mysql.create import get_connection
from repurposing.mysql.create import create_database
from repurposing.mysql.create import use_database
from repurposing.mysql.create import create_tables
from repurposing.mysql.create import drop_tables
from repurposing.mysql.create import create_ingredient_counts
from repurposing.mysql.create import DB_NAME
from repurposing.mysql.create import TABLES


#from repurposing.datasets.dataset_1Mrecipes import get_procpaths
from repurposing.mysql.insert import insert_recipes_from
from repurposing.mysql.insert import insert_all_recipes
from repurposing.mysql.insert import insert_ingredients_from
from repurposing.mysql.insert import insert_all_ingredients
from repurposing.mysql.insert import insert_all_constituents

from repurposing.mysql.filter import filter_ingredients_to_table
from repurposing.mysql.filter import filter_recipes_to_table

from repurposing.file_io.paths import get_rawpaths
from repurposing.file_io.paths import get_procpaths
def main(option, user, password, db_name=DB_NAME,
        tables_to_drop=None, force_drop=False, exclude_tables_to_drop=None,
        tables_to_create=None, 
        exclude_tables_to_create=None, 
        force_create=False,
        input_subdir=None, max_ifile_id=15, min_ifile_id=1):
    cnx = get_connection(user=user, password=password, db_name=None)
    cursor = cnx.cursor()        

    # assume that the database is pre-existing
    new_database = False
    if option == 'create_database' or option == 'create_all':
        print("Creating database!")
        try:
            create_database(cursor)
            # if create database is successful we have a new database
            new_database = True
        except mysql.connector.Error as err:
            print("Error constructing database")
            print(err)
            pass

    use_database(cnx, cursor, db_name)

    if option == 'drop_tables' or (option == 'create_all' and not new_database):
        print("Dropping tables!")
        fragile = not force_drop
        drop_tables(
            cnx, cursor, table_names=tables_to_drop,
            exclude_table_names=exclude_tables_to_drop,
            fragile=fragile)
        new_database = True

    if option == 'create_tables' or (option == 'create_all' and new_database):
        print("Creating tables")
        fragile = not force_create
        create_tables(
            cnx, cursor, table_names=tables_to_create,
            exclude_table_names=exclude_tables_to_create,
            fragile=fragile)

    # get file names for input files (will be used for multiple options
    print(f"input_subdir = {input_subdir}")
    ifpaths = get_procpaths(
        input_subdir, maxid=max_ifile_id, minid=min_ifile_id)

    # insert recipes this means the title, url and id
    if option == 'insert_recipes'  or option == 'create_all':
        insert_all_recipes(cnx, cursor, ifpaths)

    # insert ingredients, these are the ingredients that have been "name"d
    # by the preprocessing, possibly shared by more than one recipe
    if option == 'insert_ingredients'  or option == 'create_all':
        insert_all_ingredients(cnx, cursor, ifpaths)

    # insert constituents these are the ingredients for specific recipes
    if option == 'insert_constituents' or option == 'create_all':
        insert_all_constituents(cnx, cursor, ifpaths)

    if option == 'create_ingredient_counts' or option == 'create_all':    
        create_ingredient_counts(cnx, cursor)

    if option == 'filter_ingredient_counts':    
        filter_ingredients_to_table(
            cnx, cursor,
            constituents_table='filtered_constituents',
            ingredients_table_to='filtered_ingredient_counts',
            ingredients_table_from='ingredients')

    if option == 'filter_recipes':
        filter_recipes_to_table(
                cnx, cursor, threshold=None,
                ingredient_counts_table='filtered_ingredient_counts',
                recipe_table_from='recipes',
                recipe_table_to='filtered_recipes',
                desired_fraction=0.5, verbose=True)    

    cursor.close()
    cnx.close()

def create_parser():
    description= """
        Provides functionality to create, populate and manipulate recipe
        database from json files (typically that have been processed in
        some way"""
    parser = argparse.ArgumentParser(
        prog='recipe_database_construction',
        description=description,
        epilog='See git repository readme for more details.')
        
    parser.add_argument('--user', '-u', type=str,
        help='mysql user-name for secure connection.')
    parser.add_argument('--password', '-p', type=str,
        help='mysql password for secure connection.')
    options = [
        'create_database', 'drop_tables', 'create_tables', 'insert_recipes',
        'insert_ingredients', 'insert_constituents', 'insert_instructions',
        'insert_equipment', 'create_ingredient_counts',
        'filter_ingredient_names', 'filter_recipes',
        'create_all']
    parser.add_argument('--option', '-o', type=str, choices=options,
        default='create_all', help='What do you want to do?')
    # dropping tables
    parser.add_argument('--tables-to-drop', type=str,
        help='For option: drop_tables. Comma separated list of tables to drop')
    parser.add_argument('--exclude-tables-to-drop', type=str,
        help='For option: drop_tables. Comma separated list of tables not to drop')
    parser.add_argument('--force-drop',action='store_true',
        help='Force drop tables call on all tables, even if errors encountered?')
    # creating tables
    parser.add_argument('--tables-to-create', type=str,
        help='For option: create_tables. Comma separated list of tables to create')
    parser.add_argument('--exclude-tables-to-create', type=str,
        help='For option: create_tables. Comma separated list of tables to create')
    parser.add_argument('--force-create',action='store_true',
        help='Force create tables call on all tables, even if errors encountered?')
    # inserting recipes
    parser.add_argument('--input-subdir', type=str, default='processed',
        help='Input subdirectory for data')
    parser.add_argument('--max-ifile-id', type=int, default=15,
        help='Maximum input file id for processing')
    parser.add_argument('--min-ifile-id', type=int, default=1,
        help='Minimum input file id for processing')
    
    return parser

if __name__ == '__main__':
    import argparse
    args = create_parser().parse_args()
    main(**vars(args))    
    
    

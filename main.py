# Kor!
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number

# LangChain Models
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# Standard Helpers
import pandas as pd
import requests
import time
import os
import json
from datetime import datetime
from dotenv import load_dotenv
# Text Helpers
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# For token counting
from langchain.callbacks import get_openai_callback

from langchain.embeddings import OpenAIEmbeddings
from langchain.cache import UpstashRedisCache
from upstash_redis import Redis
import functools
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from langcorn import create_service

def printOutput(output):
    print(json.dumps(output, sort_keys=True, indent=3))

load_dotenv()  # Load the environment variables from the .env file

# Access the environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")

URL = os.environ.get("UPSTASH_REDIS_REST_URL")
TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN")

langchain.llm_cache = UpstashRedisCache(redis_=Redis(url=URL, token=TOKEN))

llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=1.3,
    max_tokens=2000,
    openai_api_key=openai_api_key
)

product_schema = Object(
    id="product",
    description="Information about a product",

    # Notice I put multiple fields to pull out different attributes
    attributes=[
        Text(
            id="Name",
            description="The name of the product."
        ),
        Text(
            id="SKU",
            description="Code used to identify the product"
        ),
        Text(
            id="Size",
            description="Size or sizes of product, usually integer, float, or number followed by letter eg 8.5W, but can be size in words"
        ),
        Number(
            id="Quantity",
            description="Amount of each size of the product, always default to 1 when none"
        ),
        Text(
            id="Cost",
            description="Cost of the product"
        ),
        Text(
            id="List Price",
            description="Price of an individual size"
        ),
        Text(
            id="Condition",
            description="Condition of individual size or overall product, db means damaged box, nb means no box, nl means no label, could be condition in words"
        )
    ],
    examples=[
        (
            """Women collection all brand new $200
    Bq6472-107 10.5w/9 men $60
    DH0210-100 10.5w/9 men
    Bq6472-202 10.5w/9 men $100
    Dh5894-600 10.5w/9 men db
    Dm9126-104 size 11w/9.5 men $60
    Cv5276-001 size 11w/9.5 men $120
    BQ6472-102 size 11w/9.5 men $110""",
            [
                {
                    "NAME": "Women collection all brand new",
                    "SKU": "Bq6472-107",
                    "SIZE": "10.5w/9 men",
                    "QUANTITY": "1",
                    "LIST PRICE": "$60",
                    "COST": "$200"
                },
                {
                    "NAME": "Women collection all brand new",
                    "SKU": "DH0210-100",
                    "SIZE": "10.5w/9 men",
                    "QUANTITY": "1",
                    "LIST PRICE": "None",
                    "COST": "$200"
                },
                {
                    "NAME": "Women collection all brand new",
                    "SKU": "Bq6472-202",
                    "SIZE": "10.5w/9 men",
                    "QUANTITY": "1",
                    "LIST PRICE": "$100",
                    "COST": "$200"
                },
                {
                    "NAME": "Women collection all brand new",
                    "SKU": "Dh5894-600",
                    "SIZE": "10.5w/9 men",
                    "QUANTITY": "1",
                    "LIST PRICE": "None",
                    "COST": "$200"
                },
                {
                    "NAME": "Women collection all brand new",
                    "SKU": "Dm9126-104",
                    "SIZE": "11w/9.5 men",
                    "QUANTITY": "1",
                    "LIST PRICE": "$60",
                    "COST": "$200"
                },
                {
                    "NAME": "Women collection all brand new",
                    "SKU": "Cv5276-001",
                    "SIZE": "11w/9.5 men",
                    "QUANTITY": "1",
                    "LIST PRICE": "$120",
                    "COST": "$200"
                },
                {
                    "NAME": "Women collection all brand new",
                    "SKU": "BQ6472-102",
                    "SIZE": "11w/9.5 men",
                    "QUANTITY": "1",
                    "LIST PRICE": "$110",
                    "COST": "$200"
                }
            ],
        ),
        (
            """924453-004 vapormax black  $110
8 db
8.5
9x2 2db
9.5x2
11x2
11.5x2
12""",
            [
                {
                    "NAME": " Vapormax Black",
                    "SKU": "924453-004",
                    "SIZE": "8",
                    "QUANTITY": "1",
                    "COST": "$110",
                    "LIST PRICE": "None",
                    "CONDITION": "Damaged box"
                },
                {
                    "NAME": " Vapormax Black",
                    "SKU": "924453-004",
                    "SIZE": "8.5",
                    "QUANTITY": "1",
                    "COST": "$110",
                    "LIST PRICE": "None"
                },
                {
                    "NAME": " Vapormax Black",
                    "SKU": "924453-004",
                    "SIZE": "9",
                    "QUANTITY": "2",
                    "COST": "$110",
                    "LIST PRICE": "None",
                    "CONDITION": "2 Damaged boxes"
                },
                {
                    "NAME": " Vapormax Black",
                    "SKU": "924453-004",
                    "SIZE": "9.5",
                    "QUANTITY": "2",
                    "COST": "$110",
                    "LIST PRICE": "None"
                },
                {
                    "NAME": " Vapormax Black",
                    "SKU": "924453-004",
                    "SIZE": "11",
                    "QUANTITY": "2",
                    "COST": "$110",
                    "LIST PRICE": "None"
                },
                {
                    "NAME": " Vapormax Black",
                    "SKU": "924453-004",
                    "SIZE": "11.5",
                    "QUANTITY": "2",
                    "COST": "$110",
                    "LIST PRICE": "None"
                },
                {
                    "NAME": " Vapormax Black",
                    "SKU": "924453-004",
                    "SIZE": "12",
                    "QUANTITY": "1",
                    "COST": "$110",
                    "LIST PRICE": "None"
                }

            ]
        ),
        (
            """DV9956-103 damage
9
AV2187-117 damage box 
10.5x3
CD9065-116 damage box 
6y
DQ4914-103 damaged 
5.5""",
            [
                {
                    "SHOE": "",
                    "SKU": "DV9956-103",
                    "SIZE": "9",
                    "QUANTITY": "1",
                    "COST": "",
                    "LIST PRICE": "",
                    "CONDITION": "damage"
                },
                {
                    "SHOE": "",
                    "SKU": "AV2187-117",
                    "SIZE": "10.5",
                    "QUANTITY": "3",
                    "COST": "",
                    "LIST PRICE": "",
                    "CONDITION": "damage box"
                },
                {
                    "SHOE": "",
                    "SKU": "CD9065-116",
                    "SIZE": "6y",
                    "QUANTITY": "1",
                    "COST": "",
                    "LIST PRICE": "",
                    "CONDITION": "damage box"
                },
                {
                    "SHOE": "",
                    "SKU": "DQ4914-103",
                    "SIZE": "5.5",
                    "QUANTITY": "1",
                    "COST": "",
                    "LIST PRICE": "",
                    "CONDITION": "damaged"
                }
            ]
        )

    ]
)


text = """"""

chain = create_extraction_chain(llm, product_schema)
output = chain.run(text=text)['data']

if __name__ == "__app__":
  printOutput(output)

# Extract the list of products from the "product" key
products = output.get("product", [])

if not products:
    print("No products found in the extracted data.")
else:
    # Define the order of columns in the Google Sheets document
    # The "Name" column will go to the "Shoe" column, and the rest are the same as the keys
    column_order = ['Shoe'] + [col for col in products[0] if col != 'Name']

    # Connect to your Google Sheets document
    gc = gspread.service_account(filename='secretkey.json')  # Replace with your JSON credentials file
    spreadsheet = gc.open('Inventory')  # Replace with your document name

    # Select the worksheet in your Google Sheets document
    worksheet = spreadsheet.get_worksheet(0)  # Replace with the index of your worksheet (0 for the first sheet)

    for product in products:
        # Preprocess the data (remove '$' from "List Price" and "Cost")
        list_price = product.get('List Price', '').replace('$', '')
        cost = product.get('Cost', '').replace('$', '')
    
        # Create a list to store the data for the current product in the order defined by column_order
        data_list = [product.get(col, '').replace('$', '') if col not in ['Shoe', 'List Price', 'Cost']
                     else list_price if col == 'List Price'
                     else cost if col == 'Cost'
                     else product.get('Name', '') for col in column_order]

        # Append a row for the current product
        worksheet.append_rows([data_list])

    print(f"{len(products)} rows added to Google Sheets for the products.")
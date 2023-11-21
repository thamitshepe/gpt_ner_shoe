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

from fastapi import FastAPI, HTTPException, Form
from langchain.embeddings import OpenAIEmbeddings
from langchain.cache import UpstashRedisCache
from upstash_redis import Redis
import functools
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Configure the FastAPI application
app = FastAPI()

load_dotenv()  # Load the environment variables from the .env file

# Access the environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")

URL = os.environ.get("UPSTASH_REDIS_REST_URL")
TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN")

langchain.llm_cache = UpstashRedisCache(redis_=Redis(url=URL, token=TOKEN))

scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
secret_key_filename = 'secretkey.json'  # Replace with your JSON credentials file


# Load Google Sheets credentials for the first sheet
creds_sheet1 = ServiceAccountCredentials.from_json_keyfile_name("secretkey.json", scopes=scopes)
file_sheet1 = gspread.authorize(creds_sheet1)
workbook_sheet1 = file_sheet1.open("WholeCell Inventory Template")  # Change to the actual name of your first sheet
sheet1 = workbook_sheet1.sheet1

# Load Google Sheets credentials for the second sheet
creds_sheet2 = ServiceAccountCredentials.from_json_keyfile_name("secretkey.json", scopes=scopes)
file_sheet2 = gspread.authorize(creds_sheet2)
workbook_sheet2 = file_sheet2.open("product catalog template")  # Change to the actual name of your second sheet
sheet2 = workbook_sheet2.sheet1

llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=1.1,
    max_tokens=3000,
    openai_api_key=openai_api_key
)

product_schema = Object(
    id="product",
    description="Information about a product",

    # Notice I put multiple fields to pull out different attributes
    attributes=[
        Text(
            id="Model",
            description="The name or model of the product. If none then put Sku in Model and Sku."
        ),
        Text(
            id="Manufacturer",
            description="The manufacturer of the product. Use the Model to figure out who manufactured the product, if unknown use Nike"
        ),
        Text(
            id="Code",
            description="Take 2 letters from manufacturer, ie say if Nike, then NK, put -BR in front of it, then take the capacity and sku, write the three one after another split by underscore and dash eg(NK-BR_7.5-XXXXXXX)."
        ),
        Text(
            id="Sku",
            description="Code used to identify the product, if none the generate a random one with a random 3 digit number and always starting with GEN, for instance GEN034, number always random"
        ),
        Text(
            id="Capacity",
            description="Capacity/Size or sizes of product, usually integer, float, or number followed by letter eg 8.5W, but can be size in words"
        ),
        Number(
            id="Quantity",
            description="Quantity of each capacity/size of the product, always default to 1 when none"
        ),
        Text(
            id="Price Paid",
            description="Price paid for the product, if none leave empty"
        ),
        Text(
            id="Cost",
            description="Price of an individual capacity/size, if none leave empty"
        ),
        Text(
            id="Grade",
            description="Grade of individual capacity/size or overall product, either new or used, if none then New"
        ),
        Text(
            id="Damages",
            description="Damages or condition of individual capacity/size or overall product, db means damaged box, nb means no box, nl means no label, could be condition in words, if none leave empty"
        ),
        Text(
            id="Complete",
            description="This is just to show that a product was addded/a deal was complete, the value is always Complete"
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
                "MODEL": "Women collection all brand new",
                "MANUFACTURER": "Nike",
                "CODE": "NK-BR_10.5w/9 men-Bq6472-107",
                "SKU": "Bq6472-107",
                "CAPACITY": "10.5w/9 men",
                "QUANTITY": "1",
                "COST": "$60",
                "PRICE PAID": "$200",
                "GRADE": "New",
                "DAMAGES": "",
                "COMPLETE": "Complete"
            },
            {
                "MODEL": "Women collection all brand new",
                "MANUFACTURER": "Nike",
                "CODE": "NK-BR_10.5w/9 men-Bq6472-107",
                "SKU": "DH0210-100",
                "CAPACITY": "10.5w/9 men",
                "QUANTITY": "1",
                "COST": "None",
                "PRICE PAID": "$200",
                "GRADE": "New",
                "DAMAGES": "",
                "COMPLETE": "Complete"
            },
            {
                "MODEL": "Women collection all brand new",
                "MANUFACTURER": "Nike",
                "CODE": "NK-BR_10.5w/9 men-Bq6472-107",
                "SKU": "Bq6472-202",
                "CAPACITY": "10.5w/9 men",
                "QUANTITY": "1",
                "COST": "$100",
                "PRICE PAID": "$200",
                "GRADE": "New",
                "DAMAGES": "",
                "COMPLETE": "Complete"
            },
            {
                "MODEL": "Women collection all brand new",
                "MANUFACTURER": "Nike",
                "CODE": "NK-BR_10.5w/9 men-Bq6472-107",
                "SKU": "Dh5894-600",
                "CAPACITY": "10.5w/9 men",
                "QUANTITY": "1",
                "COST": "None",
                "PRICE PAID": "$200",
                "GRADE": "New",
                "DAMAGES": "1 Damaged Box",
                "COMPLETE": "Complete"
            },
            {
                "MODEL": "Women collection all brand new",
                "MANUFACTURER": "Nike",
                "CODE": "NK-BR_11w/9.5 men-Bq6472-107",
                "SKU": "Dm9126-104",
                "CAPACITY": "11w/9.5 men",
                "QUANTITY": "1",
                "COST": "$60",
                "PRICE PAID": "$200",
                "GRADE": "New",
                "DAMAGES": "",
                "COMPLETE": "Complete"
            },
            {
                "MODEL": "Women collection all brand new",
                "MANUFACTURER": "Nike",
                "CODE": "NK-BR_11w/9.5 men-Bq6472-107",
                "SKU": "Cv5276-001",
                "CAPACITY": "11w/9.5 men",
                "QUANTITY": "1",
                "COST": "$120",
                "PRICE PAID": "$200",
                "GRADE": "New",
                "DAMAGES": "",
                "COMPLETE": "Complete"
            },
            {
                "MODEL": "Women collection all brand new",
                "MANUFACTURER": "Nike",
                "CODE": "NK-BR_11w/9.5 men-Bq6472-107",
                "SKU": "BQ6472-102",
                "CAPACITY": "11w/9.5 men",
                "QUANTITY": "1",
                "COST": "$110",
                "PRICE PAID": "$200",
                "GRADE": "New",
                "DAMAGES": "",
                "COMPLETE": "Complete"
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
                "MODEL": " Vapormax Black",
                "MANUFACTURER": "Nike",
                "CODE": "NK-BR_8-924453-004",
                "SKU": "924453-004",
                "CAPACITY": "8",
                "QUANTITY": "1",
                "PRICE PAID": "$110",
                "COST": "",
                "GRADE": "New",
                "DAMAGES": "1 Damaged box",
                "COMPLETE": "Complete"
            },
            {
                "MODEL": " Vapormax Black",
                "MANUFACTURER": "Nike",
                "CODE": "NK-BR_8.5-924453-004",
                "SKU": "924453-004",
                "CAPACITY": "8.5",
                "QUANTITY": "1",
                "PRICE PAID": "$110",
                "COST": "",
                "COMPLETE": "Complete"
            },
            {
                "MODEL": " Vapormax Black",
                "MANUFACTURER": "Nike",
                "CODE": "NK-BR_9-924453-004",
                "SKU": "924453-004",
                "CAPACITY": "9",
                "QUANTITY": "2",
                "PRICE PAID": "$110",
                "COST": "",
                "GRADE": "New",
                "DAMAGES": "2 Damaged boxes",
                "COMPLETE": "Complete"
            },
            {
                "MODEL": " Vapormax Black",
                "MANUFACTURER": "Nike",
                "CODE": "NK-BR_9.5-924453-004",
                "SKU": "924453-004",
                "CAPACITY": "9.5",
                "QUANTITY": "2",
                "PRICE PAID": "$110",
                "COST": "",
                "GRADE": "New",
                "DAMAGES": "",
                "COMPLETE": "Complete"
            },
            {
                "MODEL": " Vapormax Black",
                "MANUFACTURER": "Nike",
                "CODE": "NK-BR_11-924453-004",
                "SKU": "924453-004",
                "CAPACITY": "11",
                "QUANTITY": "2",
                "PRICE PAID": "$110",
                "COST": "",
                "GRADE": "New",
                "DAMAGES": "",
                "COMPLETE": "Complete"
            },
            {
                "MODEL": " Vapormax Black",
                "MANUFACTURER": "Nike",
                "CODE": "NK-BR_11.5-924453-004",
                "SKU": "924453-004",
                "CAPACITY": "11.5",
                "QUANTITY": "2",
                "PRICE PAID": "$110",
                "COST": "",
                "GRADE": "New",
                "DAMAGES": "1 Damaged box",
                "COMPLETE": "Complete"
            },
            {
                "MODEL": " Vapormax Black",
                "MANUFACTURER": "Nike",
                "CODE": "NK-BR_12-924453-004",
                "SKU": "924453-004",
                "CAPACITY": "12",
                "QUANTITY": "1",
                "PRICE PAID": "$110",
                "COST": "",
                "GRADE": "New",
                "DAMAGES": "1 Damaged box",
                "COMPLETE": "Complete"
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
                "MODEL": "DV9956-103",
                "MANUFACTURER": "Nike",
                "CODE": "NK-BR_9-DV9956-103",
                "SKU": "DV9956-103",
                "CAPACITY": "9",
                "QUANTITY": "1",
                "PRICE PAID": "",
                "COST": "",
                "GRADE": "Used",
                "DAMAGES": "Damage",
                "COMPLETE": "Complete"
            },
            {
                "MODEL": "AV2187-117",
                "MANUFACTURER": "Nike",
                "CODE": "NK-BR_10.5-DV9956-103",
                "SKU": "AV2187-117",
                "CAPACITY": "10.5",
                "QUANTITY": "3",
                "PRICE PAID": "",
                "COST": "",
                "GRADE": "New",
                "DAMAGES": "1 Damaged box",
                "COMPLETE": "Complete"
            },
            {
                "MODEL": "CD9065-116",
                "MANUFACTURER": "Nike",
                "CODE": "NK-BR_6y-DV9956-103",
                "SKU": "CD9065-116",
                "CAPACITY": "6y",
                "QUANTITY": "1",
                "PRICE PAID": "",
                "COST": "",
                "GRADE": "New",
                "DAMAGES": "1 Damaged box",
                "COMPLETE": "Complete"
            },
            {
                "MODEL": "DQ4914-103",
                "MANUFACTURER": "Nike",
                "CODE": "NK-BR_5.5-DV9956-103",
                "SKU": "DQ4914-103",
                "CAPACITY": "5.5",
                "QUANTITY": "1",
                "PRICE PAID": "",
                "COST": "",
                "GRADE": "Used",
                "DAMAGES": "Damaged",
                "COMPLETE": "Complete"
            }
        ]
    )
]

)


def printOutput(output):
    return json.dumps(output, sort_keys=True, indent=3)

def extract_and_store_data(text):
    # Create an extraction chain using the LangChain LLM and the product schema
    chain = create_extraction_chain(llm, product_schema)
    
    # Run the extraction chain on the input text
    output = chain.run(text=text)['data']
    
    # Extract the list of products from the "product" key
    products = output.get("product", [])

    if not products:
        return "No products found in the extracted data."

    # Get the column names from the first row of both worksheets
    header_row_wholecell = sheet1.row_values(1)
    header_row_catalog = sheet2.row_values(1)

    for product in products:
        # Create empty dictionaries to store data for the current product in both sheets
        data_dict_wholecell = {}
        data_dict_catalog = {}

        # Define the keys you want in data_dict for each sheet
        keys_to_extract_wholecell = ['Model', 'Manufacturer', 'Cost', 'Price Paid', 'Sku', 'Grade', 'Damages', 'Capacity', 'Quantity', 'Complete']
        keys_to_extract_catalog = ['Model', 'Code', 'Manufacturer', 'Sku', 'Grade', 'Damages', 'Capacity']

        # Iterate through the keys and add them to data_dict for each sheet if they exist in the product
        for key in keys_to_extract_wholecell:
            if key in product:
                data_dict_wholecell[key] = product[key]

        for key in keys_to_extract_catalog:
            if key in product:
                data_dict_catalog[key] = product[key]

        # Append a row for the current product in both sheets
        data_list_wholecell = [data_dict_wholecell.get(col, '') for col in header_row_wholecell]
        sheet1.append_rows([data_list_wholecell])

        data_list_catalog = [data_dict_catalog.get(col, '') for col in header_row_catalog]
        sheet2.append_rows([data_list_catalog])

    # Return the message about the number of rows added to both sheets
    result_message = f"{len(products)} rows added to 'WholeCell Inventory Template' and 'Product Catalog Template'."

    # Return the result message
    return result_message

@app.post("/aishoe/")
async def process_text(text: str):
    try:
        result = extract_and_store_data(text)
        return {"message": result}
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

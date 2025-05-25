import pandas as pd
import numpy as np
from faker import Faker
from sklearn.preprocessing import LabelEncoder
import random
import os

fake = Faker()

# deepseek_api_key="sk-195567327fa14246975a5c1084ad587e"

# --- User Input ---
csv_file = input("Enter path to your .csv file: ")
desired_rows = int(input("Enter total number of rows you want: "))

# --- Load Existing Data ---
df = pd.read_csv(csv_file)
print(f"Original dataset shape: {df.shape}")

# --- Define Required Columns ---
REQUIRED_COLUMNS = ['flight_number', 'airline', 'airplane', 'departure_city', 'arrival_city',
                    'departure_time', 'arrival_time', 'date', 'class_type', 'sold_count', 'price']

# --- Identify Missing Columns ---
missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
print(f"Missing columns: {missing_cols}")

# --- Function to Generate Fake Values ---
def generate_fake_value(column):
    if column == 'flight_number':
        return f"FN{random.randint(1000, 9999)}"
    elif column == 'airline':
        return random.choice(['AirX', 'JetStream', 'SkyWing', 'TransGlobal'])
    elif column == 'airplane':
        return random.choice(['Boeing 737', 'Airbus A320', 'Embraer E190'])
    elif column == 'departure_city':
        return fake.city()
    elif column == 'arrival_city':
        return fake.city()
    elif column == 'departure_time':
        return fake.time()
    elif column == 'arrival_time':
        return fake.time()
    elif column == 'date':
        return fake.date_this_year()
    elif column == 'class_type':
        return random.choice(['ECONOMY', 'BUSINESS', 'FIRST'])
    elif column == 'sold_count':
        return random.randint(0, 150)
    elif column == 'price':
        return round(random.uniform(50, 1500), 2)
    else:
        return None

# --- Generate New Rows ---
new_rows = []
existing_rows = df.shape[0]
rows_to_add = max(0, desired_rows - existing_rows)

for _ in range(rows_to_add):
    row = {}
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            row[col] = np.nan  # will fill later if needed
        else:
            row[col] = generate_fake_value(col)
    new_rows.append(row)

# --- Append & Fill Missing Values in Existing Rows ---
df = df.reindex(columns=REQUIRED_COLUMNS)
df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

# Fill missing values for existing rows where possible
for col in REQUIRED_COLUMNS:
    if df[col].isnull().any():
        df[col] = df[col].apply(lambda x: generate_fake_value(col) if pd.isnull(x) else x)

# --- Save Output ---
output_file = os.path.splitext(csv_file)[0] + "_filled.csv"
df.to_csv(output_file, index=False)
print(f"✅ Generated dataset saved to: {output_file}")

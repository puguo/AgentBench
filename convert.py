import csv
import datetime
import os

def convert_timestamp(file_name):
    def parse_timestamp(timestamp_str):
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.datetime.strptime(timestamp_str, fmt)
            except ValueError:
                raise ValueError(f"time data '{timestamp_str}' not identified.")

    with open(file_name, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    for row in rows:
        if row:
            try:
                timestamp_str = row[0]
                timestamp_dt = parse_timestamp(timestamp_str)
                row[0] = str(timestamp_dt.timestamp())
            except ValueError as e:
                print(e)
    new_file_name = f"converted_{os.path.basename(file_name)}"
    with open(new_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

convert_timestamp('WS_logging.csv')

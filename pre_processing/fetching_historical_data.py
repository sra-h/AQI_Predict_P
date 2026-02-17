import requests
import csv
from datetime import datetime, timedelta
import time
import calendar
import pytz

# Configuration
API_KEY = "" # 
LATITUDE = 32.0849  
LONGITUDE = 72.6890 

# Pakistan Standard Time (UTC+5)
PKT = pytz.timezone('Asia/Karachi')
UTC = pytz.UTC

# Date range in Pakistan Standard Time
START_DATE_PKT = PKT.localize(datetime(2023, 1, 1, 0, 0, 0))
END_DATE_PKT = PKT.localize(datetime(2026, 2, 3, 23, 0, 0))

# Convert to UTC for API calls (OpenWeather uses UTC)
START_DATE_UTC = START_DATE_PKT.astimezone(UTC)
END_DATE_UTC = END_DATE_PKT.astimezone(UTC)

# OpenWeather Air Pollution API endpoint
BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"

def fetch_aqi_data(lat, lon, start_timestamp, end_timestamp, api_key):
    """
    Fetch historical AQI data from OpenWeather API
    """
    params = {
        'lat': lat,
        'lon': lon,
        'start': start_timestamp,
        'end': end_timestamp,
        'appid': api_key
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def analyze_gaps(data_list):
    """
    Analyze and identify all gaps in hourly data
    Returns list of gap details
    """
    if len(data_list) < 2:
        return []

    gaps = []

    for i in range(1, len(data_list)):
        prev_timestamp = data_list[i-1]['dt']
        curr_timestamp = data_list[i]['dt']
        time_diff = curr_timestamp - prev_timestamp

        # If gap is more than 1 hour (allowing 100 second tolerance)
        if time_diff > 3700:
            prev_dt_utc = datetime.fromtimestamp(prev_timestamp, tz=UTC)
            curr_dt_utc = datetime.fromtimestamp(curr_timestamp, tz=UTC)
            prev_dt_pkt = prev_dt_utc.astimezone(PKT)
            curr_dt_pkt = curr_dt_utc.astimezone(PKT)

            gap_hours = (time_diff / 3600) - 1

            gaps.append({
                'from_utc': prev_dt_utc,
                'to_utc': curr_dt_utc,
                'from_pkt': prev_dt_pkt,
                'to_pkt': curr_dt_pkt,
                'hours_missing': gap_hours,
                'prev_timestamp': prev_timestamp,
                'curr_timestamp': curr_timestamp
            })

    return gaps

def generate_complete_hourly_range(start_dt_utc, end_dt_utc):
    """
    Generate all expected hourly timestamps in the range
    """
    expected_timestamps = []
    current = start_dt_utc.replace(minute=0, second=0, microsecond=0)
    end = end_dt_utc.replace(minute=0, second=0, microsecond=0)

    while current <= end:
        expected_timestamps.append(int(current.timestamp()))
        current += timedelta(hours=1)

    return expected_timestamps

def merge_data_with_gaps(api_data, expected_timestamps):
    """
    Merge API data with expected timestamps, filling gaps with null values
    """
    # Create a dictionary of actual data keyed by timestamp
    actual_data_dict = {}
    for item in api_data:
        actual_data_dict[item['dt']] = item

    # Create complete dataset with nulls for missing data
    complete_data = []

    for timestamp in expected_timestamps:
        if timestamp in actual_data_dict:
            # Actual data exists
            complete_data.append({
                'timestamp': timestamp,
                'data': actual_data_dict[timestamp],
                'is_gap': False
            })
        else:
            # Gap - create null entry
            complete_data.append({
                'timestamp': timestamp,
                'data': None,
                'is_gap': True
            })

    return complete_data

def save_to_csv_with_gaps(complete_data, filename='aqi_data_complete.csv'):
    """
    Save AQI data to CSV file with gaps filled as null
    """
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'datetime_pkt', 'datetime_utc', 'timestamp', 'is_gap',
            'aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for entry in complete_data:
            timestamp = entry['timestamp']
            dt_utc = datetime.fromtimestamp(timestamp, tz=UTC)
            dt_pkt = dt_utc.astimezone(PKT)

            if entry['is_gap']:
                # Gap entry - all values are null
                row = {
                    'datetime_pkt': dt_pkt.strftime('%Y-%m-%d %H:%M:%S'),
                    'datetime_utc': dt_utc.strftime('%Y-%m-%d %H:%M:%S'),
                    'timestamp': timestamp,
                    'is_gap': 'TRUE',
                    'aqi': 'null',
                    'co': 'null',
                    'no': 'null',
                    'no2': 'null',
                    'o3': 'null',
                    'so2': 'null',
                    'pm2_5': 'null',
                    'pm10': 'null',
                    'nh3': 'null'
                }
            else:
                # Actual data
                item = entry['data']
                row = {
                    'datetime_pkt': dt_pkt.strftime('%Y-%m-%d %H:%M:%S'),
                    'datetime_utc': dt_utc.strftime('%Y-%m-%d %H:%M:%S'),
                    'timestamp': timestamp,
                    'is_gap': 'FALSE',
                    'aqi': item['main']['aqi'],
                    'co': item['components'].get('co', ''),
                    'no': item['components'].get('no', ''),
                    'no2': item['components'].get('no2', ''),
                    'o3': item['components'].get('o3', ''),
                    'so2': item['components'].get('so2', ''),
                    'pm2_5': item['components'].get('pm2_5', ''),
                    'pm10': item['components'].get('pm10', ''),
                    'nh3': item['components'].get('nh3', '')
                }

            writer.writerow(row)

    print(f"\n✓ Data saved to {filename}")

def print_gap_analysis(gaps, total_missing):
    """
    Print detailed gap analysis
    """
    if not gaps:
        print("\n✓ No gaps found! All hourly data is present.")
        return

    print(f"\n{'='*80}")
    print(f"GAP ANALYSIS - Found {len(gaps)} gaps totaling {total_missing} missing hours")
    print(f"{'='*80}")

    print("\nAll gaps (Pakistan Standard Time):")
    print(f"{'#':<5} {'From (PKT)':<22} {'To (PKT)':<22} {'Missing Hours':<15}")
    print("-" * 80)

    for idx, gap in enumerate(gaps, 1):
        from_str = gap['from_pkt'].strftime('%Y-%m-%d %H:%M:%S')
        to_str = gap['to_pkt'].strftime('%Y-%m-%d %H:%M:%S')
        print(f"{idx:<5} {from_str:<22} {to_str:<22} {gap['hours_missing']:<15.0f}")

    print("-" * 80)
    print(f"Total missing hours: {total_missing}")
    print(f"{'='*80}\n")

def main():
    """
    Main function to fetch and save AQI data with gap filling
    """
    print("="*80)
    print("FETCHING HISTORICAL AQI DATA")
    print("="*80)
    print(f"Location: Latitude {LATITUDE}, Longitude {LONGITUDE}")
    print(f"Date Range (Pakistan Time):")
    print(f"  Start: {START_DATE_PKT.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  End:   {END_DATE_PKT.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"\nDate Range (UTC - API uses this):")
    print(f"  Start: {START_DATE_UTC.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  End:   {END_DATE_UTC.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Generate expected hourly timestamps
    print("\nGenerating expected hourly timestamps...")
    expected_timestamps = generate_complete_hourly_range(START_DATE_UTC, END_DATE_UTC)
    print(f"Expected total hours: {len(expected_timestamps)}")

    # Fetch data year by year (handling leap years)
    current_start = START_DATE_UTC
    all_data = []

    while current_start < END_DATE_UTC:
        # Get end of current calendar year in UTC
        current_year = current_start.year
        year_end_utc = UTC.localize(datetime(current_year, 12, 31, 23, 59, 59))
        current_end = min(year_end_utc, END_DATE_UTC)

        start_timestamp = int(current_start.timestamp())
        end_timestamp = int(current_end.timestamp())

        is_leap = calendar.isleap(current_year)
        leap_indicator = "🔁 LEAP YEAR" if is_leap else ""

        print(f"\n{'='*80}")
        print(f"Fetching Year {current_year} {leap_indicator}")
        print(f"From: {current_start.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"To:   {current_end.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        data = fetch_aqi_data(LATITUDE, LONGITUDE, start_timestamp, end_timestamp, API_KEY)

        if data and 'list' in data:
            records = len(data['list'])
            all_data.extend(data['list'])
            print(f"Retrieved: {records} records")

            if records > 0:
                first = datetime.fromtimestamp(data['list'][0]['dt'], tz=UTC)
                last = datetime.fromtimestamp(data['list'][-1]['dt'], tz=UTC)
                print(f"First: {first.astimezone(PKT).strftime('%Y-%m-%d %H:%M:%S PKT')}")
                print(f"Last:  {last.astimezone(PKT).strftime('%Y-%m-%d %H:%M:%S PKT')}")
        else:
            print(" No data received for this period")

        # Move to start of next year
        current_start = UTC.localize(datetime(current_year + 1, 1, 1, 0, 0, 0))

        # Rate limiting
        time.sleep(1)

    print(f"\n{'='*80}")
    print("DATA COLLECTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total records retrieved from API: {len(all_data)}")
    print(f"Expected hourly records: {len(expected_timestamps)}")
    print(f"Missing records: {len(expected_timestamps) - len(all_data)}")

    # Analyze gaps
    print("\nAnalyzing data for gaps...")
    gaps = analyze_gaps(all_data)
    total_missing_hours = sum(gap['hours_missing'] for gap in gaps)

    # Print gap analysis
    print_gap_analysis(gaps, int(total_missing_hours))

    # Merge data with gaps
    print("Merging data with gaps (filling missing hours with null)...")
    complete_data = merge_data_with_gaps(all_data, expected_timestamps)

    gap_count = sum(1 for entry in complete_data if entry['is_gap'])
    actual_count = sum(1 for entry in complete_data if not entry['is_gap'])

    print(f"Complete dataset prepared:")
    print(f"  Total rows: {len(complete_data)}")
    print(f"  Actual data: {actual_count}")
    print(f"  Gap rows (null): {gap_count}")

    # Save to CSV
    print("\nSaving to CSV...")
    save_to_csv_with_gaps(complete_data, 'aqi_data_complete_with_gaps.csv')

    print("\n" + "="*80)
    print("PROCESS COMPLETE!")
    print("="*80)
    print(f"✓ CSV file created: aqi_data_complete_with_gaps.csv")
    print(f"✓ Total rows: {len(complete_data)}")
    print(f"✓ Gaps filled with 'null' values")
    print(f"✓ All timestamps in Pakistan Standard Time")
    print("="*80)

if __name__ == "__main__":
    main()

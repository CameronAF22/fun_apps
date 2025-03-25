import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import csv
from io import StringIO

# Set page config
st.set_page_config(
    page_title="TDY Cost Calculator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve layout
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #ccc;
        cursor: help;
    }
    .small-text {
        font-size: 0.8em;
        color: #666;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def get_excel_path():
    """Get the path to the Excel file, checking multiple possible locations."""
    possible_paths = [
        "3-TDY Calculator FY24.xlsm",  # Current directory
        os.path.join("tdy_cost_app", "3-TDY Calculator FY24.xlsm"),  # tdy_cost_app subdirectory
        os.path.join("..", "3-TDY Calculator FY24.xlsm"),  # Parent directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def clean_dataframe(df):
    """Clean and prepare a dataframe for processing."""
    # Drop completely empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Convert all string columns to string type, handling NaN values
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).replace('nan', '')
    
    return df

# Read Excel file
@st.cache_data
def load_excel_data():
    try:
        excel_file = get_excel_path()
        if excel_file is None:
            st.error("Could not find the Excel file. Please ensure it's in the correct location.")
            return None
            
        # Read all sheets to find the factors table
        xls = pd.ExcelFile(excel_file)
        sheets_dict = {}
        
        # Display available sheets for debugging
        st.sidebar.subheader("Available Sheets")
        st.sidebar.write(xls.sheet_names)
        
        # First, try to identify the main rates sheet
        rates_sheet = None
        for sheet_name in xls.sheet_names:
            # Read the sheet
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                # Look for rate-related columns
                columns = [str(col).lower() for col in df.columns]
                if any('rate' in col or 'lodging' in col or 'meal' in col or 'location' in col for col in columns):
                    rates_sheet = sheet_name
                    break
            except Exception as e:
                st.warning(f"Warning: Could not check sheet '{sheet_name}': {str(e)}")
        
        if rates_sheet:
            st.sidebar.success(f"Found rates in sheet: {rates_sheet}")
            try:
                # Read the rates sheet first
                df = pd.read_excel(excel_file, sheet_name=rates_sheet)
                sheets_dict[rates_sheet] = clean_dataframe(df)
                
                # Read other sheets for reference
                for sheet_name in xls.sheet_names:
                    if sheet_name != rates_sheet:
                        try:
                            df = pd.read_excel(excel_file, sheet_name=sheet_name)
                            sheets_dict[sheet_name] = clean_dataframe(df)
                        except Exception as e:
                            st.warning(f"Warning: Could not read sheet '{sheet_name}': {str(e)}")
            except Exception as e:
                st.error(f"Error reading rates sheet: {str(e)}")
                return None
        else:
            st.error("Could not find a rates sheet in the Excel file. Available sheets:")
            for sheet_name in xls.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    st.write(f"Sheet: {sheet_name}")
                    st.write("Columns:", list(df.columns))
                except Exception as e:
                    st.warning(f"Could not read sheet '{sheet_name}': {str(e)}")
            return None
            
        return sheets_dict
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return None

def get_rates_and_factors(data):
    # Extract rates and factors from the sheets
    rates = {}
    factors = {
        'first_day': 0.75,  # 75% of meals for first day
        'last_day': 0.75    # 75% of meals for last day
    }
    
    # Find the rates sheet
    for sheet_name, df in data.items():
        if 'rate' in sheet_name.lower() or 'location' in sheet_name.lower():
            # Try to identify the location column
            location_col = None
            for col in df.columns:
                col_str = str(col).lower()
                if 'location' in col_str or 'city' in col_str or 'state' in col_str:
                    location_col = col
                    break
            
            if location_col is None and len(df.columns) > 0:
                # Assume first non-empty column is location
                for col in df.columns:
                    if df[col].notna().any():
                        location_col = col
                        break
            
            if location_col is not None:
                # Convert rates dataframe to dictionary
                for idx, row in df.iterrows():
                    if pd.notna(row[location_col]):
                        location = str(row[location_col]).strip()
                        if location and location.lower() not in ['location', 'city', 'state']:  # Skip header rows
                            rate_data = {}
                            
                            # Look for rate columns
                            for col in df.columns:
                                col_str = str(col).lower()
                                if 'lodging' in col_str:
                                    rate_data['lodging'] = pd.to_numeric(row[col], errors='coerce') or 0
                                elif 'meal' in col_str or 'm&ie' in col_str:
                                    rate_data['meals'] = pd.to_numeric(row[col], errors='coerce') or 0
                                elif 'incidental' in col_str:
                                    rate_data['incidentals'] = pd.to_numeric(row[col], errors='coerce') or 0
                            
                            if rate_data and any(rate_data.values()):  # Only add if there are non-zero rates
                                rates[location] = rate_data
    
    # Display the data structure in debug section
    with st.expander("Debug Info", expanded=False):
        st.json(rates)
        for sheet_name, df in data.items():
            st.subheader(f"Sheet: {sheet_name}")
            st.dataframe(df)
    
    return rates, factors

def calculate_tdy_cost(location, start_date, end_date, rates, factors):
    if location not in rates:
        return {
            'lodging': 0,
            'meals': 0,
            'incidentals': 0,
            'total': 0,
            'error': 'Location not found in rates table'
        }
    
    # Calculate number of days
    days = (end_date - start_date).days + 1
    
    location_rates = rates[location]
    
    # Calculate daily costs
    lodging_rate = location_rates.get('lodging', 0)
    meals_rate = location_rates.get('meals', 0)
    incidentals_rate = location_rates.get('incidentals', 0)
    
    # Calculate meals with first/last day adjustments
    if days == 1:
        total_meals = meals_rate * factors['first_day']
    else:
        first_day_meals = meals_rate * factors['first_day']
        last_day_meals = meals_rate * factors['last_day']
        full_days_meals = meals_rate * (days - 2) if days > 2 else 0
        total_meals = first_day_meals + full_days_meals + last_day_meals
    
    # Calculate other totals
    total_lodging = lodging_rate * days
    total_incidentals = incidentals_rate * days
    
    total = total_lodging + total_meals + total_incidentals
    
    return {
        'days': days,
        'start_date': start_date.strftime('%m/%d/%Y'),
        'end_date': end_date.strftime('%m/%d/%Y'),
        'location': location,
        'lodging': lodging_rate,
        'lodging_total': total_lodging,
        'meals': meals_rate,
        'meals_total': total_meals,
        'incidentals': incidentals_rate,
        'incidentals_total': total_incidentals,
        'total': total,
        'error': None,
        'first_day_meals': meals_rate * factors['first_day'] if days > 0 else 0,
        'last_day_meals': meals_rate * factors['last_day'] if days > 1 else 0,
        'full_days_meals': meals_rate * (days - 2) if days > 2 else 0
    }

def format_currency(amount):
    return f"${amount:,.2f}"

def create_csv_string(results):
    """Create a CSV string from the results dictionary."""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['TDY Cost Calculation Results'])
    writer.writerow([])
    writer.writerow(['Trip Information'])
    writer.writerow(['Location', results['location']])
    writer.writerow(['Start Date', results['start_date']])
    writer.writerow(['End Date', results['end_date']])
    writer.writerow(['Duration (days)', results['days']])
    writer.writerow([])
    writer.writerow(['Daily Rates'])
    writer.writerow(['Lodging', format_currency(results['lodging'])])
    writer.writerow(['Meals', format_currency(results['meals'])])
    writer.writerow(['Incidentals', format_currency(results['incidentals'])])
    writer.writerow([])
    writer.writerow(['Total Breakdown'])
    writer.writerow(['Lodging Total', format_currency(results['lodging_total'])])
    if results['days'] == 1:
        writer.writerow(['Meals (Single Day - 75%)', format_currency(results['first_day_meals'])])
    else:
        writer.writerow(['Meals (First Day - 75%)', format_currency(results['first_day_meals'])])
        if results['days'] > 2:
            writer.writerow(['Meals (Full Days)', format_currency(results['full_days_meals'])])
        writer.writerow(['Meals (Last Day - 75%)', format_currency(results['last_day_meals'])])
    writer.writerow(['Meals Total', format_currency(results['meals_total'])])
    writer.writerow(['Incidentals Total', format_currency(results['incidentals_total'])])
    writer.writerow([])
    writer.writerow(['Final Total', format_currency(results['total'])])
    
    return output.getvalue()

def main():
    st.title("TDY Cost Calculator")
    
    # Add help text in sidebar
    with st.sidebar:
        st.header("Help & Information")
        st.markdown("""
        ### About This Calculator
        This tool helps estimate TDY (Temporary Duty) travel costs based on official rates.
        
        ### Key Features
        * Location-based rate lookup
        * Automatic meal proration
        * Date range calculation
        * Detailed cost breakdown
        
        ### Rate Information
        * **Lodging Rate**: Daily lodging allowance
        * **Meals Rate**: Daily meals allowance
        * **Incidentals**: Daily incidental expenses
        
        ### Meal Proration
        * First Day: 75% of meals rate
        * Last Day: 75% of meals rate
        * Full Days: 100% of meals rate
        """)
    
    # Load data
    data = load_excel_data()
    
    if data is None:
        return

    # Get rates and factors
    rates, factors = get_rates_and_factors(data)
    
    # Create columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        
        # Create a sorted list of locations
        locations = sorted(list(rates.keys()))
        if locations:
            # Add location search
            search = st.text_input("Search Locations", "", help="Type to filter locations")
            filtered_locations = [loc for loc in locations if search.lower() in loc.lower()]
            
            if len(filtered_locations) > 0:
                location = st.selectbox("Select Location", filtered_locations)
                
                # Date selection
                today = datetime.now().date()
                start_date = st.date_input("Start Date", today, help="First day of TDY")
                end_date = st.date_input("End Date", today, help="Last day of TDY")
                
                if start_date > end_date:
                    st.error("End date must be after start date")
                else:
                    if st.button("Calculate"):
                        results = calculate_tdy_cost(location, start_date, end_date, rates, factors)
                        
                        with col2:
                            st.subheader("Results")
                            if results['error']:
                                st.error(results['error'])
                            else:
                                # Create a summary card
                                st.info(f"""
                                **Trip Summary**
                                * Location: {results['location']}
                                * Dates: {results['start_date']} - {results['end_date']}
                                * Duration: {results['days']} days
                                * Total Cost: {format_currency(results['total'])}
                                """)
                                
                                st.write("Daily Rates:")
                                st.write(f"Lodging: {format_currency(results['lodging'])}")
                                st.write(f"Meals & Incidentals (Full Day): {format_currency(results['meals'] + results['incidentals'])}")
                                st.write("---")
                                st.write(f"Total Breakdown ({results['days']} days):")
                                st.write(f"Lodging: {format_currency(results['lodging_total'])}")
                                
                                # Detailed meals breakdown
                                st.write("Meals Breakdown:")
                                if results['days'] == 1:
                                    st.write(f"  Single Day (75%): {format_currency(results['first_day_meals'])}")
                                else:
                                    st.write(f"  First Day (75%): {format_currency(results['first_day_meals'])}")
                                    if results['days'] > 2:
                                        st.write(f"  Full Days: {format_currency(results['full_days_meals'])}")
                                    st.write(f"  Last Day (75%): {format_currency(results['last_day_meals'])}")
                                st.write(f"Total Meals: {format_currency(results['meals_total'])}")
                                
                                st.write(f"Incidentals: {format_currency(results['incidentals_total'])}")
                                st.write("---")
                                st.write(f"Total: {format_currency(results['total'])}")
                                
                                # Add download buttons
                                col3, col4 = st.columns(2)
                                with col3:
                                    st.download_button(
                                        "Download as JSON",
                                        data=json.dumps(results, indent=2),
                                        file_name=f"tdy_cost_{results['start_date'].replace('/', '')}.json",
                                        mime="application/json",
                                        help="Download the calculation results as a JSON file"
                                    )
                                with col4:
                                    st.download_button(
                                        "Download as CSV",
                                        data=create_csv_string(results),
                                        file_name=f"tdy_cost_{results['start_date'].replace('/', '')}.csv",
                                        mime="text/csv",
                                        help="Download the calculation results as a CSV file"
                                    )
            else:
                st.warning("No locations found matching your search.")
        else:
            st.error("No locations found in the rates table")

if __name__ == "__main__":
    main()

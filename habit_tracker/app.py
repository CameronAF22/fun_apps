import streamlit as st
from datetime import datetime, timedelta
import json
import os

# Initialize session state for habits if it doesn't exist
if 'habits' not in st.session_state:
    st.session_state.habits = {}

if 'habit_data' not in st.session_state:
    st.session_state.habit_data = []

# File paths for data persistence
HABITS_FILE = 'habits.json'
PROGRESS_FILE = 'progress.json'

# Load saved habits and progress data
def load_data():
    if os.path.exists(HABITS_FILE):
        with open(HABITS_FILE, 'r') as f:
            st.session_state.habits = json.load(f)
    
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            st.session_state.habit_data = json.load(f)

# Save habits and progress data
def save_data():
    with open(HABITS_FILE, 'w') as f:
        json.dump(st.session_state.habits, f)
    
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(st.session_state.habit_data, f)

# Add new habit
def add_habit():
    if st.session_state.new_habit and st.session_state.new_habit not in st.session_state.habits:
        st.session_state.habits[st.session_state.new_habit] = {
            'created_date': datetime.now().strftime('%Y-%m-%d'),
            'current_streak': 0,
            'type': st.session_state.habit_type,
            'target': st.session_state.habit_target
        }
        save_data()

# Remove habit
def remove_habit(habit_name):
    if habit_name in st.session_state.habits:
        # Remove from habits dictionary
        del st.session_state.habits[habit_name]
        # Remove all progress data for this habit
        st.session_state.habit_data = [
            entry for entry in st.session_state.habit_data 
            if entry['habit'] != habit_name
        ]
        save_data()

# Update habit progress
def update_habit_progress(habit_name, value):
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Find existing entry for today
    existing_entry = None
    for entry in st.session_state.habit_data:
        if entry['date'] == today and entry['habit'] == habit_name:
            existing_entry = entry
            break
    
    if existing_entry:
        existing_entry['value'] = value
    else:
        entry = {
            'date': today,
            'habit': habit_name,
            'completed': True,
            'value': value
        }
        st.session_state.habit_data.append(entry)
    
    # Update streak
    if value >= st.session_state.habits[habit_name]['target']:
        st.session_state.habits[habit_name]['current_streak'] += 1
    save_data()

# Reset streaks for habits not completed yesterday
def check_and_reset_streaks():
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    yesterday_entries = {e['habit']: e['value'] for e in st.session_state.habit_data if e['date'] == yesterday}
    
    for habit in st.session_state.habits:
        if habit not in yesterday_entries or yesterday_entries[habit] < st.session_state.habits[habit]['target']:
            st.session_state.habits[habit]['current_streak'] = 0
    save_data()

# Get completion status and values for a date range
def get_progress_data(habit, days=30):
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
    values = []
    
    for date in dates:
        entry = next((e for e in st.session_state.habit_data 
                     if e['date'] == date and e['habit'] == habit), None)
        values.append(entry['value'] if entry else 0)
    
    return dates[::-1], values[::-1]

# Main app
def main():
    st.title('Habit Tracker')
    
    # Load saved data
    load_data()
    
    # Check and reset streaks
    check_and_reset_streaks()
    
    # Add new habit section
    st.subheader('Add New Habit')
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.text_input('Enter new habit:', key='new_habit')
    with col2:
        st.selectbox('Type', ['Minutes', 'Count'], key='habit_type')
    with col3:
        st.number_input('Daily Target', min_value=1, value=1, key='habit_target')
    st.button('Add Habit', on_click=add_habit)
    
    # Display current habits and progress
    st.subheader('Track Your Habits')
    
    if st.session_state.habits:
        # Today's date for reference
        today = datetime.now().strftime('%Y-%m-%d')
        
        for habit in st.session_state.habits:
            st.write("---")
            # Header with habit name and streak
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{habit}**")
                st.write(f"Daily Target: {st.session_state.habits[habit]['target']} {st.session_state.habits[habit]['type']}")
            with col2:
                st.write(f"Streak: {st.session_state.habits[habit]['current_streak']}")
            with col3:
                if st.button('🗑️ Delete', key=f'delete_{habit}'):
                    remove_habit(habit)
                    st.rerun()
            
            # Today's progress
            today_entry = next((e for e in st.session_state.habit_data 
                              if e['date'] == today and e['habit'] == habit), None)
            current_value = today_entry['value'] if today_entry else 0
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button('-', key=f'dec_{habit}'):
                    new_value = max(0, current_value - 1)
                    update_habit_progress(habit, new_value)
            with col2:
                st.write(f"Today's Progress: {current_value} {st.session_state.habits[habit]['type']}")
            with col3:
                if st.button('+', key=f'inc_{habit}'):
                    new_value = current_value + 1
                    update_habit_progress(habit, new_value)
            
            # Progress visualization
            st.write("Monthly Progress:")
            dates, values = get_progress_data(habit)
            
            # Create a simple bar chart using Streamlit
            progress_data = [{"date": d, "value": v} for d, v in zip(dates, values)]
            chart_data = {"index": list(range(len(values))), "values": values}
            st.bar_chart(chart_data["values"])
            
            # Display last 7 days in detail
            st.write("Last 7 days detail:")
            week_cols = st.columns(7)
            for i, (date, value, col) in enumerate(zip(dates[-7:], values[-7:], week_cols)):
                with col:
                    date_display = datetime.strptime(date, '%Y-%m-%d').strftime('%m/%d')
                    st.write(date_display)
                    st.write(f"{value}")
                    target = st.session_state.habits[habit]['target']
                    if value >= target:
                        st.markdown('✅')
                    else:
                        st.markdown('❌')
    else:
        st.info('Add your first habit to get started!')

if __name__ == '__main__':
    main()

import sys
from typing import Optional, Tuple, List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import matplotlib.ticker as mticker
import re
from datetime import date

# Constants
ROLLING_AVERAGE_MONTHS: int = 12  # Number of months for rolling average
DEFAULT_PROJECTION_MONTHS: int = 48  # Default number of months to project
INTERNAL_ALLOCATIONS_COLUMN: str = 'Internal allocations resources'
EXCLUDED_EXPENDITURE_TERMS: List[str] = ['fec', 'Debtors research external charges']
REQUIRED_COLUMNS: List[str] = ['Expenditure Type', 'Accounting Date', 'Expenditure Category', 'Raw Cost']

def get_future_dates(last_date: pd.Timestamp, projection_end_date: Optional[pd.Timestamp] = None, n_months: int = 48) -> Tuple[pd.DatetimeIndex, int]:
    """Generate future dates for projection.
    
    Args:
        last_date: Last date in the historical data
        projection_end_date: Optional end date for projections
        n_months: Number of months to project (default 48)
    
    Returns:
        Tuple of (future_dates, n_months)
    
    Raises:
        ValueError: If projection_end_date is before last_date
    """
    if projection_end_date is not None:
        if projection_end_date <= last_date:
            raise ValueError(f"projection_end_date ({projection_end_date}) must be after last_date ({last_date})")
        # Generate months up to and including the end date
        future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), projection_end_date, freq='ME')
        n_months = len(future_dates)
    else:
        future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=n_months, freq='ME')
    return future_dates, n_months

def get_fin_year_and_label(date: pd.Timestamp, start_month: int = 4) -> Tuple[int, str]:
    """Calculate financial year and label for a given date.
    
    Args:
        date: Date to calculate financial year for
        start_month: Month when financial year starts (1-12)
    
    Returns:
        Tuple of (financial_year_start, label_string)
    """
    calendar_year: int = date.year
    if date.month >= start_month:
        financial_year_start: int = calendar_year
        financial_year_end: int = calendar_year + 1
    else:
        financial_year_start = calendar_year - 1
        financial_year_end = calendar_year
    # If start_month is 1 (January), use 'YYYY' as label
    if start_month == 1:
        label = f"{financial_year_start}"
    else:
        start_month_name: str = pd.Timestamp(year=2000, month=start_month, day=1).strftime('%B')
        end_month_number: int = (start_month - 1) if start_month > 1 else 12
        end_month_name: str = pd.Timestamp(year=2000, month=end_month_number, day=1).strftime('%B')
        label = f"{start_month_name} {str(financial_year_start)[-2:]}/{end_month_name} {str(financial_year_end)[-2:]}"
    return financial_year_start, label


def create_cumulative_plot(
    categories: List[str],
    title: str,
    filename: str,
    cumulative_monthly_costs: pd.DataFrame,
    cumulative_projection: pd.DataFrame,
    category_color_map: Dict[str, str],
    last_actual_date: pd.Timestamp,
    march_date_locations: List[pd.Timestamp],
    year_start_month: int,
    output_dir: str,
    figsize: Tuple[int, int] = (14, 7),
    format_y_axis: bool = False
) -> None:
    """Create and save a cumulative cost plot for given categories.
    
    Args:
        categories: List of category column names to plot
        title: Plot title
        filename: Output filename (without path)
        cumulative_monthly_costs: Historical cumulative costs DataFrame
        cumulative_projection: Projected cumulative costs DataFrame
        category_color_map: Mapping of categories to colors
        last_actual_date: Last date with actual data
        march_date_locations: List of dates for x-axis ticks
        year_start_month: Month when financial year starts
        output_dir: Directory to save the plot
        figsize: Figure size tuple
        format_y_axis: Whether to format y-axis as integers
    """
    plt.figure(figsize=figsize)
    for category in categories:
        if category in cumulative_monthly_costs.columns:
            plt.plot(cumulative_monthly_costs.index, cumulative_monthly_costs[category], 
                    label=f"{category} (actual)", linestyle='solid', color=category_color_map[category])
            plt.plot(cumulative_projection.index, cumulative_projection[category], 
                    label="_nolegend_", linestyle='dashed', color=category_color_map[category])
    plt.axvline(x=mdates.date2num(last_actual_date.to_pydatetime()), color='gray', linestyle='--', linewidth=1)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Burdened Cost in Provider Ledger Currency')
    plt.legend(
        title='Expenditure Category',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize='small',
        title_fontsize='small'
    )
    # Set x-ticks at every March and label as financial year
    current_axis = plt.gca()
    current_axis.set_xticks([mdates.date2num(d.to_pydatetime()) for d in march_date_locations])
    current_axis.set_xticklabels([get_fin_year_and_label(d, year_start_month)[1] for d in march_date_locations], rotation=45, ha='right')
    if format_y_axis:
        current_axis.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    except Exception as e:
        plt.close()
        raise RuntimeError(f"Failed to save plot {filename}: {e}") from e
    plt.close()

def get_quarter_label(date: pd.Timestamp, year_start_month: int, financial_year_label: str) -> str:
    """Get quarter label for a date based on financial year.
    
    Args:
        date: Date to get quarter for
        year_start_month: Month when financial year starts (1-12)
        financial_year_label: Financial year label (e.g., "April 23/March 24")
    
    Returns:
        Quarter label string (e.g., "Q1 Apr-Jun 23/24")
    """
    month = date.month
    # Calculate which quarter (1-4) based on financial year start month
    # Adjust month to be relative to financial year start
    adjusted_month = ((month - year_start_month) % 12) + 1
    
    if adjusted_month <= 3:
        quarter_num = 1
        quarter_months = [year_start_month, (year_start_month + 1) % 12 or 12, (year_start_month + 2) % 12 or 12]
    elif adjusted_month <= 6:
        quarter_num = 2
        quarter_months = [(year_start_month + 3) % 12 or 12, (year_start_month + 4) % 12 or 12, (year_start_month + 5) % 12 or 12]
    elif adjusted_month <= 9:
        quarter_num = 3
        quarter_months = [(year_start_month + 6) % 12 or 12, (year_start_month + 7) % 12 or 12, (year_start_month + 8) % 12 or 12]
    else:
        quarter_num = 4
        quarter_months = [(year_start_month + 9) % 12 or 12, (year_start_month + 10) % 12 or 12, (year_start_month + 11) % 12 or 12]
    
    # Get month names
    start_month_name = pd.Timestamp(year=2000, month=quarter_months[0], day=1).strftime('%b')
    end_month_name = pd.Timestamp(year=2000, month=quarter_months[2], day=1).strftime('%b')
    
    # Use the financial year label as the year suffix for consistency
    return f"Q{quarter_num} {start_month_name}-{end_month_name} {financial_year_label}"


def get_column_sort_key(column_name: str) -> Tuple[float, int, int, int]:
    """Get sort key for table columns.
    
    Args:
        column_name: Name of the column to sort
    
    Returns:
        Tuple of (year, is_projected_flag, is_quarter_flag, quarter_num)
    """
    if column_name == 'Total':
        return (float('inf'), 0, 0, 0)
    
    # Check if it's a quarter column
    is_quarter = column_name.startswith('Q')
    
    # Extract the year from the label
    # For yearly columns: 'May 23/April 24' or 'July 23/June 24'
    # For quarter columns: 'Q1 Apr-Jun April 23/March 24'
    year_match = re.search(r'(\d{2,4})[ ]*/', column_name) or re.search(r'(\d{2,4})', column_name)
    if year_match:
        year_str = year_match.group(1)
        year_value = int(year_str) if len(year_str) == 4 else 2000 + int(year_str)
    else:
        year_value = 0
    
    is_projected = 1 if 'proj' in column_name else 0
    
    # Extract quarter number if it's a quarter column
    quarter_num = 0
    if is_quarter:
        q_match = re.search(r'Q(\d)', column_name)
        if q_match:
            quarter_num = int(q_match.group(1))
    
    return (year_value, is_projected, 1 if is_quarter else 0, quarter_num)

def error_exit(message: str) -> None:
    """Print error message and exit with code 1."""
    print(f"Error: {message}")
    sys.exit(1)

def process_folder(
    input_dir: str,
    projection_end_date: Optional[pd.Timestamp],
    year_start_month: int
) -> None:
    """Process a single folder containing Excel files and generate visualizations.
    
    Args:
        input_dir: Path to directory containing .xlsx files
        projection_end_date: Optional end date for projections
        year_start_month: Month when financial year starts (1-12)
    """
    if not os.path.isdir(input_dir):
        print(f"Warning: Skipping {input_dir} - path is not a directory")
        return

    xlsx_files: List[str] = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]
    if not xlsx_files:
        print(f"Warning: Skipping {input_dir} - no .xlsx files found")
        return

    # Create output directory with today's date (yyyy-mm-dd format)
    today_str: str = date.today().strftime('%Y-%m-%d')
    date_output_dir: str = os.path.join(os.path.abspath(input_dir), today_str)
    os.makedirs(date_output_dir, exist_ok=True)
    output_dir: str = date_output_dir
    
    # Separate Corrections.xlsx from other files
    corrections_file: Optional[str] = None
    data_files: List[str] = []
    for xlsx_file in xlsx_files:
        if xlsx_file == 'Corrections.xlsx':
            corrections_file = xlsx_file
        else:
            data_files.append(xlsx_file)
    
    if not data_files:
        print(f"Warning: Skipping {input_dir} - no data files found (only Corrections.xlsx)")
        return
    
    # Load corrections file if it exists
    corrections_df: Optional[pd.DataFrame] = None
    if corrections_file:
        corrections_path = os.path.join(input_dir, corrections_file)
        try:
            corrections_df = pd.read_excel(corrections_path, engine='openpyxl')
            print(f"Loaded corrections file: {corrections_file}")
        except Exception as e:
            print(f"Warning: Failed to read {corrections_file} in {input_dir}: {e}")
            corrections_df = None
    
    # Load and combine all data .xlsx files (excluding Corrections.xlsx)
    dataframes_to_combine: List[pd.DataFrame] = []
    for xlsx_file in data_files:
        file_path = os.path.join(input_dir, xlsx_file)
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            dataframes_to_combine.append(df)
        except Exception as e:
            print(f"Warning: Failed to read {xlsx_file} in {input_dir}: {e}")
            return
    
    if not dataframes_to_combine:
        print(f"Warning: Skipping {input_dir} - no valid data loaded from Excel files")
        return
    
    expense_data: pd.DataFrame = pd.concat(dataframes_to_combine, ignore_index=True)
    
    # Apply corrections if Corrections.xlsx exists
    if corrections_df is not None:
        # Check if required columns exist in corrections file
        if 'Transaction Number' not in corrections_df.columns:
            print(f"Warning: Corrections.xlsx missing 'Transaction Number' column, skipping corrections")
        elif 'Corrected Date' not in corrections_df.columns:
            print(f"Warning: Corrections.xlsx missing 'Corrected Date' column, skipping corrections")
        elif 'Transaction Number' not in expense_data.columns:
            print(f"Warning: Data files missing 'Transaction Number' column, skipping corrections")
        else:
            # Create a mapping from Transaction Number to Corrected Date
            corrections_df = corrections_df.dropna(subset=['Transaction Number', 'Corrected Date'])
            corrections_df['Corrected Date'] = pd.to_datetime(corrections_df['Corrected Date'], errors='coerce')
            corrections_df = corrections_df.dropna(subset=['Corrected Date'])
            
            if len(corrections_df) > 0:
                # Create a dictionary mapping Transaction Number to Corrected Date
                # Convert Transaction Numbers to string, strip whitespace, and remove .0 suffix for consistent matching
                corrections_df['Transaction Number'] = (
                    corrections_df['Transaction Number']
                    .astype(str)
                    .str.strip()
                    .str.replace(r'\.0$', '', regex=True)  # Remove trailing .0
                )
                corrections_map: Dict[str, pd.Timestamp] = dict(
                    zip(corrections_df['Transaction Number'], corrections_df['Corrected Date'])
                )
                
                print(f"\nLoaded {len(corrections_map)} corrections from Corrections.xlsx")
                print(f"Sample correction Transaction Numbers: {list(corrections_map.keys())[:5]}")
                
                # Apply corrections to expense_data
                if 'Transaction Number' in expense_data.columns:
                    # Convert Transaction Number to string, strip whitespace, and remove .0 suffix for matching
                    expense_data['Transaction Number'] = (
                        expense_data['Transaction Number']
                        .astype(str)
                        .str.strip()
                        .str.replace(r'\.0$', '', regex=True)  # Remove trailing .0
                    )
                    # Filter out 'nan' strings from matching (but keep the rows)
                    valid_mask = expense_data['Transaction Number'] != 'nan'
                    
                    # Debug: Show sample transaction numbers from expense data
                    valid_transaction_numbers = expense_data.loc[valid_mask, 'Transaction Number'].unique()
                    print(f"Found {len(valid_transaction_numbers)} unique Transaction Numbers in expense data")
                    print(f"Sample expense Transaction Numbers: {list(valid_transaction_numbers[:5])}")
                    
                    # Count how many corrections will be applied (only on valid transaction numbers)
                    mask = valid_mask & expense_data['Transaction Number'].isin(corrections_map.keys())
                    num_corrections = mask.sum()
                    
                    # Debug: Show which transaction numbers should match
                    if num_corrections == 0:
                        # Try to find potential matches (case-insensitive, partial matches)
                        corrections_keys_lower = {k.lower(): k for k in corrections_map.keys()}
                        expense_txn_lower = expense_data.loc[valid_mask, 'Transaction Number'].str.lower()
                        potential_matches = expense_txn_lower.isin(corrections_keys_lower.keys())
                        if potential_matches.any():
                            print(f"\nWarning: Found {potential_matches.sum()} potential matches (case-insensitive)")
                            print("This suggests there might be case sensitivity issues.")
                        else:
                            # Show some examples of what we're looking for vs what we have
                            print(f"\nDebug: Looking for Transaction Numbers like: {list(corrections_map.keys())[:3]}")
                            print(f"Debug: Have Transaction Numbers like: {list(valid_transaction_numbers[:3])}")
                    
                    if num_corrections > 0:
                        # Store original dates before applying corrections
                        corrections_applied = expense_data.loc[mask, ['Transaction Number', 'Accounting Date']].copy()
                        corrections_applied['Original Date'] = corrections_applied['Accounting Date']
                        corrections_applied['Corrected Date'] = corrections_applied['Transaction Number'].map(corrections_map)
                        
                        # Apply corrections
                        expense_data.loc[mask, 'Accounting Date'] = expense_data.loc[mask, 'Transaction Number'].map(corrections_map)
                        
                        # Output corrections details
                        print(f"\nApplied {num_corrections} date corrections from Corrections.xlsx:")
                        print("=" * 80)
                        for idx, row in corrections_applied.iterrows():
                            print(f"Transaction Number: {row['Transaction Number']}")
                            print(f"  Original Date: {row['Original Date']}")
                            print(f"  Corrected Date: {row['Corrected Date']}")
                            print("-" * 80)
                        
                        # Save corrections log to CSV file
                        corrections_log_path = os.path.join(output_dir, f"{os.path.basename(input_dir) or 'combined'}_corrections_applied.csv")
                        try:
                            corrections_applied[['Transaction Number', 'Original Date', 'Corrected Date']].to_csv(
                                corrections_log_path, index=False, float_format='%.2f'
                            )
                            print(f"\nCorrections log saved to: {corrections_log_path}")
                        except Exception as e:
                            print(f"Warning: Failed to save corrections log: {e}")
                    else:
                        print(f"No matching transaction numbers found for corrections")
    
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in expense_data.columns]
    if missing_columns:
        print(f"Warning: Skipping {input_dir} - missing required columns: {', '.join(missing_columns)}")
        return
    
    if len(expense_data) == 0:
        print(f"Warning: Skipping {input_dir} - no data found in Excel files")
        return

    # Use folder name or "combined" as base filename
    base_filename: str = os.path.basename(output_dir) or "combined" 

    # Remove rows where 'Expenditure Type' contains excluded terms
    expense_type_mask = pd.Series([False] * len(expense_data), index=expense_data.index)
    for term in EXCLUDED_EXPENDITURE_TERMS:
        expense_type_mask |= expense_data['Expenditure Type'].str.contains(term, case=False, na=False)
    expense_data = expense_data[~expense_type_mask]

    # Ensure the date column is datetime
    expense_data['Accounting Date'] = pd.to_datetime(expense_data['Accounting Date'])

    # Group by Accounting Date and Expenditure Category, summing the costs
    grouped_by_date_and_category: pd.DataFrame = expense_data.groupby(['Accounting Date', 'Expenditure Category'])['Raw Cost'].sum().reset_index()

    # Pivot for plotting and resample monthly
    monthly_costs_by_category: pd.DataFrame = grouped_by_date_and_category.pivot(index='Accounting Date', columns='Expenditure Category', values='Raw Cost')
    monthly_costs_by_category = monthly_costs_by_category.resample('ME').sum()

    if len(monthly_costs_by_category) == 0:
        print(f"Warning: Skipping {input_dir} - no monthly data available after processing")
        return

    last_actual_date: pd.Timestamp = monthly_costs_by_category.index[-1]
    try:
        projection_dates, num_projection_months = get_future_dates(last_actual_date, projection_end_date, DEFAULT_PROJECTION_MONTHS)
    except ValueError as e:
        print(f"Warning: Skipping {input_dir} - {e}")
        return

    # Cumulative sum for historical data
    cumulative_monthly_costs: pd.DataFrame = monthly_costs_by_category.cumsum()

    # Initialize monthly_projection using vectorized operations
    rolling_means = monthly_costs_by_category.fillna(0).rolling(window=ROLLING_AVERAGE_MONTHS, min_periods=1).mean().iloc[-1]
    monthly_projection = pd.DataFrame(
        {col: rolling_means[col] for col in monthly_costs_by_category.columns},
        index=projection_dates
    )

    # Cumulative projection for the plot
    cumulative_projection: pd.DataFrame = monthly_projection.cumsum() + cumulative_monthly_costs.iloc[-1]

    # Assign a color to each expenditure category
    color_cycle: List[str] = plt.rcParams['axes.prop_cycle'].by_key()['color']
    category_color_map: Dict[str, str] = {
        category: color_cycle[i % len(color_cycle)] 
        for i, category in enumerate(monthly_costs_by_category.columns)
    }

    # Identify the Staff Costs and other columns
    staff_cost_columns: List[str] = [
        col for col in monthly_costs_by_category.columns 
        if 'staff costs' in col.lower() or col == 'Internal allocations resources'
    ]
    other_category_columns: List[str] = [col for col in monthly_costs_by_category.columns if col not in staff_cost_columns]

    # Combine 'Internal allocations resources' with staff costs for both graphs and table
    if INTERNAL_ALLOCATIONS_COLUMN in monthly_costs_by_category.columns:
        staff_cols_to_update: List[str] = [col for col in staff_cost_columns if col != INTERNAL_ALLOCATIONS_COLUMN]
        if staff_cols_to_update:
            first_staff_col = staff_cols_to_update[0]
            monthly_costs_by_category[first_staff_col] += monthly_costs_by_category[INTERNAL_ALLOCATIONS_COLUMN]
            if INTERNAL_ALLOCATIONS_COLUMN in monthly_projection.columns:
                monthly_projection[first_staff_col] += monthly_projection[INTERNAL_ALLOCATIONS_COLUMN]
            cumulative_monthly_costs = monthly_costs_by_category.cumsum()
            cumulative_projection = monthly_projection.cumsum() + cumulative_monthly_costs.iloc[-1]
            staff_cost_columns = [col for col in staff_cost_columns if col != INTERNAL_ALLOCATIONS_COLUMN]

    # Combine actual and projection indices for x-ticks
    all_date_indices = cumulative_monthly_costs.index.union(cumulative_projection.index).sort_values()
    # Find dates for x-axis ticks: try month 3, then year_start_month, then month 1
    for month in [3, year_start_month, 1]:
        march_date_locations = [d for d in all_date_indices if d.month == month]
        if march_date_locations:
            break

    # Plot categories
    plot_configs = [
        (staff_cost_columns, f'Cumulative Burdened Cost for Staff Costs Over Time (with {num_projection_months}-Month Cumulative Projection)',
         f"{base_filename}_staff_costs_cumulative.png", (14, 5), True),
        (other_category_columns, f'Cumulative Burdened Cost by Expenditure Category (excluding Pay and Staff Costs) Over Time (with {num_projection_months}-Month Cumulative Projection)',
         f"{base_filename}_other_cumulative.png", (14, 7), False)
    ]
    
    for categories, title, filename, figsize, format_y in plot_configs:
        if categories:
            try:
                create_cumulative_plot(
                    categories=categories, title=title, filename=filename,
                    cumulative_monthly_costs=cumulative_monthly_costs,
                    cumulative_projection=cumulative_projection,
                    category_color_map=category_color_map,
                    last_actual_date=last_actual_date,
                    march_date_locations=march_date_locations,
                    year_start_month=year_start_month,
                    output_dir=output_dir, figsize=figsize, format_y_axis=format_y
                )
            except Exception as e:
                print(f"Warning: Failed to create plot {filename} in {input_dir}: {e}")
                return

    # --- Add table with quarterly totals for each Expenditure Category ---

    # Combine historical and projected (not cumulative) for quarterly sums
    combined_monthly_data: pd.DataFrame = pd.concat([monthly_costs_by_category, monthly_projection])
    # Index is already datetime, no conversion needed
    combined_monthly_data['Month'] = combined_monthly_data.index.to_series().dt.month

    # Set financial year and label based on year_start_month (compute once per date)
    fin_year_results = list(combined_monthly_data.index.map(lambda d: get_fin_year_and_label(d, year_start_month)))
    combined_monthly_data['FinYear'], combined_monthly_data['FinYearLabel'] = zip(*fin_year_results)
    
    # Add quarter labels
    combined_monthly_data['QuarterLabel'] = combined_monthly_data.apply(
        lambda row: get_quarter_label(row.name, year_start_month, row['FinYearLabel']), axis=1
    )

    # Get all unique financial years in order
    all_financial_years: List[str] = list(combined_monthly_data['FinYearLabel'].unique())
    # Get all unique quarters in order
    all_quarters: List[str] = list(combined_monthly_data['QuarterLabel'].unique())
    
    # Find the financial year label for the last actual date
    transition_financial_year: Optional[str] = None
    transition_quarter: Optional[str] = None
    if len(monthly_costs_by_category) > 0:
        last_date_mask = combined_monthly_data.index == monthly_costs_by_category.index[-1]
        if last_date_mask.any():
            transition_financial_year = combined_monthly_data.loc[last_date_mask, 'FinYearLabel'].iloc[0]
            transition_quarter = combined_monthly_data.loc[last_date_mask, 'QuarterLabel'].iloc[0]

    # Prepare a dict to hold the new columns
    yearly_totals_dict: Dict[str, Dict[str, float]] = {}

    # Pre-compute financial year labels for projection dates to avoid repeated calculations
    projection_fin_year_labels: pd.Series = pd.Series(
        [get_fin_year_and_label(d, year_start_month)[1] for d in monthly_projection.index],
        index=monthly_projection.index
    )

    # Generate table, excluding 'Internal allocations resources' as a separate row
    categories_for_table: List[str] = [col for col in monthly_costs_by_category.columns if col != INTERNAL_ALLOCATIONS_COLUMN]

    # Pre-compute masks for efficiency
    last_actual_month_number: int = last_actual_date.month

    # Pre-compute financial year masks for historical data (aligned with monthly_costs_by_category index)
    historical_fin_year_masks: Dict[str, pd.Series] = {
        fy: (combined_monthly_data.loc[monthly_costs_by_category.index, 'FinYearLabel'] == fy) 
        for fy in all_financial_years
    }
    projection_masks: Dict[str, pd.Series] = {
        fy: projection_fin_year_labels == fy for fy in all_financial_years
    }
    
    # Pre-compute quarter labels for projection dates
    projection_quarter_labels: pd.Series = pd.Series(
        [get_quarter_label(d, year_start_month, projection_fin_year_labels[d]) for d in monthly_projection.index],
        index=monthly_projection.index
    )
    
    # Pre-compute quarter masks for historical data
    historical_quarter_masks: Dict[str, pd.Series] = {
        q: (combined_monthly_data.loc[monthly_costs_by_category.index, 'QuarterLabel'] == q) 
        for q in all_quarters
    }
    projection_quarter_masks: Dict[str, pd.Series] = {
        q: projection_quarter_labels == q for q in all_quarters
    }

    # Pre-compute month mask for historical data
    historical_month_mask = combined_monthly_data.loc[monthly_costs_by_category.index, 'Month'] <= last_actual_month_number

    for category_column in categories_for_table:
        yearly_totals_dict[category_column] = {}
        # Calculate quarterly totals only
        for quarter_label in all_quarters:
            # Check if this quarter has historical data
            has_historical = quarter_label in historical_quarter_masks and historical_quarter_masks[quarter_label].any()
            # Check if this quarter has projection data
            has_projection = quarter_label in projection_quarter_masks and projection_quarter_masks[quarter_label].any()
            
            if has_historical and has_projection:
                # Split quarter: part actual, part projected
                quarter_mask = historical_quarter_masks[quarter_label]
                # Actual part: months up to and including last_actual_date
                actual_portion: float = monthly_costs_by_category[quarter_mask & (monthly_costs_by_category.index <= last_actual_date)][category_column].sum()
                # Projected part
                projection_mask = projection_quarter_masks[quarter_label] & (monthly_projection.index > last_actual_date)
                projected_portion: float = monthly_projection[projection_mask][category_column].sum()
                yearly_totals_dict[category_column][f"{quarter_label} (actual)"] = actual_portion
                yearly_totals_dict[category_column][f"{quarter_label} (proj)"] = projected_portion
            elif has_historical:
                # Fully actual quarter
                actual_total: float = monthly_costs_by_category[historical_quarter_masks[quarter_label]][category_column].sum()
                yearly_totals_dict[category_column][f"{quarter_label} (actual)"] = actual_total
            elif has_projection:
                # Fully projected quarter
                projected_total: float = monthly_projection[projection_quarter_masks[quarter_label]][category_column].sum()
                yearly_totals_dict[category_column][f"{quarter_label} (proj)"] = projected_total

    # Convert to DataFrame for pretty printing and round all values to 2 decimal places
    yearly_totals_table: pd.DataFrame = pd.DataFrame(yearly_totals_dict).T.fillna(0).round(2)
    yearly_totals_table['Total'] = yearly_totals_table.sum(axis=1).round(2)

    # Add a total row at the bottom
    total_row = yearly_totals_table.sum(axis=0).round(2)
    total_row.name = 'Total'
    yearly_totals_table = pd.concat([yearly_totals_table, total_row.to_frame().T])

    yearly_totals_table = yearly_totals_table.reindex(sorted(yearly_totals_table.columns, key=get_column_sort_key), axis=1)

    csv_output_path: str = os.path.join(output_dir, f"{base_filename}_quarterly_totals_by_expenditure_category.csv")
    try:
        yearly_totals_table.to_csv(csv_output_path, float_format='%.2f')
        print(f"Successfully processed: {input_dir}")
    except Exception as e:
        print(f"Warning: Failed to save CSV file in {input_dir}: {e}")


def read_config_file(config_path: str) -> List[Tuple[str, Optional[pd.Timestamp], Optional[int]]]:
    """Read folder paths and settings from a configuration file.
    
    The config file should contain one folder path per line, optionally followed by settings.
    Format: folder_path [projection_end_date] [year_start_month]
    Empty lines and lines starting with '#' are ignored.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        List of tuples: (folder_path, projection_end_date, year_start_month)
        projection_end_date and year_start_month are None if not specified
    
    Raises:
        SystemExit: If the config file cannot be read or has invalid format
    """
    if not os.path.isfile(config_path):
        error_exit(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            entries = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                # Parse from the end backwards to handle folder paths with spaces
                projection_end_date: Optional[pd.Timestamp] = None
                year_start_month: Optional[int] = None
                folder_path_parts: List[str] = []
                
                # Check if last token is a month number (1-12)
                if len(parts) >= 2:
                    try:
                        last_token_as_int = int(parts[-1])
                        if 1 <= last_token_as_int <= 12:
                            year_start_month = last_token_as_int
                            parts = parts[:-1]  # Remove the month number
                    except ValueError:
                        pass  # Last token is not a number, continue
                
                # Check if last token (or second-to-last if month was found) is a date
                if len(parts) >= 2 and '-' in parts[-1]:
                    try:
                        projection_end_date = pd.to_datetime(parts[-1])  # type: ignore
                        parts = parts[:-1]  # Remove the date
                    except (ValueError, TypeError):
                        pass  # Not a valid date, treat as part of folder path
                
                # Everything remaining is the folder path
                folder_path = ' '.join(parts)
                
                entries.append((folder_path, projection_end_date, year_start_month))
        return entries
    except Exception as e:
        error_exit(f"Failed to read config file {config_path}: {e}")
        return []  # Never reached, but satisfies type checker


def main() -> None:
    """Main entry point for the cost visualization script."""
    # Parse command-line arguments for config file, optional projection end date, and year start month
    if len(sys.argv) < 2:
        print("Usage: plot-costs <config_file> [projection_end_date] [year_start_month]")
        print("  config_file: Path to text file containing folder paths (one per line)")
        print("  projection_end_date (optional): Default end date for projections (YYYY-MM-DD)")
        print("  year_start_month (optional): Default month when financial year starts (1-12, default: 4)")
        print("\nConfig file format:")
        print("  One folder path per line, optionally followed by settings:")
        print("    folder_path [projection_end_date] [year_start_month]")
        print("  Empty lines and lines starting with '#' are ignored")
        print("  Settings in config file override command-line defaults")
        print("\n  Examples:")
        print("    /path/to/folder1")
        print("    /path/to/folder2 2027-12-31 7")
        print("    /path/to/folder3 7")
        print("    /path/to/folder4 2028-06-30")
        print("    # This is a comment")
        sys.exit(1)
    
    config_file: str = sys.argv[1]

    # Optional: end date for projections and year start month
    projection_end_date: Optional[pd.Timestamp]
    year_start_month: int
    if len(sys.argv) >= 4 and '-' in sys.argv[2]:
        try:
            projection_end_date = pd.to_datetime(sys.argv[2])  # type: ignore
            year_start_month = int(sys.argv[3]) if len(sys.argv) >= 4 else 4
        except (ValueError, TypeError) as e:
            error_exit(f"Invalid date or month format: {e}")
    elif len(sys.argv) >= 3 and '-' not in sys.argv[2]:
        projection_end_date = None
        try:
            year_start_month = int(sys.argv[2])
        except ValueError:
            error_exit(f"Invalid month number: {sys.argv[2]}")
    else:
        projection_end_date = None
        year_start_month = 4
    
    if not (1 <= year_start_month <= 12):
        error_exit(f"year_start_month must be between 1 and 12, got {year_start_month}")

    # Read folder list from config file
    folder_entries: List[Tuple[str, Optional[pd.Timestamp], Optional[int]]] = read_config_file(config_file)
    
    if not folder_entries:
        error_exit(f"No folders found in config file: {config_file}")
    
    print(f"Found {len(folder_entries)} folder(s) to process")
    
    # Process each folder
    successful = 0
    failed = 0
    for folder_path, folder_projection_end_date, folder_year_start_month in folder_entries:
        # Use folder-specific settings if provided, otherwise use command-line defaults
        effective_projection_end_date = folder_projection_end_date if folder_projection_end_date is not None else projection_end_date
        effective_year_start_month = folder_year_start_month if folder_year_start_month is not None else year_start_month
        
        print(f"\nProcessing: {folder_path}")
        if folder_projection_end_date is not None or folder_year_start_month is not None:
            settings_info = []
            if folder_projection_end_date is not None:
                settings_info.append(f"projection_end_date={folder_projection_end_date.strftime('%Y-%m-%d')}")
            if folder_year_start_month is not None:
                settings_info.append(f"year_start_month={folder_year_start_month}")
            print(f"  Using settings: {', '.join(settings_info)}")
        
        try:
            process_folder(folder_path, effective_projection_end_date, effective_year_start_month)
            successful += 1
        except Exception as e:
            print(f"Error processing {folder_path}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete: {successful} successful, {failed} failed")
    print(f"{'='*60}")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()

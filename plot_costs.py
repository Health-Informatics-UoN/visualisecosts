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
DEFAULT_YEAR_START_MONTH: int = 4  # Default financial year start month (April)
MIN_MONTH: int = 1
MAX_MONTH: int = 12
INTERNAL_ALLOCATIONS_COLUMN: str = 'Internal allocations resources'
EXCLUDED_EXPENDITURE_TERMS: List[str] = ['fec', 'Debtors research external charges']
REQUIRED_COLUMNS: List[str] = ['Expenditure Type', 'Accounting Date', 'Expenditure Category', 'Raw Cost']
CORRECTIONS_FILENAME: str = 'Corrections.xlsx'
CORRECTED_DATE_COLUMN: str = 'Corrected Date'
TRANSACTION_NUMBER_COLUMN: str = 'Transaction Number'

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
    """Calculate financial year and label for a given date."""
    calendar_year = date.year
    financial_year_start = calendar_year if date.month >= start_month else calendar_year - 1
    financial_year_end = calendar_year + 1 if date.month >= start_month else calendar_year
    if start_month == 1:
        return financial_year_start, f"{financial_year_start}"
    start_month_name = pd.Timestamp(year=2000, month=start_month, day=1).strftime('%B')
    end_month_name = pd.Timestamp(year=2000, month=((start_month - 1) if start_month > 1 else 12), day=1).strftime('%B')
    return financial_year_start, f"{start_month_name} {str(financial_year_start)[-2:]}/{end_month_name} {str(financial_year_end)[-2:]}"


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
    """Get quarter label for a date based on financial year."""
    adjusted_month = ((date.month - year_start_month) % 12) + 1
    quarter_num = ((adjusted_month - 1) // 3) + 1
    start_month = ((year_start_month + (quarter_num - 1) * 3 - 1) % 12) + 1
    end_month = ((year_start_month + quarter_num * 3 - 1) % 12) + 1
    start_month_name = pd.Timestamp(year=2000, month=start_month, day=1).strftime('%b')
    end_month_name = pd.Timestamp(year=2000, month=end_month, day=1).strftime('%b')
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


def normalize_transaction_number(series: pd.Series) -> pd.Series:
    """Normalize transaction numbers for consistent matching.
    
    Converts to string, strips whitespace, and removes trailing .0 suffix.
    
    Args:
        series: Series containing transaction numbers
    
    Returns:
        Normalized series
    """
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r'\.0$', '', regex=True)
    )


def parse_command_line_args() -> Tuple[Optional[pd.Timestamp], int]:
    """Parse command-line arguments for projection end date and year start month.
    
    Returns:
        Tuple of (projection_end_date, year_start_month)
    """
    projection_end_date: Optional[pd.Timestamp]
    year_start_month: int
    
    if len(sys.argv) >= 4 and '-' in sys.argv[2]:
        try:
            projection_end_date = pd.to_datetime(sys.argv[2])  # type: ignore
            year_start_month = int(sys.argv[3]) if len(sys.argv) >= 4 else DEFAULT_YEAR_START_MONTH
        except (ValueError, TypeError) as e:
            error_exit(f"Invalid date or month format: {e}")
    elif len(sys.argv) >= 3:
        if '-' in sys.argv[2]:
            try:
                projection_end_date = pd.to_datetime(sys.argv[2])  # type: ignore
                year_start_month = DEFAULT_YEAR_START_MONTH
            except (ValueError, TypeError) as e:
                error_exit(f"Invalid date format: {sys.argv[2]} ({e})")
        else:
            projection_end_date = None
            try:
                year_start_month = int(sys.argv[2])
            except ValueError:
                # Check if this might be part of a split path
                if len(sys.argv) >= 2:
                    potential_path = ' '.join([sys.argv[1], sys.argv[2]])
                    if os.path.isfile(potential_path) or os.path.isdir(potential_path):
                        error_exit(f"Invalid month number: '{sys.argv[2]}'\n"
                                  f"Note: Your path appears to have been split. Did you mean: '{potential_path}'?\n"
                                  f"Please quote the path in your command: \"{potential_path}\"")
                error_exit(f"Invalid month number: '{sys.argv[2]}'. Expected a number between {MIN_MONTH}-{MAX_MONTH}, or a date in YYYY-MM-DD format.\n"
                          f"Note: If '{sys.argv[2]}' is part of a folder path, please quote the entire path in your command.")
    else:
        projection_end_date = None
        year_start_month = DEFAULT_YEAR_START_MONTH
    
    if not (MIN_MONTH <= year_start_month <= MAX_MONTH):
        error_exit(f"year_start_month must be between {MIN_MONTH} and {MAX_MONTH}, got {year_start_month}")
    
    return projection_end_date, year_start_month

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
    output_dir = os.path.join(os.path.abspath(input_dir), date.today().strftime('%Y-%m-%d'))
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate Corrections.xlsx from other files
    corrections_file = next((f for f in xlsx_files if f == CORRECTIONS_FILENAME), None)
    data_files = [f for f in xlsx_files if f != CORRECTIONS_FILENAME]
    
    if not data_files:
        print(f"Warning: Skipping {input_dir} - no data files found (only Corrections.xlsx)")
        return
    
    # Load corrections file if it exists
    corrections_df = None
    if corrections_file:
        try:
            corrections_df = pd.read_excel(os.path.join(input_dir, corrections_file), engine='openpyxl')
            print(f"Loaded corrections file: {corrections_file}")
        except Exception as e:
            print(f"Warning: Failed to read {corrections_file} in {input_dir}: {e}")
    
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
    if corrections_df is not None and all(col in corrections_df.columns for col in [TRANSACTION_NUMBER_COLUMN, CORRECTED_DATE_COLUMN]) and TRANSACTION_NUMBER_COLUMN in expense_data.columns:
        corrections_df = corrections_df.dropna(subset=[TRANSACTION_NUMBER_COLUMN, CORRECTED_DATE_COLUMN])
        corrections_df[CORRECTED_DATE_COLUMN] = pd.to_datetime(corrections_df[CORRECTED_DATE_COLUMN], errors='coerce')
        corrections_df = corrections_df.dropna(subset=[CORRECTED_DATE_COLUMN])
        if len(corrections_df) > 0:
            corrections_df[TRANSACTION_NUMBER_COLUMN] = normalize_transaction_number(corrections_df[TRANSACTION_NUMBER_COLUMN])
            corrections_map = dict(zip(corrections_df[TRANSACTION_NUMBER_COLUMN], corrections_df[CORRECTED_DATE_COLUMN]))
            print(f"\nLoaded {len(corrections_map)} corrections from {CORRECTIONS_FILENAME}")
            print(f"Sample correction Transaction Numbers: {list(corrections_map.keys())[:5]}")
            expense_data[TRANSACTION_NUMBER_COLUMN] = normalize_transaction_number(expense_data[TRANSACTION_NUMBER_COLUMN])
            valid_mask = expense_data[TRANSACTION_NUMBER_COLUMN] != 'nan'
            valid_transaction_numbers = expense_data.loc[valid_mask, TRANSACTION_NUMBER_COLUMN].unique()
            print(f"Found {len(valid_transaction_numbers)} unique Transaction Numbers in expense data")
            print(f"Sample expense Transaction Numbers: {list(valid_transaction_numbers[:5])}")
            mask = valid_mask & expense_data[TRANSACTION_NUMBER_COLUMN].isin(corrections_map.keys())
            num_corrections = mask.sum()
            if num_corrections == 0:
                corrections_keys_lower = {k.lower(): k for k in corrections_map.keys()}
                if expense_data.loc[valid_mask, TRANSACTION_NUMBER_COLUMN].str.lower().isin(corrections_keys_lower.keys()).any():
                    print(f"\nWarning: Found potential matches (case-insensitive)")
                else:
                    print(f"\nDebug: Looking for Transaction Numbers like: {list(corrections_map.keys())[:3]}")
                    print(f"Debug: Have Transaction Numbers like: {list(valid_transaction_numbers[:3])}")
            if num_corrections > 0:
                corrections_applied = expense_data.loc[mask, [TRANSACTION_NUMBER_COLUMN, 'Accounting Date']].copy()
                corrections_applied['Original Date'] = corrections_applied['Accounting Date']
                corrections_applied[CORRECTED_DATE_COLUMN] = corrections_applied[TRANSACTION_NUMBER_COLUMN].map(corrections_map)
                expense_data.loc[mask, 'Accounting Date'] = expense_data.loc[mask, TRANSACTION_NUMBER_COLUMN].map(corrections_map)
                print(f"\nApplied {num_corrections} date corrections from Corrections.xlsx:")
                print("=" * 80)
                for idx, row in corrections_applied.iterrows():
                    print(f"Transaction Number: {row[TRANSACTION_NUMBER_COLUMN]}\n  Original Date: {row['Original Date']}\n  Corrected Date: {row[CORRECTED_DATE_COLUMN]}\n" + "-" * 80)
                try:
                    corrections_applied[[TRANSACTION_NUMBER_COLUMN, 'Original Date', CORRECTED_DATE_COLUMN]].to_csv(
                        os.path.join(output_dir, f"{os.path.basename(input_dir) or 'combined'}_corrections_applied.csv"), index=False, float_format='%.2f')
                    print(f"\nCorrections log saved")
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
    # Optimized: use regex with | to match all terms at once
    if EXCLUDED_EXPENDITURE_TERMS:
        pattern = '|'.join(re.escape(term) for term in EXCLUDED_EXPENDITURE_TERMS)
        expense_type_mask = expense_data['Expenditure Type'].str.contains(pattern, case=False, na=False, regex=True)
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
    monthly_projection = pd.DataFrame({col: rolling_means[col] for col in monthly_costs_by_category.columns}, index=projection_dates)
    cumulative_projection = monthly_projection.cumsum() + cumulative_monthly_costs.iloc[-1]

    # Assign a color to each expenditure category
    color_cycle: List[str] = plt.rcParams['axes.prop_cycle'].by_key()['color']
    category_color_map: Dict[str, str] = {
        category: color_cycle[i % len(color_cycle)] 
        for i, category in enumerate(monthly_costs_by_category.columns)
    }

    # Identify the Staff Costs and other columns
    staff_cost_columns = [col for col in monthly_costs_by_category.columns if 'staff costs' in col.lower() or col == INTERNAL_ALLOCATIONS_COLUMN]
    other_category_columns = [col for col in monthly_costs_by_category.columns if col not in staff_cost_columns]
    # Combine 'Internal allocations resources' with staff costs
    if INTERNAL_ALLOCATIONS_COLUMN in monthly_costs_by_category.columns:
        staff_cols_to_update = [col for col in staff_cost_columns if col != INTERNAL_ALLOCATIONS_COLUMN]
        if staff_cols_to_update:
            first_staff_col = staff_cols_to_update[0]
            monthly_costs_by_category[first_staff_col] += monthly_costs_by_category[INTERNAL_ALLOCATIONS_COLUMN]
            if INTERNAL_ALLOCATIONS_COLUMN in monthly_projection.columns:
                monthly_projection[first_staff_col] += monthly_projection[INTERNAL_ALLOCATIONS_COLUMN]
            cumulative_monthly_costs, cumulative_projection = monthly_costs_by_category.cumsum(), monthly_projection.cumsum() + cumulative_monthly_costs.iloc[-1]
            staff_cost_columns = [col for col in staff_cost_columns if col != INTERNAL_ALLOCATIONS_COLUMN]

    # Find dates for x-axis ticks: try month 3, then year_start_month, then month 1
    all_date_indices = cumulative_monthly_costs.index.union(cumulative_projection.index).sort_values()
    march_date_locations = next(([d for d in all_date_indices if d.month == m] for m in [3, year_start_month, 1]), [])

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
    combined_monthly_data = pd.concat([monthly_costs_by_category, monthly_projection])
    combined_monthly_data['Month'] = combined_monthly_data.index.to_series().dt.month
    fin_year_results = [get_fin_year_and_label(d, year_start_month) for d in combined_monthly_data.index]
    combined_monthly_data['FinYear'], combined_monthly_data['FinYearLabel'] = zip(*fin_year_results)
    combined_monthly_data['QuarterLabel'] = [get_quarter_label(date, year_start_month, label) for date, label in zip(combined_monthly_data.index, combined_monthly_data['FinYearLabel'])]
    all_financial_years = list(combined_monthly_data['FinYearLabel'].unique())
    all_quarters = list(combined_monthly_data['QuarterLabel'].unique())
    transition_financial_year = transition_quarter = None
    if len(monthly_costs_by_category) > 0:
        last_date_mask = combined_monthly_data.index == monthly_costs_by_category.index[-1]
        if last_date_mask.any():
            transition_financial_year, transition_quarter = combined_monthly_data.loc[last_date_mask, ['FinYearLabel', 'QuarterLabel']].iloc[0]

    yearly_totals_dict: Dict[str, Dict[str, float]] = {}
    projection_fin_year_labels = pd.Series([get_fin_year_and_label(d, year_start_month)[1] for d in monthly_projection.index], index=monthly_projection.index)
    categories_for_table = [col for col in monthly_costs_by_category.columns if col != INTERNAL_ALLOCATIONS_COLUMN]
    last_actual_month_number = last_actual_date.month
    historical_fin_year_masks = {fy: (combined_monthly_data.loc[monthly_costs_by_category.index, 'FinYearLabel'] == fy) for fy in all_financial_years}
    projection_masks = {fy: projection_fin_year_labels == fy for fy in all_financial_years}
    projection_quarter_labels = pd.Series([get_quarter_label(d, year_start_month, projection_fin_year_labels[d]) for d in monthly_projection.index], index=monthly_projection.index, dtype='object')
    historical_quarter_masks = {q: (combined_monthly_data.loc[monthly_costs_by_category.index, 'QuarterLabel'] == q) for q in all_quarters}
    projection_quarter_masks = {q: projection_quarter_labels == q for q in all_quarters}
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

    yearly_totals_table = pd.DataFrame(yearly_totals_dict).T.fillna(0).round(2)
    yearly_totals_table['Total'] = yearly_totals_table.sum(axis=1).round(2)
    total_row = yearly_totals_table.sum(axis=0).round(2)
    total_row.name = 'Total'
    yearly_totals_table = pd.concat([yearly_totals_table, total_row.to_frame().T]).reindex(sorted(yearly_totals_table.columns, key=get_column_sort_key), axis=1)
    try:
        yearly_totals_table.to_csv(os.path.join(output_dir, f"{base_filename}_quarterly_totals_by_expenditure_category.csv"), float_format='%.2f')
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
                
                projection_end_date = year_start_month = None
                if line.startswith('"') or line.startswith("'"):
                    quote_char, end_quote_idx = line[0], line.find(line[0], 1)
                    folder_path, parts = (line[1:end_quote_idx], line[end_quote_idx + 1:].strip().split()) if end_quote_idx > 0 else (line, [])
                else:
                    parts = line.split()
                    if not parts:
                        continue
                    if len(parts) >= 2:
                        try:
                            if (len(parts[-1]) == 1 or (len(parts[-1]) == 2 and parts[-1].startswith('0'))) and 1 <= int(parts[-1]) <= 12:
                                year_start_month, parts = int(parts[-1]), parts[:-1]
                        except ValueError:
                            pass
                    if len(parts) >= 2 and parts[-1].count('-') == 2:
                        try:
                            projection_end_date, parts = pd.to_datetime(parts[-1]), parts[:-1]
                        except (ValueError, TypeError):
                            pass
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
        print("Usage:")
        print("  Single folder (backward compatible):")
        print("    plot-costs <directory> [projection_end_date] [year_start_month]")
        print("\n  Multiple folders (config file):")
        print("    plot-costs <config_file> [projection_end_date] [year_start_month]")
        print("\n  Arguments:")
        print("    directory/config_file: Path to directory OR config file containing folder paths")
        print(f"    projection_end_date (optional): Default end date for projections (YYYY-MM-DD)")
        print(f"    year_start_month (optional): Default month when financial year starts ({MIN_MONTH}-{MAX_MONTH}, default: {DEFAULT_YEAR_START_MONTH})")
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
        print("    \"/path/to/folder with spaces\"")
        print("    \"/path/to/folder with spaces\" 2027-12-31 7")
        print("    /path/to/folder with spaces")
        print("    # This is a comment")
        print("\n  Note: Folder paths with spaces can be used with or without quotes.")
        print("        If a folder path ends with a number 1-12, use quotes to avoid ambiguity.")
        sys.exit(1)
    
    first_arg: str = sys.argv[1]
    
    # Check if first argument is a directory (backward compatibility)
    # If it's a directory, treat it as a single folder to process
    if os.path.isdir(first_arg):
        # Old interface: single directory
        folder_path = first_arg
        projection_end_date, year_start_month = parse_command_line_args()
        
        # Process single folder
        print(f"Processing single folder: {folder_path}")
        try:
            process_folder(folder_path, projection_end_date, year_start_month)
        except Exception as e:
            error_exit(f"Error processing {folder_path}: {e}")
        return
    
    # New interface: config file
    # Check if the file exists, if not, maybe the path was split due to spaces
    if not os.path.isfile(first_arg) and not os.path.isdir(first_arg):
        # Check if combining first few arguments might make a valid path
        if len(sys.argv) >= 3:
            potential_path = ' '.join(sys.argv[1:3])
            if os.path.isfile(potential_path) or os.path.isdir(potential_path):
                error_exit(f"Config file or directory not found: '{first_arg}'\n"
                          f"Note: Your path contains spaces. Did you mean: '{potential_path}'?\n"
                          f"Please quote the path in your command: \"{' '.join(sys.argv[1:3])}\"")
        error_exit(f"Config file or directory not found: '{first_arg}'\n"
                  f"If your path contains spaces, please quote it: \"{first_arg}\"")
    
    config_file: str = first_arg
    projection_end_date, year_start_month = parse_command_line_args()

    # Read folder list from config file
    folder_entries: List[Tuple[str, Optional[pd.Timestamp], Optional[int]]] = read_config_file(config_file)
    
    if not folder_entries:
        error_exit(f"No folders found in config file: {config_file}")
    
    print(f"Found {len(folder_entries)} folder(s) to process")
    
    successful = failed = 0
    for folder_path, folder_projection_end_date, folder_year_start_month in folder_entries:
        effective_projection_end_date = folder_projection_end_date or projection_end_date
        effective_year_start_month = folder_year_start_month or year_start_month
        print(f"\nProcessing: {folder_path}")
        if folder_projection_end_date or folder_year_start_month:
            settings = [f"projection_end_date={folder_projection_end_date.strftime('%Y-%m-%d')}" if folder_projection_end_date else None, f"year_start_month={folder_year_start_month}" if folder_year_start_month else None]
            print(f"  Using settings: {', '.join(s for s in settings if s)}")
        try:
            process_folder(folder_path, effective_projection_end_date, effective_year_start_month)
            successful += 1
        except Exception as e:
            print(f"Error processing {folder_path}: {e}")
            failed += 1
    print(f"\n{'='*60}\nProcessing complete: {successful} successful, {failed} failed\n{'='*60}")
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()

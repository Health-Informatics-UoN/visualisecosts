import sys
from typing import Optional, Tuple, List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import matplotlib.ticker as mticker
import re

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

def get_column_sort_key(column_name: str) -> Tuple[float, int]:
    """Get sort key for table columns.
    
    Args:
        column_name: Name of the column to sort
    
    Returns:
        Tuple of (year, is_projected_flag)
    """
    if column_name == 'Total':
        return (float('inf'), 0)
    # Extract the year from the label, e.g., 'May 23/April 24' or 'July 23/June 24'
    year_match = re.search(r'(\d{2,4})[ ]*/', column_name) or re.search(r'(\d{4})', column_name)
    if year_match:
        year_str = year_match.group(1)
        year_value = int(year_str) if len(year_str) == 4 else 2000 + int(year_str)
    else:
        year_value = 0
    is_projected = 1 if 'proj' in column_name else 0
    return (year_value, is_projected)

def error_exit(message: str) -> None:
    """Print error message and exit with code 1."""
    print(f"Error: {message}")
    sys.exit(1)

def main() -> None:
    """Main entry point for the cost visualization script."""
    # Parse command-line arguments for directory, optional projection end date, and year start month
    if len(sys.argv) < 2:
        print("Usage: plot-costs <directory> [projection_end_date] [year_start_month]")
        print("  directory: Path to directory containing .xlsx files")
        print("  projection_end_date (optional): End date for projections (YYYY-MM-DD)")
        print("  year_start_month (optional): Month when financial year starts (1-12, default: 4)")
        sys.exit(1)
    input_dir: str = sys.argv[1]

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

    if not os.path.isdir(input_dir):
        error_exit(f"Path is not a directory: {input_dir}")

    xlsx_files: List[str] = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]
    if not xlsx_files:
        error_exit(f"No .xlsx files found in directory: {input_dir}")

    # Load and combine all .xlsx files
    output_dir: str = os.path.abspath(input_dir)  # Use the folder as output directory
    dataframes_to_combine: List[pd.DataFrame] = []
    for xlsx_file in xlsx_files:
        file_path = os.path.join(input_dir, xlsx_file)
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            dataframes_to_combine.append(df)
        except Exception as e:
            error_exit(f"Failed to read {xlsx_file}: {e}")
    
    if not dataframes_to_combine:
        error_exit("No valid data loaded from Excel files")
    
    expense_data: pd.DataFrame = pd.concat(dataframes_to_combine, ignore_index=True)
    
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in expense_data.columns]
    if missing_columns:
        error_exit(f"Missing required columns: {', '.join(missing_columns)}\nAvailable columns: {', '.join(expense_data.columns)}")
    
    if len(expense_data) == 0:
        error_exit("No data found in Excel files")

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
        error_exit("No monthly data available after processing")

    last_actual_date: pd.Timestamp = monthly_costs_by_category.index[-1]
    try:
        projection_dates, num_projection_months = get_future_dates(last_actual_date, projection_end_date, DEFAULT_PROJECTION_MONTHS)
    except ValueError as e:
        error_exit(str(e))

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
                error_exit(f"Failed to create plot {filename}: {e}")

    # --- Add table with yearly totals for each Expenditure Category ---

    # Combine historical and projected (not cumulative) for yearly sums
    combined_monthly_data: pd.DataFrame = pd.concat([monthly_costs_by_category, monthly_projection])
    # Index is already datetime, no conversion needed
    combined_monthly_data['Month'] = combined_monthly_data.index.to_series().dt.month

    # Set financial year and label based on year_start_month (compute once per date)
    fin_year_results = list(combined_monthly_data.index.map(lambda d: get_fin_year_and_label(d, year_start_month)))
    combined_monthly_data['FinYear'], combined_monthly_data['FinYearLabel'] = zip(*fin_year_results)

    # Get all unique financial years in order
    all_financial_years: List[str] = list(combined_monthly_data['FinYearLabel'].unique())
    # Find the financial year label for the last actual date
    transition_financial_year: Optional[str] = None
    if len(monthly_costs_by_category) > 0:
        last_date_mask = combined_monthly_data.index == monthly_costs_by_category.index[-1]
        if last_date_mask.any():
            transition_financial_year = combined_monthly_data.loc[last_date_mask, 'FinYearLabel'].iloc[0]

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

    # Pre-compute month mask for historical data
    historical_month_mask = combined_monthly_data.loc[monthly_costs_by_category.index, 'Month'] <= last_actual_month_number

    for category_column in categories_for_table:
        yearly_totals_dict[category_column] = {}
        for financial_year_label in all_financial_years:
            if transition_financial_year is not None and financial_year_label < transition_financial_year:
                # Fully actual - use pre-computed mask aligned with historical data
                actual_total: float = monthly_costs_by_category[historical_fin_year_masks[financial_year_label]][category_column].sum()
                yearly_totals_dict[category_column][f"{financial_year_label} (actual)"] = actual_total
            elif transition_financial_year is not None and financial_year_label > transition_financial_year:
                # Fully projected - use pre-computed mask
                projected_total: float = monthly_projection[projection_masks[financial_year_label]][category_column].sum()
                yearly_totals_dict[category_column][f"{financial_year_label} (proj)"] = projected_total
            elif transition_financial_year is not None and financial_year_label == transition_financial_year:
                # Split year: part actual, part projected
                fin_year_mask = historical_fin_year_masks[financial_year_label]
                # Actual part: months up to and including last_actual_month
                actual_portion: float = monthly_costs_by_category[fin_year_mask & historical_month_mask][category_column].sum()
                # Projected part: use pre-computed mask
                projection_mask = projection_masks[financial_year_label] & (monthly_projection.index > last_actual_date)
                projected_portion: float = monthly_projection[projection_mask][category_column].sum()
                yearly_totals_dict[category_column][f"{financial_year_label} (actual)"] = actual_portion
                yearly_totals_dict[category_column][f"{financial_year_label} (proj)"] = projected_portion

    # Convert to DataFrame for pretty printing and round all values to 2 decimal places
    yearly_totals_table: pd.DataFrame = pd.DataFrame(yearly_totals_dict).T.fillna(0).round(2)
    yearly_totals_table['Total'] = yearly_totals_table.sum(axis=1).round(2)

    # Add a total row at the bottom
    total_row = yearly_totals_table.sum(axis=0).round(2)
    total_row.name = 'Total'
    yearly_totals_table = pd.concat([yearly_totals_table, total_row.to_frame().T])

    yearly_totals_table = yearly_totals_table.reindex(sorted(yearly_totals_table.columns, key=get_column_sort_key), axis=1)

    csv_output_path: str = os.path.join(output_dir, f"{base_filename}_yearly_totals_by_expenditure_category.csv")
    try:
        yearly_totals_table.to_csv(csv_output_path, float_format='%.2f')
    except Exception as e:
        error_exit(f"Failed to save CSV file: {e}")

if __name__ == "__main__":
    main()

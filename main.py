from gen_df import process_finance_data
from MAD_optimizer import clean_returns_data, mad_portfolio_optimization, display_portfolio_results
from plot_frontier import plot_efficient_frontier

file_path = '/home/theorist/Desktop/Programming/python/sc_assig/FinanceDataForAssignment.xlsx'
prices_df, returns_df = process_finance_data(file_path)
if 'Unnamed: 34' in prices_df.columns:
    prices_df.drop(columns=['Unnamed: 34'], inplace=True)

if 'Unnamed: 34' in returns_df.columns:
    returns_df.drop(columns=['Unnamed: 34'], inplace=True)

# print("Stock Prices DataFrame:")
# print(prices_df)
# print("\nExpected Returns DataFrame:")
# print(returns_df)

cleaned_market_returns = returns_df['Market Portfolio.1'].dropna()

target_return = 0.006
print(f"Target return: {target_return:.4f}")

results = mad_portfolio_optimization(returns_df, target_return)
display_portfolio_results(results)

# Define the min and max target returns based on your data
min_return = clean_returns_data(returns_df).mean().min()  # Minimum expected return
max_return = clean_returns_data(returns_df).mean().max()  # Maximum expected return

# Plot the efficient frontier
plot_efficient_frontier(returns_df, min_return, max_return)

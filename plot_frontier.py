import matplotlib.pyplot as plt
from MAD_optimizer import mad_portfolio_optimization, display_portfolio_results
import numpy as np


def plot_efficient_frontier(returns_df, min_return, max_return, step_size=0.0005):
    target_returns = np.arange(min_return, max_return, step_size)
    mad_risks = []
    portfolio_returns = []

    for target_return in target_returns:
        results = mad_portfolio_optimization(returns_df, target_return)
        display_portfolio_results(results)
        if results['status'] == 'Optimal':
            mad_risks.append(results['portfolio_mad'])
            portfolio_returns.append(results['portfolio_return'])
        else:
            # In case optimization fails for a target return, use NaN
            mad_risks.append(np.nan)
            portfolio_returns.append(np.nan)

    # Plotting the efficient frontier
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_returns, mad_risks, label='Efficient Frontier', color='b')
    plt.xlabel('Expected Return')
    plt.ylabel('MAD Risk')
    plt.title('Efficient Frontier (MAD Optimization)')
    plt.grid(True)
    plt.show()

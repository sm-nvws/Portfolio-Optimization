This portfolio simulator models and compares actual vs optimized stock allocations using Monte Carlo methods. It pulls stock data from FinancialModelingPrep (FMP) with caching support to reduce API load. Each stock's expected return and volatility is derived from its beta, and paths are simulated using correlated t-distributed noise to better capture market extremes.

The simulation uses 1,000 trials over 252 days to assess risk metrics: Value at Risk (VaR), Expected Shortfall (ES), annualized return, volatility, and Sharpe ratio. Optimization is performed with a genetic algorithm from DEAP, constrained to valid portfolio weights summing to 1. The tool prints summary stats and visualizes the portfolio value distributions. It also suggests rebalancing actions based on the optimized weights, highlighting under-allocated assets.

Run simulate_portfolio_dual({...}, "YYYY-MM-DD") with your ticker-share dictionary to analyze your portfolio.








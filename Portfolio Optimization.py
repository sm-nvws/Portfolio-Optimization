import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from pandas.tseries.offsets import BDay
from deap import base, creator, tools, algorithms
import random
from datetime import date
import time

np.random.seed(42)
num_simulations = 1000  
num_days = 252  
time_horizon = 0.00273972602 
risk_free_rate = float(input("What is the risk-free rate?"))

mean_returns = []
volatilities = []
initial_prices = []

def get_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    stock_data['Returns'] = stock_data['Adj Close'].pct_change()
    stock_data = stock_data.dropna()
    if stock_data.empty:
        raise ValueError(f"No data for {ticker} between {start} and {end}.")
    mean_return = stock_data['Returns'].mean()
    volatility = stock_data['Returns'].std()
    return mean_return, volatility

def adjust_to_business_day(date_input):
    date = pd.to_datetime(date_input)
    return date if pd.Timestamp(date).isoweekday() < 6 else date + BDay(1)

def simulate_correlated_price_paths(S0, mu, sigma, corr_matrix, num_days, num_simulations):
    dt = time_horizon / num_days
    L = np.linalg.cholesky(corr_matrix)
    num_assets = len(S0)
    price_paths = np.zeros((num_simulations, num_assets, num_days))
    price_paths[:, :, 0] = S0
    for t in range(1, num_days):
        Z = np.random.standard_normal((num_simulations, num_assets))  
        correlated_randoms = np.dot(L, Z.T).T
        for i in range(num_assets):
            price_paths[:, i, t] = price_paths[:, i, t-1] * np.exp((mu[i] - 0.5 * sigma[i] ** 2) * dt + sigma[i] * np.sqrt(dt) * correlated_randoms[:, i])
    return price_paths

def get_correlation_matrix(stocks, start, end):
    stock_data = {}
    for stock in stocks:
        time.sleep(1)
        stock_data[stock] = yf.download(stock, start=start, end=end)['Adj Close'].pct_change().dropna()
    returns_df = pd.DataFrame(stock_data)
    correlation_matrix = returns_df.corr().values
    return correlation_matrix

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def fitness(individual):
    weights = np.array(individual)
    portfolio_return = np.dot(mean_returns, weights)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
    return sharpe_ratio,

def check_sum(individual):
    return sum(individual) == 1

def ensure_two_offspring(offspring):
    if len(offspring) < 2:
        return offspring * 2
    return offspring

def ensure_individual_format(offspring):
    new_offspring = []
    for ind in offspring:
        if isinstance(ind, float):
            ind = [ind]
        if len(ind) != len(mean_returns):
            ind = np.random.dirichlet(np.ones(len(mean_returns)), size=1).tolist()[0]
        new_offspring.append(creator.Individual(ind))
    return new_offspring

def optimize_portfolio_deap():
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(mean_returns))
    
    def even_population(n):
        return n if n % 2 == 0 else n + 1
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.decorate("mate", tools.DeltaPenalty(check_sum, 1.0))
    toolbox.decorate("mutate", tools.DeltaPenalty(check_sum, 1.0))

    population = toolbox.population(n=even_population(300))

    for generation in range(100):
        offspring = list(map(toolbox.clone, population))
        for i in range(1, len(offspring), 2):
            if random.random() < 0.7:
                offspring[i - 1], offspring[i] = ensure_two_offspring(toolbox.mate(offspring[i - 1], offspring[i]))
            offspring = ensure_individual_format(offspring)
            if random.random() < 0.2:
                toolbox.mutate(offspring[i - 1])
                toolbox.mutate(offspring[i])
            del offspring[i - 1].fitness.values
            del offspring[i].fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population = toolbox.select(offspring, len(population))

    best_individual = tools.selBest(population, 1)[0]
    return np.array(best_individual)

entrance = input("Press Y to begin: ").upper()

our_portfolio = {}

while entrance == "Y":
    stock = input("Enter the stock ticker: ")
    shares = input("Number of shares: ")
    try:
        shares = int(shares)
        our_portfolio[stock] = shares
    except ValueError:
        print("Invalid number of shares. Please enter an integer value.")
        continue
    checker = input("Add more? Press Y if yes, or any other key to finish: ").upper()
    if checker != "Y":
        break

start_input = input("Start? (format YYYY-MM-DD): ")
end_input = date.today()

try:
    adjusted_start_date = adjust_to_business_day(start_input)
    adjusted_end_date = adjust_to_business_day(end_input)
except Exception as e:
    print(f"Error adjusting dates: {e}")
    exit(1)

for ticker, shares in our_portfolio.items():
    try:
        mean_return, volatility = get_stock_data(ticker, start=adjusted_start_date, end=adjusted_end_date)
        mean_returns.append(mean_return)
        volatilities.append(volatility)
        initial_data = yf.download(ticker, start=adjusted_start_date, end=adjusted_end_date + BDay(1))
        if not initial_data.empty:
            initial_price = initial_data['Adj Close'].values[0]
            initial_prices.append(initial_price)
        else:
            raise ValueError(f"No data available for {ticker} on specified dates.")
    except Exception as e:
        print(f"Error processing data for {ticker}: {e}")
        initial_prices.append(0)

mean_returns = np.array(mean_returns)
volatilities = np.array(volatilities)
initial_prices = np.array(initial_prices)

if len(mean_returns) == 0 or len(volatilities) == 0:
    print("No valid stock data available. Exiting.")
    exit(1)

correlation_matrix = get_correlation_matrix(our_portfolio.keys(), adjusted_start_date, adjusted_end_date)

cov_matrix = np.diag(volatilities**2)

optimal_weights = optimize_portfolio_deap()

simulated_paths = simulate_correlated_price_paths(initial_prices, mean_returns, volatilities, correlation_matrix, num_days, num_simulations)
final_portfolio_values = np.sum(simulated_paths[:, :, -1] * optimal_weights, axis=1)

VaR = np.percentile(final_portfolio_values, 95)
expected_shortfall = np.mean(final_portfolio_values[final_portfolio_values <= VaR])

portfolio_returns = final_portfolio_values / np.sum(initial_prices * np.array(list(our_portfolio.values()))) - 1
annualized_portfolio_return = np.mean(portfolio_returns) * num_days / time_horizon
portfolio_volatility = np.std(portfolio_returns) * np.sqrt(num_days)
sharpe_ratio = (annualized_portfolio_return - risk_free_rate) / portfolio_volatility

plt.figure(figsize=(10, 6))
plt.hist(final_portfolio_values, bins=50, alpha=0.6, color='b')
plt.axvline(x=VaR, color='r', linestyle='--', label=f'VaR (5%): ${VaR:,.2f}')
plt.axvline(x=expected_shortfall, color='g', linestyle='--', label=f'ES (Expected Shortfall): ${expected_shortfall:,.2f}')
plt.title(f"Distribution of Portfolio Values after 1 Year\n(Initial Portfolio Value: ${np.sum(initial_prices):,.2f})")
plt.xlabel('Portfolio Value')
plt.ylabel('Frequency')
plt.legend()
plt.show(block=False)

print(f"Initial Portfolio Value: ${np.sum(initial_prices):,.2f}")
print(f"Value at Risk (VaR) at 95% confidence level: ${VaR:,.2f}")
print(f"Expected Shortfall (ES) beyond VaR: ${expected_shortfall:,.2f}")
print(f"Annualized Portfolio Return: {annualized_portfolio_return:.2%}")
print(f"Portfolio Volatility (Standard Deviation): {portfolio_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

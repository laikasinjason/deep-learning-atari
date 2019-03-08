# deep-q-learning-trading-system-on-hk-stocks-market
Deep q learning on determining buy/sell signal and placing orders



----------
This project is inspired by the paper: 
[A Multi-agent Q-learning Framework for Optimizing Stock Trading Systems](https://link.springer.com/chapter/10.1007/3-540-46146-9_16) 


Trading System Structure
------------------------
The trading system takes 4 agents: buy signal agent, buy order agent, sell order agent, sell signal agent.

  * Buy signal agent makes the long decision by considering the state of stocks in current day
  * Buy order agent determines the buy price after buy signal agent gave the buy signal
  * Sell signal agent makes the short decision ( after holding the stocks, no short sell )
  * Sell order agent determines the sell price after sell signal agent gave the sell signal
  

States of stocks
----------------
#### Buy/ Sell signal agents
This strategy makes use of the price difference between the current price to the previous high/low turning points to determine the buy/ sell signal

#### Buy/ Sell order agents
Different technical indicators, e.g. sma, high/low difference...

Sell order agent contains additional parameters: e.g. profit

Agent's actions
---------------
#### Buy signal agent
Buy, Not Buy

#### Sell signal agent
Sell, Hold

#### Buy Order agent
Percentage of price of ma(5) of T-1 day
Buy price is based on the equation: bp = ma5 + action / 100 * ma5
Buy order is triggered on T day, if bp is lower than low price, it won't be executed.

#### Sell Order agent
Percentage of price of ma(5) of T-1 day
Sell price is based on the equation: sp = ma5 + action / 100 * ma5
Sell order is triggered on T day, if sp is higher than high price, sell order will be executed at close.


System flow
-----------
1) State of stocks is feed into the Buy Signal Agent
    BUY:
        Invoke Buy Order Agent
    NOT-BUY:
        Go to 1)
2) Buy Order Agent takes the state of same day BSA gives the buy signal. A buy price is determined
    If order is successfully executed:
        Invoke Sell Signal agent
    Else:
        Go back to 1)
3) Sell Signal Agent starts taking states of stocks a day after the buy order is executed
    SELL:
        Invoke Sell Signal Agent
    HOLD:
        Go to 3)
4) Sell Order Agent takes the state of same day SSA gives the sell signal. A sell price is determined
   If sell price is not met, it should be executed at close price.
   Go to 1)

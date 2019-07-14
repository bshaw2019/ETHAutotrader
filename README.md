# ETH_autotrader

## Guidelines

**DON'T SHARE/MARKET THIS CODE WITH ANYONE**

## Instructions

1. Run setup.py

2. Comment out [sending email](https://github.com/and-rewsmith/ETH_autotrader/blob/server/fit_functions.py#L333) and [logging to twitter](https://github.com/and-rewsmith/ETH_autotrader/blob/server/fit_functions.py#L444). This functionality will only work if you have my passwords.

3. Open initial_fit.py and select the parameters you desire (i.e. num_timesteps and num_targets). ```num_timesteps``` is an input to the neural network where ```num_targets``` represents an output.

4. Run initial_fit.py.

5. You can now run online_fit.py. This will call the ```online_fit()``` function. This function will return the future predictions (how many will depend on your selected value for ```num_targets```). Since this will no longer log to Twitter you'll need to edit this file to do some sort of logging yourself. I would suggest calling print() on the function itself (online_fit).


## Directory Structure

**got3/** - Directrory containing an open source tool to get tweets without using Twitter's official API. Twitter's official API only returns tweets up to two weeks old which will not do.

**model/model.py** - Contains a function to build the model.

**date_handler.py** - Contains function to bypass the official Twitter API by scraping the website with rotating proxies.

**fit_functions.py** - Contains functions to initial/online fit the model along with other utility functions.

**get_db_info.py** - Prints every record of embedded sqlite3 database to stdout.

**initial_fit.py** - Fits the model on the past few years of price/sentiment data.

**online_fit.py** - Fits the model since the last performed fit (be it initial or online).

**proxy_selector.py** - Contains function to select a proxy from a list of free proxies.

**requirements.txt** - Contains all the dependencies needed to run the project. Use pip to install the dependencies.

**send_email.py** - Contains a function to send an email notification containing a graph representing the initial fit. Called from the initial_fit() function.

**setup.py** - Run this script to setup the project for use.

**twitter_logger.py** - Contains function to log predictions to twitter. This function is called after every online_fit.

## Roadmap (order doesn't matter here)

1. Perform an initial fit on a model that takes in more tweets and only predicts for one timestep.

2. Start logging to twitter and observe winrate.

3. Fix online_fit learning rate.

4. Add features to dataset.

5. Implement q learning.

6. Link to some exchange's trading API (either ETH/USD or ETH/Tether).

7. Smart contract implementation.



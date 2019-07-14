import matplotlib
matplotlib.use('Agg')
import math
import matplotlib.pyplot as plt
import numpy as np
from model.model import build_model
from model.model import save_model
from model.model import load_model
import time
import arrow
from send_email import send_email
import ccxt
from date_handler import dates_to_sentiment
from twitter_logger import log_to_twitter
import sqlite3
import os.path
import keras




np.set_printoptions(suppress=True)




def get_historical(num_tweets, from_date="", is_online=False):

    hour = 3600000

    bitfinex = ccxt.bitfinex()

    from_datetime = None
    if from_date == "":
        from_datetime = '2017-02-05 00:00:00'
    else:
        from_datetime = arrow.get(from_date).format('YYYY-MM-DD HH:mm:ss')

    #from_datetime = '2018-02-14 00:00:00'

    from_timestamp = bitfinex.parse8601(from_datetime)
    now = bitfinex.milliseconds()
    data = []

    while from_timestamp < now:
        print('Fetching candles starting from', bitfinex.iso8601(from_timestamp))

        ohlcvs = bitfinex.fetch_ohlcv('ETH/USD', '1h', from_timestamp)

        # don't hit the rateLimit or you will be banned
        time.sleep(10 * bitfinex.rateLimit / 1000)

        from_timestamp += len(ohlcvs) * hour

        data.extend(ohlcvs)

    data = np.asarray(data)

    dates = np.copy(data[:, 0])
    dates = [arrow.get(float(date)/1000).format("YYYY-MM-DD") for date in dates]
    sentiments_abbr = dates_to_sentiment(dates, "ETH", num_tweets)
    sentiments_verb = dates_to_sentiment(dates, "Ethereum", num_tweets)

    prices = data[:, 1:-2]
    volumes = data[:, -1]
    targets = data[:, -2]

    out = np.column_stack((data[:,0], prices, volumes, sentiments_abbr[:, 0], sentiments_abbr[:, 1], sentiments_verb[:, 0], sentiments_verb[:, 1], targets))

    dir_path = os.path.dirname(os.path.abspath(__file__))

    conn = sqlite3.connect(os.path.join(dir_path, 'historical.db'))
    cursor = conn.cursor()

    #The from_date argument specifies that this will be a partial fit (i.e. insert rows into table instead of dropping/recreating table)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    if (len(cursor.fetchall()) != 0 and from_date == ""):
        cursor.execute("""DROP TABLE historical""")

    if from_date == "":
        cursor.execute("""CREATE TABLE historical (
                    date text,
                    open real,
                    high real,
                    low real,
                    volume real,
                    sent_abbrev_pol real,
                    sent_abbrev_subj real,
                    sent_verb_pol real,
                    sent_verb_subj real,
                    close real
                    )""")

    for count, date in enumerate(data[:,0]):
        if is_online:
            cursor.execute("SELECT * FROM historical WHERE date={};".format(date))
            if len(cursor.fetchall()) > 0:
                print("WARNING: DUPLICATE ENTRIES IN SQLITE HISTORICAL DB")
                continue
        open_price = prices[count][0]
        high_price = prices[count][1]
        low_price = prices[count][2]
        volume = volumes[count]
        sent_abbrev_pol = sentiments_abbr[count][0]
        sent_abbrev_subj = sentiments_abbr[count][1]
        sent_verb_pol = sentiments_verb[count][0]
        sent_verb_subj = sentiments_verb[count][1]
        close = targets[count]

        cursor.execute("INSERT INTO historical VALUES ('{}', {}, {}, {}, {}, {}, {}, {}, {}, {})".format(data[count][0], open_price, high_price, low_price, volume, sent_abbrev_pol, sent_abbrev_subj, sent_verb_pol, sent_verb_subj, \
            close))

    cursor.execute("SELECT * FROM historical")
    db_data = cursor.fetchall()

    conn.commit()
    conn.close()

    return out




def normalize_timestep(timestep, reference_list):

    reference_price = timestep[0][0]
    reference_volume = timestep[0][3]

    temp_prices = np.copy(timestep[:, 0:3])
    temp_volume = np.copy(timestep[:, 3])
    temp_sent = np.copy(timestep[:, 4:8])
    temp_close = np.copy(timestep[:, -1])

    temp_prices = (temp_prices / float(reference_price)) - 1
    temp_volume = (temp_volume / float(reference_volume)) - 1
    temp_close = (temp_close / float(reference_price)) - 1

    reference_list.append(reference_price)
    out = np.column_stack((temp_prices, temp_volume, temp_sent, temp_close))

    return out




#forms timeseries AND removes date from the timestep
def split_into_timeseries(data, combined_length):

    result = []
    for index in range(len(data) - combined_length + 1): 
        #validates timestep dateranges
        last_date = None
        should_continue = False
        for i in range(index, index + combined_length):
            if last_date == None:
                last_date = data[index][0]
                continue
            elif data[index][0]-last_date >= 10800000:
                print("WARNING: Huge time difference in timestep. Throwing out this timestep.")
                should_continue = True 
                break

        if should_continue == True:
            continue
        else:
            time_series = data[index: index + combined_length, 1:]
            result.append(time_series[:])

    result = np.asarray(result)
    return result




#take data and split into timeseries so that we can train the model
def load_data(data, num_timesteps, num_targets, train_percent=.93):

    # iterate so that we can also capture a sequence for a target
    combined_length = num_timesteps + num_targets

    print("SPLITTING INTO TIMESERIES")

    result = split_into_timeseries(data, combined_length)

    # normalize
    reference_points = [] #for de-normalizing outside of the function
    for i in range(0, len(result)):
        result[i] = normalize_timestep(result[i], reference_points)


    # train test split
    row = round(train_percent * result.shape[0])
    train = result[:int(row), :]
    test = result[int(row):, :]

    split_index = len(train[0]) - num_targets
    x_train = train[:, :split_index]
    y_train = train[:, split_index:, -1]

    x_test = test[:, :split_index]
    y_test = test[:, split_index:, -1]

    return [x_train, y_train, x_test, y_test, reference_points]




def initial_fit(num_timesteps, num_targets, train_percent=.93, num_tweets=300):
    print("started init fit")
    dir_path = os.path.dirname(os.path.abspath(__file__))

    #clear contents of log files
    open(os.path.join(dir_path, 'logs/context_prices.txt'), 'w').close()
    open(os.path.join(dir_path, 'logs/actuals.txt'), 'w').close()
    open(os.path.join(dir_path, 'logs/predictions.txt'), 'w').close()
    open(os.path.join(dir_path, 'logs/history.txt'), 'w').close()
    open(os.path.join(dir_path, 'logs/proxy_log.txt'), 'w').close()

    data = get_historical(num_tweets, from_date="")

    X_train, y_train, X_test, y_test, ref = load_data(data, num_timesteps, num_targets=num_targets, train_percent=train_percent) #TODO: make higher percentage of training when this goes into "prod"

    # store recent data so that we can get a live prediction
    recent_reference = []
    recent_data = data[-num_timesteps:, 1:]
    recent_data = normalize_timestep(recent_data, recent_reference)

    print("    X_train", X_train.shape)
    print("    y_train", y_train.shape)
    print("    X_test", X_test.shape)
    print("    y_test", y_test.shape)
    
    model = build_model([9, num_timesteps, num_targets])
    #train the model
    print("TRAINING")
    model.fit(
    X_train,
    y_train,
    batch_size=512,
    epochs=600,
    validation_split=0.1,
    verbose=2)
    save_model(model)

    trainScore = model.evaluate(X_train, y_train, verbose=100)
    print('Train Score: %.2f MSE (%.2f RMSE) (%.2f)' % (trainScore[0], math.sqrt(trainScore[0]), trainScore[1]))

    testScore = model.evaluate(X_test, y_test, verbose=100)
    print('Test Score: %.2f MSE (%.2f RMSE) (%.2f)' % (testScore[0], math.sqrt(testScore[0]), testScore[1]))

    #make predictions
    print("PREDICTING")
    p = model.predict(X_test)

    recent_data = [recent_data] # One-sample predictions need list wrapper. Argument must be 3d.
    recent_data = np.asarray(recent_data)
    future = model.predict(recent_data)

    # document results in file
    print("WRITING TO LOG")
    file = open(os.path.join(dir_path, "logs/log_initial.txt"), "w")
    for i in range(0, len(X_train)):
        for s in range(0, num_timesteps):
            file.write(str(X_train[i][s]) + "\n")
        file.write("Target: " + str(y_train[i]) + "\n")
        file.write("\n")

    for i in range(0, len(X_test)):
        for s in range(0, num_timesteps):
            file.write(str(X_test[i][s]) + "\n")
        file.write("Target: " + str(y_test[i]) + "\n")
        file.write("Prediction: " + str(p[i]) + "\n")
        file.write("\n")
    file.close()

    # de-normalize
    print("DENORMALIZING")
    for i in range(0, len(p)):
        p[i] = (p[i] + 1) * ref[round(.9 * len(ref) + i)]
        y_test[i] = (y_test[i] + 1) * ref[round(.9 * len(ref) + i)]

    future[0] = (future[0] + 1) * recent_reference[0]
    recent_data[0] = (recent_data[0] + 1) * recent_reference[0]

    file = open(os.path.join(dir_path, "logs/predictions.txt"), "a")
    file.write(str(future[0][0]) + "\n")
    file.close()

    # plot historical predictions
    print("PLOTTING")
    for i in range(0, len(p)):
        if i % (num_targets*2) == 0:
            plot_index = i #for filling plot indexes
            plot_indexes = []
            plot_values = p[i]
            for j in range(0, num_targets):
                plot_indexes.append(plot_index)
                plot_index += 1
            plt.plot(plot_indexes, plot_values, color="red")

    # plot historical actual
    plt.plot(y_test[:, 0], color='blue', label='Actual') # actual price history

    # plot recent prices
    plot_indexes = [len(y_test) - 1]
    plot_values = [y_test[-1, 0]]
    plot_index = None
    for i in range(0, len(recent_data[0])):
        plot_values.append(recent_data[0][i][0])
        plot_index = len(y_test) + i
        plot_indexes.append(len(y_test)+i)
    plt.plot(plot_indexes, plot_values, color='blue')

    # plot future predictions
    plot_indexes = [plot_index]
    plot_values = [recent_data[0][-1][0]]
    for i in range(0, len(future[0])):
        plot_index += 1
        plot_values.append(future[0][i])
        plot_indexes.append(plot_index)
    plt.plot(plot_indexes, plot_values, color="red", label="Prediction")

    #show/save plot
    print("SENDING EMAILS")
    plt.legend(loc="upper left")
    plt.title("ETH Price Predictions")
    plt.xlabel("Hours")
    plt.ylabel("Price ($)")
    filename = str(arrow.utcnow().format("YYYY-MM-DD"))
    plt.savefig(os.path.join(dir_path, "graphs/" + filename))
    #plt.show()
    plt.close()
    send_email()

    return




def stabilize_logs():
    dir_path = os.path.dirname(os.path.abspath(__file__))

    #clear contents of log files
    open(os.path.join(dir_path, 'logs/context_prices.txt'), 'w').close()
    open(os.path.join(dir_path, 'logs/actuals.txt'), 'w').close()
    
    file = open(os.path.join(dir_path, 'logs/predictions.txt'), "r")
    lines = file.readlines()
    last_prediction = lines[-1]
    file.close()

    file = open(os.path.join(dir_path, 'logs/predictions.txt'), "w")
    file.write(last_prediction) #last prediction already has a \n
    file.close()




def online_fit(num_timesteps, num_targets, num_tweets=300):

    stabilize_logs()

    dir_path = os.path.dirname(os.path.abspath(__file__))

    conn = sqlite3.connect(os.path.join(dir_path, 'historical.db'))

    cursor = conn.cursor()

    #for debugging to simulate a 1 hour pass time
    #cursor.execute("DELETE FROM historical ORDER BY date DESC LIMIT 1")

    cursor.execute("SELECT * FROM historical ORDER BY date DESC LIMIT 1")
    last_record = cursor.fetchall()
    from_date = arrow.get((float(last_record[0][0]) + 3600000)/1000).format('YYYY-MM-DD HH:mm:ss')

    combined_length = num_timesteps + num_targets
    cursor.execute("SELECT * FROM historical ORDER BY date DESC LIMIT {}".format(combined_length - 1)) # need to fit with some data in the db as the model didn't fit itself with said data on the past fit
    precomputed_data = np.asarray(cursor.fetchall(), dtype=np.float32)
    precomputed_data = precomputed_data[::-1]

    file = open(os.path.join(dir_path, "logs/context_prices.txt"), "a")
    file.write(str(precomputed_data[-1][-1]) + "\n")
    file.close()

    conn.commit()
    conn.close()

    unseen_data = get_historical(num_tweets, from_date=from_date, is_online=True)

    #actual price from last prediction used for logging with twitter
    actual_price = unseen_data[0][-1]
    file = open(os.path.join(dir_path, "logs/actuals.txt"), "a")
    file.write(str(actual_price) + "\n")
    file.close()

    all_data = np.concatenate((precomputed_data, unseen_data), axis=0)

    # store recent data so that we can get a live prediction
    recent_reference = []
    recent_data = all_data[-num_timesteps:, 1:]
    recent_data = normalize_timestep(recent_data, recent_reference)

    timesteps = split_into_timeseries(all_data, combined_length)

    reference = []
    for i in range(0, len(timesteps)):
        timesteps[i] = normalize_timestep(timesteps[i], reference)

    split_index = len(timesteps[0]) - num_targets
    X_train = timesteps[:, :split_index]
    y_train = timesteps[:, split_index:, -1]

    model = load_model()

    #train the model
    print("TRAINING")
    model.fit(
    X_train,
    y_train,
    batch_size=512,
    epochs=10,
    validation_split=0,
    verbose=2)
    save_model(model)

    recent_data = np.asarray([recent_data.tolist()])

    future = model.predict(recent_data)
    predictions = (future[0] + 1) * recent_reference[0]
    recent_data[0] = (recent_data[0] + 1) * recent_reference[0]

    # document results in file
    print("WRITING TO LOG")
    file = open(os.path.join(dir_path, "logs/log_online.txt"), "w")
    for timestep in recent_data:
        file.write(str(timestep) + "\n")
    file.write(str(future[0]) + "\n")
    file.close()

    file = open(os.path.join(dir_path, "logs/predictions.txt"), "a")
    file.write(str(predictions[0]) + "\n")
    file.close()

    log_to_twitter(predictions)

    return predictions

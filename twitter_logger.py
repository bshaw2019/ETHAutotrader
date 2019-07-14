import numpy as np
import tweepy
import os.path

def log_to_twitter(predictions):

	dir_path = os.path.dirname(os.path.abspath(__file__))


	file = open(os.path.join(dir_path, "logs/actuals.txt"), "r")
	count = -1
	actual_price = None
	for line in file:
		actual_price = float(line)
		count += 1
	file.close()

	file = open(os.path.join(dir_path, "logs/predictions.txt"), "r")
	predicted_price = None
	for i, line in enumerate(file):
		if i == count:
			predicted_price = float(line)
			break
	file.close()

	file = open(os.path.join(dir_path, "logs/context_prices.txt"), "r")
	context_price = None
	for i, line in enumerate(file):
		if i == count:
			context_price = float(line)
			break
	file.close()

	out = ""

	if np.sign(float(predicted_price) - float(context_price)) == np.sign(float(actual_price) - float(context_price)):
		out += "PAST PREDICTION SUCCESS DETAILS:\n"
		file = open(os.path.join(dir_path, "logs/history.txt"), "a")
		file.write("W" + "\n")
		file.close()
	else:
		out += "PAST PREDICTION FAILURE DETAILS:\n"
		file = open(os.path.join(dir_path, "logs/history.txt"), "a")
		file.write("L" + "\n")
		file.close()

	file = open(os.path.join(dir_path, "logs/history.txt"), "r")
	wins = 0
	num_predictions = 0
	for line in file:
		num_predictions += 1
		if line.rstrip("\n") == "W":
			wins += 1
	win_rate = float(wins)/num_predictions

	out += "previous actual: " + str(context_price) + "\n"
	out += "next hour prediction: " + str(predicted_price) + "\n"
	out += "next hour actual: " + str(actual_price) + "\n"
	out += "\n"
	out += "FUTURE PREDICTIONS:" + "\n"
	out += str(predictions) + "\n"
	out += "\n"
	out += "WIN RATE:" + "\n"
	out += str(win_rate) + "\n"

	file = open(os.path.join(dir_path, "logs/keys.txt"))
	keys = file.readlines()

	access_token = keys[3].rstrip("\n")
	access_token_secret = keys[4].rstrip("\n")
	consumer_key = keys[1].rstrip("\n")
	consumer_secret = keys[2].rstrip("\n")

	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth)
	api.update_status(status=out)










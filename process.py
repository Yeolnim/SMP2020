import csv
import json

file_path = "./result/usual_result.csv"
csvfile = open(file_path, 'r')
jsonfile = open('./result/usual_result.json', 'w')

reader = csv.DictReader(csvfile)

out = json.dumps([row for row in reader])

jsonfile.write(out)

file_path = "./result/virus_result.csv"
csvfile = open(file_path, 'r')
jsonfile = open('./result/virus_result.json', 'w')

reader = csv.DictReader(csvfile)

out = json.dumps([row for row in reader])

jsonfile.write(out)
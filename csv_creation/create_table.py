text = input()
name = input()
files = text.split("\n")

elements_for_csv = []

for file in files: 
  elements = file.split("|")
  for char in ["|", " ", "", "[model]", "[log]","[config]", "(", ")"]: 
    for index in range(len(elements)): 
      elements[index] = elements[index].replace(char, "")
  
  config_file = elements[9].split("master/")[1]
  checkpoint_link = elements[10].split("&#124;")[0].replace("&#124;", "")
  checkpoint_file = checkpoint_link.split("/")[-1]
  
  elements_for_csv.append([checkpoint_link, checkpoint_file, config_file])
  

import pandas as pd

file_name = name + ".csv"
csv = pd.DataFrame(elements_for_csv, columns=['checkpoint_link', 'checkpoint_file', 'config_file'])
csv.to_csv(file_name)


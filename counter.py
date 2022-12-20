import os
K = "8"

main_dir = "./KlebsiellaPneumoniae/" 
for file_index, filename in enumerate(os.listdir(main_dir + "raw_data")):    
    print("File number: ", file_index, filename)
    os.system("./KlebsiellaPneumoniae/tools/kmc -k" + K + " -ci0 -cs500000 -fm " + main_dir + "raw_data/" + filename + " ./counts_results_" + K + "/" + filename +  " ./")
    os.system("./KlebsiellaPneumoniae/tools/kmc_dump ./counts_results_" + K + "/" + filename +  " ./counts_data_" + K + "/" + '.'.join(filename.split('.')[0:-1]))
    

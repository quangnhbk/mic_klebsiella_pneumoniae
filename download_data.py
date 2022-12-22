import os.path
import os

f = open("./genomes.txt", "r")
line = f.read()
print(line)

i = 0
for file_name in line.split(" "):
    file_name = file_name.strip()
    print(file_name)
    if len(file_name) == 0: continue
    
    file_fna = "./raw_data/" + file_name + ".fna"
    print(file_fna)
    
    # Check file_fna exists
    if os.path.exists(file_fna): continue
    
    # Download file_fna
    print("Download file ", file_fna)
    download_command = "wget " + "ftp://ftp.patricbrc.org/genomes/" + file_name + "/"+ file_name + ".fna"
    print(download_command)
    os.system(download_command)
    
    i = i + 1    
    
        
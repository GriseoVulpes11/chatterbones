import os
def read_file():
    with open("/Users/riley/Desktop/said.txt", "r") as f:
        SMRF1 = f.readlines()
    return SMRF1

print(os.getcwd())
initial = read_file()
while True:
    current = read_file()
    if initial != current:
        for line in current:
            if line not in initial:
                print(line)
                #line in chat function
                os.system('scp -i /Users/riley/Desktop/stb.pem /Users/riley/Desktop/said.txt ec2-user@ec2-44-197-251-84.compute-1.amazonaws.com:~/environment/chatterbones/src/chat')

        initial = current
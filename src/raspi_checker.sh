#!/bin/sh
previous="$(cat /home/pi/Desktop/ChatterBones/said.txt)"
while :
do
  sleep 10
  content="$(cat /home/pi/Desktop/ChatterBones/said.txt)"
  if [ "$previous" != "$content" ];then
      echo "Sending data now"
      echo "Sending the text: $(echo "$content" | tr '[:upper:]' '[:lower:]')"
      scp -i /Users/riley/Desktop/stb.pem /Users/riley/Desktop/said.txt ec2-user@ec2-35-175-223-75.compute-1.amazonaws.com:~/environment/chatterbones/src/chat
fi
  previous="$content"
done
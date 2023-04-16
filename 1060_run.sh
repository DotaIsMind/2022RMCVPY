#!/bin/bash
echo "$(date) Autoaim process starting" >> /home/dota/RMCV/bash_data/1060_sh.log
#cd /home/dota/RMCV/20221YNUrmcv_main
#while true
#do
#  echo "123456" | sudo -S chmod 777 /dev/ttyUSB0 | python3 rmcv_main.py | tee "../bash_data/$(date).txt" 2>&1
  echo "123456" | sudo -S chmod 777 /dev/ttyUSB0 | python3 rmcv_main.py
#  /home/dota/anaconda3/envs/rmcv/bin/python3 rmcv_main.py | tee "../bash_data/$(date).txt" 2>&1
  if [ $? -eq 0 ]; then
    echo "succeed!" >> /home/dota/RMCV/bash_data/1060_sh.log
    exit 0
  else
    echo "failed!" >> /home/dota/RMCV/bash_data/1060_sh.log
  fi
#done

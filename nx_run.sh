#!/bin/bash
#echo "$(date) Start NX max power" >> /home/nx/rc_local_temp/bash_data/nx_sh.log
#echo "123" | sudo -S nvpmodel -m 2
#if [ $? -eq 0 ]; then
#  echo "Start NX max power done" >> /home/nx/rc_local_temp/bash_data/nx_sh.log
#fi
#echo "Cancel NX power limit" >> /home/nx/rc_local_temp/bash_data/nx_sh.log
#exho "123" | sudo -S jetson_clock --fan
#if [ $? -eq 0 ]; then
#  echo "Cancel NX power limit done" >> /home/nx/rc_local_temp/bash_data/nx_sh.log
#fi

echo "$(date) Autoaim process starting" >> /home/nx/rc_local_temp/bash_data/nx_sh.log
#echo "123" | sudo -S jetson_clocks --fan
#sleep 5s
#echo "123" | sudo sh -c "echo 200 > /sys/devices/pwm-fan/target_pwm"
#while true
#do
echo "123" | sudo -S chmod 777 /dev/ttyUSB0 | /home/nx/miniforge-pypy3/envs/rmcv/bin/python3 /home/nx/2022rmcv/rmcv_main.py | tee "../rc_local_temp/bash_data/$(date).txt" 2>&1
# 
#/home/nx/miniforge-pypy3/envs/rmcv/bin/python3 /home/nx/rmcv_nx/2022rmcvNX/yunft-2021rmcvpy-2022RMCV-/rmcv_main.py | tee "../rc_local_temp/bash_data/$(date).txt" 2>&1
if [ $? -eq 0 ]; then
  echo "Autoaim process starting succeed!" >> /home/nx/rc_local_temp/bash_data/nx_sh.log
else
  echo "Autoaim process starting failed!" >> /home/nx/rc_local_temp/bash_data/nx_sh.log
  echo "123" | sudo -S chmod 777 /dev/ttyUSB0 | /home/nx/miniforge-pypy3/envs/rmcv/bin/python3 /home/nx/2022rmcv/rmcv_main.py | tee "../rc_local_temp/bash_data/$(date).txt" 2>&1
fi
#sleep 10s
#done

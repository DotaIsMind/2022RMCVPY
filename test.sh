if [ $? -eq 0 ]; then
  echo "Autoaim process starting succeed!" >> /home/nx/rc_local_temp/bash_data/nx_sh.log
else
  echo "Autoaim process starting failed!" >> /home/nx/rc_local_temp/bash_data/nx_sh.log
  echo "123" | sudo -S chmod 777 /dev/ttyUSB0 | /home/nx/miniforge-pypy3/envs/rmcv/bin/python3 /home/nx/2022rmcv/rmcv_main.py | tee "../rc_local_temp/bash_data/$(date).txt" 2>&1
fi

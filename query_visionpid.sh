#/bin/bash
##echo "kill process provider"
while [ true ]; do
  PID=$(ps -ef | grep rmcv_main.py | grep -v grep | awk '{print $2}')
  if [ -z $PID ]; then
    echo "process rmcv not exist"
    cd /home/nx/2022rmcv
    echo "123" | sudo -S bash nx_run.sh
    sleep 10
  else
    echo "process rmcv id: $PID"
    sleep 2
  fi

done

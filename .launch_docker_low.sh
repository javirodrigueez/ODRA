docker run --gpus '"device=0,1"' -d -it \
  --name jrodriguez_reasoncbt \
  --volume "$(pwd):/ReasonCBT:rw" \
  --volume "/home/jrodriguez/emnlp26/ReasonCBT-outs:/outs:rw" \
  --shm-size=32gb \
  --memory=60gb \
  jrodriguez/synther bash

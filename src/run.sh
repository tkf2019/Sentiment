#!/bin/bash

rm ../logs/*.log

python main.py -m CNN -b 256 --mode train --hidden 256 --dropout 0.5 --patience 5 >>../logs/CNN.log
python main.py -m CNN -b 256 --mode test --hidden 256 >>../logs/CNN.log

python main.py -m RNN -b 64 --mode train --hidden 128 --dropout 0.5 --patience 5 >>../logs/RNN.log
python main.py -m RNN -b 64 --mode test --hidden 128 >>../logs/RNN.log

python main.py -m MLP -b 64 --mode train --hidden 1024 --dropout 0.3 --patience 5 >>../logs/MLP.log
python main.py -m MLP -b 64 --mode test --hidden 1024 >>../logs/MLP.log
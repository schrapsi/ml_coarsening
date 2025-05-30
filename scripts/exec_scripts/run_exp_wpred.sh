#!/bin/bash
cd ~ || exit
cd bachelor_thesis|| exit
spack env activate test
source venv/bin/activate
python main.py eval
spack env deactivate

echo "====================="
echo "graph predictions done"
echo "====================="

cd ~ || exit
cd mt-kahypar/build/ || exit
spack env activate kahypar
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make clean
make MtKaHyPar -j
spack env deactivate

echo "===================="
echo "building file done"
echo "===================="

cd ~ || exit
spack env activate test
source hypergraph_partitioner/env.sh
source bachelor_thesis/venv/bin/activate
cd hypergraph_partitioner/experiments/ || exit
python ~/hypergraph_partitioner/setup_experiments.py experiment.json -f
python ~/hypergraph_partitioner/experiments/execute_experiments.py experiment.json



#!/bin/bash
cd ..
# Number of executions
num_executions=10
total_execution_time_ns=0

for ((i=1; i<=$num_executions; i++)); do

    start_time=$(perl -MTime::HiRes -e 'printf("%.0f\n",Time::HiRes::time()*1000)')
    
    ./run_digamma.sh

    end_time=$(perl -MTime::HiRes -e 'printf("%.0f\n",Time::HiRes::time()*1000)')
   
    elapsed_time_ms=$((end_time - start_time))
    
    total_execution_time_ms=$((total_execution_time_ms + elapsed_time_ms))
done


average_execution_time_ms=$((total_execution_time_ms / num_executions))

# Display the total execution time in nanoseconds
echo "DiGamma -> Avg Execution Time: $average_execution_time_ms ms" >> "output.txt"

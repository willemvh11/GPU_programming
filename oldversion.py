#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:10:18 2018

@author: willemvh
"""

import argparse
import pyopencl as cl
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import os
import time




# Function to print devices and platforms available

def device_platforms():
    
    if args.GPU:
        for n in range(len(cl.get_platforms())):
            platform = cl.get_platforms()[n]
            for j in range(len(platform.get_devices(cl.device_type.GPU))):
                device = platform.get_devices(cl.device_type.GPU)[j]
                print('Platform', n, 'Device', j, ':', device)
    elif args.CPU:
        for n in range(len(cl.get_platforms())):
            platform = cl.get_platforms()[n]
            for j in range(len(platform.get_devices(cl.device_type.CPU))):
                device = platform.get_devices(cl.device_type.CPU)[j]
                print('Platform', n, 'Device', j, ':', device)
                
# Funtion to select a device      
          
def getdevice(b):
    if args.GPU:
        return platform.get_devices(cl.device_type.GPU)[b]
    elif args.CPU:
        return platform.get_devices(cl.device_type.CPU)[b]
    

## GPU/CPU set up
   #_____________
             
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument("--GPU", help="Processing Unit",action="store_true")
    group.add_argument("--CPU", help="Processing Unit",action="store_true")
    args = parser.parse_args()
    device_platforms()
    a = int(input('Input platform number:'))
    b = int(input('Input device number:'))
    
    ## Step #1 Obtain an OpenCL platform
    platform = cl.get_platforms()[a]
    
    ## Step #2 Obtain a device ID
    device = getdevice(b)
    
    ## Step #3 Create a context for the selected device
    context = cl.Context([device])
    print('Context created is:', context)
    q = str(input('Continue with program? [y/n] >>'))
    
    if q == 'n':
        sys.exit()
    
    start = time.time()
    
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    
    ## Initalise parameters
    #____________________________________________________________
    
    # Number of nuclear spins
    
    ni = 16
    
    # Set time step for differential equation solution
    
    dt = 0.1
    
    # Set total time that we are interested in
    
    tmax = 30
    
    # xmax calcuates total number of time steps
    
    xmax = int(tmax/dt)
    
    # iterations must be an even divisor of xmax
    
    iterations = 30
    
    size = int(xmax/iterations)
    
    # Set local size 
    
    local_size1 = 16
    local_size2 = 16
    
    # Ensures that global size can be evenly divided by local size
    
    if (ni%local_size1 == 0):
        global_size1 = ni
    else:
        global_size1 = (ni//local_size1 + 1)*local_size1
      
    a = global_size1
    
    # Global size in y dimension - 16384 is the max my macbook pro can do 
    
    global_size2 = 262144
    
    # global_sizetensors is used for final reduction in y dimension and summation over
    # monte carlo samples
    
    if (size%local_size1 == 0):
        global_sizetensors = size
    else:
        global_sizetensors = (size//local_size1 + 1)*local_size1

    
    
    # Set number of monte carlo steps
    
    mcs = 5
    
    
    # n1 is used in the 1st reduction kernel, n2 in the second
    
    n1 = math.log(local_size1, 2)
    
    n2 = math.log(local_size2, 2)
    
    # External field
    
    wi = np.zeros(3).astype(np.float32)
        
    wi[0] = 0
    wi[1] = 0
    wi[2] = 0.5
    
    wi_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=wi)
    
    # Hyperfine constants
    
    hyperfine = np.zeros(ni).astype(np.float32)
    
    hyperfine[0] =-0.999985
    
    hyperfine[1] =-0.7369246
    hyperfine[2] =0.511210
    hyperfine[3] =-0.0826998
    
    hyperfine[4] =0.0655341
    hyperfine[5] =-0.562082
    hyperfine[6] =-0.905911
    hyperfine[7] =0.357729
    hyperfine[8] =0.358593
    hyperfine[9] =0.869386
    hyperfine[10] =-0.232996
    hyperfine[11] =0.0388327
    hyperfine[12] =0.661931
    hyperfine[13] =-0.930856
    hyperfine[14] =-0.893077
    hyperfine[15] =0.0594001
    
    hyperfine_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=hyperfine)
    
    # Import kernels
    
    from kerneloldversion import kernelprecessnucspins
    from kerneloldversion import kernelprecesselecspins
    from kerneloldversion import kernelreduce
    from kerneloldversion import kernelvecbuilds
    from kerneloldversion import kernelvecbuildi
    from kerneloldversion import kerneltensors
    from kerneloldversion import kernelreduce2
    from kerneloldversion import kernelprep2
     
    # Create queue
    
    queue = cl.CommandQueue(context)
    
    # Compile programs
    
    program1 = cl.Program(context, kernelprecessnucspins(dt/2)).build()
    program2 = cl.Program(context, kernelprecesselecspins(dt)).build()
    program3 = cl.Program(context, kernelvecbuilds()).build()
    program4 = cl.Program(context, kernelreduce()).build()
    program5 = cl.Program(context, kerneltensors()).build()
    program6 = cl.Program(context, kernelreduce2()).build()
    program7 = cl.Program(context, kernelprep2()).build()
    program8 = cl.Program(context, kernelvecbuildi()).build()
    
    # Define vectors/Buffers
    
    Rxx = np.zeros(xmax).astype(np.float32)
    Rxy = np.zeros(xmax).astype(np.float32)
    Rzz = np.zeros(xmax).astype(np.float32)
    final_Rxx = np.zeros(xmax).astype(np.float32)
    final_Rxy = np.zeros(xmax).astype(np.float32)
    final_Rzz = np.zeros(xmax).astype(np.float32)
    
    Rxx_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=Rxx)
    Rxy_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=Rxy)
    Rzz_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=Rzz)
    
    n_spins = np.zeros(ni*3*global_size2).astype(np.float32)
    s_spin = np.zeros(3*global_size2).astype(np.float32)
    
    w = np.zeros((a*3*global_size2)).astype(np.float32)
    v = np.zeros((ni+1)*2*global_size2).astype(np.float32)
    
    n_spins_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=n_spins)
    s_spin_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=s_spin)
        
        
    w_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=w)
    
    v_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=v)
    
     
    # These buffers are necessary as they allow me to vary local size - 
    # as you are not allowed to have a variable sized vector inside a kernel
    
    wtemp_buf = cl.LocalMemory((3*local_size1*local_size2*sys.getsizeof(np.float32(1))))
    
    sstoretemp_buf = cl.LocalMemory((3*local_size1*local_size2*sys.getsizeof(np.float32(1))))
    
    sloc_buf = cl.LocalMemory((3*local_size2*sys.getsizeof(np.float32(1))))
    
    wloc_buf = cl.LocalMemory((3*local_size2*sys.getsizeof(np.float32(1))))
    
    iloc_buf = cl.LocalMemory((3*local_size1*local_size2*sys.getsizeof(np.float32(1))))
    
    iloc2_buf = cl.LocalMemory((3*local_size1*local_size2*sys.getsizeof(np.float32(1))))
    
    store_buf = cl.LocalMemory(local_size1*sys.getsizeof(np.float32(1)))
    
    vloc_buf = cl.LocalMemory(2*local_size1*local_size2*sys.getsizeof(np.float32(1)))
    
    outputloc_buf = cl.LocalMemory((3*local_size1*local_size2*sys.getsizeof(np.float32(1))))
    
    sstoreloc_buf = cl.LocalMemory((3*local_size1*local_size2*sys.getsizeof(np.float32(1))))
    
    sloc2_buf = cl.LocalMemory((3*local_size2*sys.getsizeof(np.float32(1))))
    
    # sstore saves all the spin vectors for one mcs iteration to minimise
    # copying from GPU to CPU
    
    sstore = np.zeros(size*3*global_size2).astype(np.float32)
    sstore_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE, sstore.nbytes)
    
    sinit = np.zeros(3*global_size2).astype(np.float32)
    sinit_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE, sinit.nbytes)
    
    # output is just a copy of sstore as if you use sstore as both an input 
    # and output in the second reduction it doesn't work if you oversaturate the GPU
    
    output = np.zeros(global_sizetensors*3*global_size2).astype(np.float32)
    output_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE, output.nbytes)
    
    for u in range(mcs):
        
        # Generate array of random numbers
        
        v = np.random.uniform(size = (ni+1)*2*global_size2).astype(np.float32)
        
        # Copy random numbers to the kernel
        
        cl.enqueue_copy(queue, v_buf, v)
        
        # Build initial electron and nuclear spin vectors
        
        program3.vecbuilds(queue, (global_size2,), (local_size2,), s_spin_buf, v_buf, np.int32(ni), sloc_buf, sinit_buf)
        program8.vecbuildi(queue, (global_size1, global_size2), (local_size1, local_size2), n_spins_buf, v_buf, np.int32(ni), vloc_buf, iloc2_buf)

       
        # Iterate over timesteps
        for j in range(iterations):
            
            for x in range(0,size):
    
               
                # Reset a at the beginning of each iteration - v important otherwise a=1 after 1st step
                
                a = global_size1
                
                program1.precessnucspins(queue, (global_size1, global_size2), (local_size1, local_size2), n_spins_buf, hyperfine_buf, s_spin_buf, np.int32(ni), iloc_buf)
                
                # Reduction in the x direction (summing together nuclear spin vectors)
                
                while (a > 1):
                   program4.reduce(queue, (global_size1, global_size2), (local_size1, local_size2), n_spins_buf, w_buf, np.int32(n1), np.int32(a), wtemp_buf, hyperfine_buf, np.int32(ni), store_buf)
                   a = a//local_size1 
                   
                program2.precesselecspins(queue, (global_size2,), (local_size2,), w_buf, wi_buf, s_spin_buf, np.int32(size), np.int32(x), sstore_buf, sloc2_buf, wloc_buf, np.int32(global_size1))
                
                program1.precessnucspins(queue, (global_size1, global_size2), (local_size1, local_size2), n_spins_buf, hyperfine_buf, s_spin_buf, np.int32(ni), iloc_buf)
    
            # Prepare sstore for Rxx, Rxy, Rzz calculation
            
            program7.prep2(queue, (global_sizetensors, global_size2), (local_size1, local_size2), sstore_buf, output_buf, np.int32(size), sstoreloc_buf, outputloc_buf, sinit_buf)
            
            #  Reset b between each monte carlo step
            
            b = global_size2
            
            # Reduction in the y direction (over different monte carlo steps running in parallel)
            # note that global size in the x direction is now related to xmax (no longer ni)
            
            while (b > 1):
                
                program6.reduce2(queue, (global_sizetensors, global_size2), (local_size1, local_size2), np.int32(n2), np.int32(b), sstoretemp_buf, output_buf)
                b = b//local_size2
                
            # Sum Rxx, Rxy, Rzz over different monte carlo step iterations - note that this is
            # now a 1D workgroup size
                
            program5.tensors(queue, (global_sizetensors,), (local_size1,), output_buf, Rxx_buf, Rxy_buf, Rzz_buf, np.int32(size), np.int32(j))
  
            
    cl.enqueue_copy(queue, Rxx, Rxx_buf)
    cl.enqueue_copy(queue, Rxy, Rxy_buf)
    cl.enqueue_copy(queue, Rzz, Rzz_buf)  
    
    final_Rxx = 2*Rxx/(mcs*global_size2)
    final_Rxy = 2*Rxy/(mcs*global_size2)
    final_Rzz = 2*Rzz/(mcs*global_size2)
    
    # Counting time
    
    times = np.zeros(xmax).astype(np.float32)
    t = 0
    for p in range(xmax):
        times[p] = t
        t += dt
     
    end = time.time()
    print(end - start)
    # Plot results
    
    plt.plot(times, final_Rxx)
    plt.plot(times, final_Rxy)
    plt.plot(times, final_Rzz)
    
    
    plt.legend(['Rxx', 'Rxy', 'Rzz'], loc='upper left')
    plt.show()
    
    

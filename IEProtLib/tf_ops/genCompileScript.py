'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file genCompileScript.py

    \brief File to create the compile script.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import argparse
import os
import tensorflow as tf

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate the compile script for the MCCNN++ operations.')
    parser.add_argument('--cudaFolder', required=True, help='Path to the CUDA folder')
    parser.add_argument('--debugInfo', action='store_true', help='Print debug information during execution (default: False)')
    parser.add_argument('--nvccCompileInfo', action='store_true', help='Print debug information during execution (default: False)')
    args = parser.parse_args()

    debugString = " -DDEBUG_INFO " if args.debugInfo else ""
    nvccCompileInfo = " -Xptxas=\"-v\" " if args.nvccCompileInfo else ""

    if not os.path.exists("build"): os.mkdir("build")
    
    with open("compile.sh", "w") as myCompileScript:
        #Clean the previous compiled files.
        myCompileScript.write("rm build/*\n")
        #Compile the cuda kernels.
        myCompileScript.write(args.cudaFolder+"/bin/nvcc"+debugString+" -std=c++11 "+
            nvccCompileInfo+" cu/src/compute_keys.cu -o build/compute_keys.cu.o "+
            "-Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")
        myCompileScript.write(args.cudaFolder+"/bin/nvcc"+debugString+" -std=c++11 "+
            nvccCompileInfo+" cu/src/build_grid_ds.cu -o build/build_grid_ds.cu.o "+
            "-Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")
        myCompileScript.write(args.cudaFolder+"/bin/nvcc"+debugString+" -std=c++11 "+
            nvccCompileInfo+" cu/src/find_ranges_grid_ds.cu -o build/find_ranges_grid_ds.cu.o "+
            "-Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")
        myCompileScript.write(args.cudaFolder+"/bin/nvcc"+debugString+" -std=c++11 "+
            nvccCompileInfo+" cu/src/count_neighbors.cu -o build/count_neighbors.cu.o "+
            "-Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")
        myCompileScript.write(args.cudaFolder+"/bin/nvcc"+debugString+" -std=c++11 "+
            nvccCompileInfo+" cu/src/elem_wise_min.cu -o build/elem_wise_min.cu.o "+
            "-Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")
        myCompileScript.write(args.cudaFolder+"/bin/nvcc"+debugString+" -std=c++11 "+
            nvccCompileInfo+" cu/src/scan_alg.cu -o build/scan_alg.cu.o "+
            "-Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")
        myCompileScript.write(args.cudaFolder+"/bin/nvcc"+debugString+" -std=c++11 "+
            nvccCompileInfo+" cu/src/store_neighbors.cu -o build/store_neighbors.cu.o "+
            "-Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")
        
        #BASIS FUNCTIONS OPERATIONS
        myCompileScript.write(args.cudaFolder+"/bin/nvcc"+debugString+" -std=c++11 "+
            nvccCompileInfo+" cu/src/basis/basis_utils.cu -o build/basis_utils.cu.o "+
            "-Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")
        myCompileScript.write(args.cudaFolder+"/bin/nvcc"+debugString+" -std=c++11 "+
            nvccCompileInfo+" cu/src/basis/basis_proj.cu -o build/basis_proj.cu.o "+
            "-Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")
        myCompileScript.write(args.cudaFolder+"/bin/nvcc"+debugString+" -std=c++11 "+
            nvccCompileInfo+" cu/src/basis/basis_proj_grads.cu -o build/basis_proj_grads.cu.o "+
            "-Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")
        myCompileScript.write(args.cudaFolder+"/bin/nvcc"+debugString+" -std=c++11 "+
            nvccCompileInfo+" cu/src/basis/basis_hproj_bilateral.cu -o build/basis_hproj_bilateral.cu.o "+
            "-Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")

        #GRAPHS
        myCompileScript.write(args.cudaFolder+"/bin/nvcc"+debugString+" -std=c++11 "+
            nvccCompileInfo+" cu/src/graph_aggregation.cu -o build/graph_aggregation.cu.o "+
            "-Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")
        myCompileScript.write(args.cudaFolder+"/bin/nvcc"+debugString+" -std=c++11 "+
            nvccCompileInfo+" cu/src/compute_topo_dist.cu -o build/compute_topo_dist.cu.o "+
            "-Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")
        myCompileScript.write(args.cudaFolder+"/bin/nvcc"+debugString+" -std=c++11 "+
            nvccCompileInfo+" cu/src/protein_pooling.cu -o build/protein_pooling.cu.o "+
            "-Icu/header -Icc/header -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")

        #Compile the library
        tensorflowInclude = tf.sysconfig.get_include()
        tensorflowLib = tf.sysconfig.get_lib()
        myCompileScript.write(
            "g++ -std=c++11"+debugString+

            " build/compute_keys.cu.o "+
            " build/build_grid_ds.cu.o "+
            " build/find_ranges_grid_ds.cu.o "+
            " build/count_neighbors.cu.o "+
            " build/elem_wise_min.cu.o "+
            " build/scan_alg.cu.o "+
            " build/store_neighbors.cu.o "+
            " build/graph_aggregation.cu.o "+
            " build/compute_topo_dist.cu.o "+
            " build/protein_pooling.cu.o "+

            " build/basis_utils.cu.o "+
            " build/basis_proj.cu.o "+
            " build/basis_proj_grads.cu.o "+
            " build/basis_hproj_bilateral.cu.o "+

            " cc/src/tf_gpu_device.cpp "+
            " cc/src/compute_keys.cpp "+
            " cc/src/build_grid_ds.cpp "+
            " cc/src/find_neighbors.cpp "+
            " cc/src/basis_proj_bilateral.cpp "+
            " cc/src/graph_aggregation.cpp "+
            " cc/src/compute_topo_dist.cpp "+
            " cc/src/protein_pooling.cpp "+

            "-o build/IEProtLib.so -shared -fPIC "+
            "-Icc/header "+
            "-Icu/header "+
            " ".join(tf.sysconfig.get_compile_flags())+
            " -I"+args.cudaFolder+"/include "+
            "-lcudart -L "+args.cudaFolder+"/lib64/ "+
            " ".join(tf.sysconfig.get_link_flags())+
            " -O2\n")

